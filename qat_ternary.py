# -*- coding: utf-8 -*-
"""
三元权重 {-1,0,+1} × 脉冲激活 的量化感知微调（QAT）。
BitNet 洞察：三元权重需训练。用 STE 同时量化权重（三元）与激活（脉冲），
微调注意力层，验证能否在保持稀疏性的同时提升精度。
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode

MODEL_PATH = "./models/Qwen3-0.6B"
K = 4.0

# 训练语料（微调用）
TRAIN_TEXTS = [
    "Spiking neural networks are inspired by the brain and mimic biological computation.",
    "The future of artificial intelligence depends on energy efficient computing.",
    "脉冲神经网络是一种受大脑启发的计算模型，具有低功耗优势。",
    "Machine learning models can learn from large amounts of data efficiently.",
    "深度学习在自然语言处理领域取得了巨大成功，推动了智能应用的发展。",
    "Quantization reduces model size while preserving most of the accuracy.",
    "Efficient inference is critical for deploying models on edge devices.",
    "神经形态计算为下一代低功耗人工智能提供了新的方向。",
    "Ternary weights enable fast sparse computation in neural networks.",
    "The brain computes with spikes, which are sparse and energy efficient.",
]
# 留出评估语料（微调未见过的句子，评估泛化）
EVAL_TEXTS = [
    "Spiking neural networks are inspired by the brain.",
    "The future of artificial intelligence is bright.",
    "脉冲神经网络是一种受大脑启发的计算模型。",
    "Machine learning models can learn from large amounts of data.",
    "深度学习在自然语言处理领域取得了巨大成功。",
]


class TernarySpikeLinearSTE(nn.Module):
    """三元权重 + 脉冲激活 可微线性层（STE）。"""
    # 记录激活稀疏度（测量用）
    act_sparsities = []

    def __init__(self, in_features, out_features, bias=False, k=K):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weight(self):
        """三元量化：alpha=mean(|W|)，W_ternary=round(W/alpha).clamp(-1,1)。"""
        alpha = self.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        w_ternary = (self.weight / alpha).round().clamp(-1, 1)
        return w_ternary, alpha

    def forward(self, x):
        # 三元权重（STE）
        w_ternary, alpha = self.quantize_weight()
        w_q = w_ternary * alpha  # 反量化权重
        w_ste = self.weight + (w_q - self.weight).detach()

        # 脉冲激活（STE）
        vth = x.abs().mean([-1], keepdim=True).float() / self.k
        vth = vth.clamp(min=1e-5, max=1e4)
        spikes_int = (x / vth).round()
        spikes_int = spike_fake_quant(spikes_int, SpikeCountBitwiseNode(is_bidirectional=True))
        TernarySpikeLinearSTE.act_sparsities.append(
            (spikes_int == 0).float().mean().item())
        x_spike = spikes_int * vth
        x_ste = x + (x_spike - x).detach()

        return nn.functional.linear(x_ste, w_ste, self.bias)


def convert_attn_ternary(model, k=K):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "self_attn" in name:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = TernarySpikeLinearSTE(module.in_features, module.out_features,
                                        bias=module.bias is not None, k=k)
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(parent, attr, new)
            count += 1
    return count


def compute_perplexity(model, tokenizer, texts):
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = input_ids[:, 1:]
        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="sum")
        total_nll += loss.item()
        total_tokens += shift_labels.numel()
    return math.exp(total_nll / total_tokens)


def measure_sparsity(model, inputs):
    """激活稀疏度 + 三元权重稀疏度。"""
    TernarySpikeLinearSTE.act_sparsities = []
    with torch.no_grad():
        model(**inputs)
    act_sp = (sum(TernarySpikeLinearSTE.act_sparsities)
              / len(TernarySpikeLinearSTE.act_sparsities)
              if TernarySpikeLinearSTE.act_sparsities else 0.0)

    # 权重稀疏度
    w_zeros, w_total = 0, 0
    for m in model.modules():
        if isinstance(m, TernarySpikeLinearSTE):
            w_ternary, _ = m.quantize_weight()
            w_zeros += (w_ternary == 0).sum().item()
            w_total += w_ternary.numel()
    w_sp = w_zeros / w_total if w_total else 0.0
    return act_sp, w_sp


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True)
    n = convert_attn_ternary(model, k=K)
    print(f"转换 {n} 个注意力层为 三元权重+脉冲 层")

    # 冻结非注意力层
    for name, p in model.named_parameters():
        if "self_attn" not in name:
            p.requires_grad = False

    # 训练数据
    enc = tokenizer(TRAIN_TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    # 评估输入
    eval_inputs = tokenizer(EVAL_TEXTS, return_tensors="pt", padding=True,
                            truncation=True, max_length=32)

    ppl_train_before = compute_perplexity(model, tokenizer, TRAIN_TEXTS)
    ppl_eval_before = compute_perplexity(model, tokenizer, EVAL_TEXTS)
    act_sp_before, w_sp_before = measure_sparsity(model, eval_inputs)
    print(f"微调前: 训练PPL={ppl_train_before:.2f} 留出PPL={ppl_eval_before:.2f} "
          f"激活稀疏={act_sp_before*100:.1f}% 权重稀疏={w_sp_before*100:.1f}%")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    steps = 60
    for step in range(steps):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if (step + 1) % 20 == 0:
            print(f"  step {step+1}: loss={loss.item():.4f}")

    model.eval()
    ppl_train_after = compute_perplexity(model, tokenizer, TRAIN_TEXTS)
    ppl_eval_after = compute_perplexity(model, tokenizer, EVAL_TEXTS)
    act_sp_after, w_sp_after = measure_sparsity(model, eval_inputs)
    print(f"微调后: 训练PPL={ppl_train_after:.2f} 留出PPL={ppl_eval_after:.2f} "
          f"激活稀疏={act_sp_after*100:.1f}% 权重稀疏={w_sp_after*100:.1f}%")
    print(f"\n留出PPL变化: {ppl_eval_before:.2f} -> {ppl_eval_after:.2f} "
          f"({(ppl_eval_after/ppl_eval_before-1)*100:+.1f}%)")
    print(f"激活稀疏: {act_sp_before*100:.1f}% -> {act_sp_after*100:.1f}%")
    print(f"权重稀疏: {w_sp_before*100:.1f}% -> {w_sp_after*100:.1f}%")


if __name__ == "__main__":
    main()
