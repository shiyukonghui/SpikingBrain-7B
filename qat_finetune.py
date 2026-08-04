# -*- coding: utf-8 -*-
"""
方向6探索：量化感知微调（QAT）。
BitNet 洞察：低位/脉冲化需训练。用 STE（直通估计器）让注意力层在脉冲化下微调，
验证能否提升脉冲模型精度。
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode
from convert_qwen3_spike import set_weight_scales

MODEL_PATH = "./models/Qwen3-0.6B"
K = 4.0
PPL_TEXTS = [
    "Spiking neural networks are inspired by the brain.",
    "The future of artificial intelligence is bright.",
    "脉冲神经网络是一种受大脑启发的计算模型。",
    "Machine learning models can learn from large amounts of data.",
    "深度学习在自然语言处理领域取得了巨大成功。",
]
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


class SpikeLinearSTE(nn.Module):
    """可微脉冲线性层（STE：前向量化，反向直通）。"""

    def __init__(self, in_features, out_features, bias=False, k=K):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # 脉冲化激活（STE）
        vth = x.abs().mean([-1], keepdim=True).float() / self.k
        vth = vth.clamp(min=1e-5, max=1e4)
        spikes_int = (x / vth).round()
        spikes_int = spike_fake_quant(spikes_int, SpikeCountBitwiseNode(is_bidirectional=True))
        x_spike = spikes_int * vth
        # STE：前向用量化值，反向梯度直通
        x_ste = x + (x_spike - x).detach()
        return nn.functional.linear(x_ste, self.weight, self.bias)


def convert_attn_ste(model, k=K):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "self_attn" in name:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = SpikeLinearSTE(module.in_features, module.out_features,
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


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True)
    n = convert_attn_ste(model, k=K)
    print(f"转换 {n} 个注意力层为可微脉冲层")

    # 冻结非注意力层
    for name, p in model.named_parameters():
        if "self_attn" not in name:
            p.requires_grad = False

    # 训练数据
    enc = tokenizer(TRAIN_TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    ppl_before = compute_perplexity(model, tokenizer, PPL_TEXTS)
    print(f"微调前 PPL = {ppl_before:.2f}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    steps = 30
    for step in range(steps):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"  step {step+1}: loss={loss.item():.4f}")

    model.eval()
    ppl_after = compute_perplexity(model, tokenizer, PPL_TEXTS)
    print(f"微调后 PPL = {ppl_after:.2f}  (变化 {(ppl_after/ppl_before-1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
