# -*- coding: utf-8 -*-
"""
组合探索：BitNet 三元量化 × SpikingBrain 脉冲化。

思路：
  1. BitNet 量化权重：三元 {-1,0,+1} + int2 打包（convert_checkpoint.py 的 quant_weight_int8）
  2. SpikingBrain 脉冲化激活：dynamic_spikes（激活 → 稀疏脉冲计数）
  3. 联合稀疏度 = 权重稀疏 + 激活稀疏 → 功耗/算力降低潜力

功耗降低机制（SpikingBrain）：
  - 激活稀疏：跳过零激活的 MAC
  - 权重稀疏：跳过零权重的 MAC
  - 联合跳过率 = 1 - (1-激活稀疏)*(1-权重稀疏)
  - int2 权重：内存带宽降 16x
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.quant_linear import dynamic_spikes
from W8ASpike.Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode

MODEL_PATH = "./models/Qwen3-0.6B"
K = 4.0
EVAL_TEXTS = [
    "Spiking neural networks are inspired by the brain.",
    "The future of artificial intelligence is bright.",
    "脉冲神经网络是一种受大脑启发的计算模型。",
    "Machine learning models can learn from large amounts of data.",
    "深度学习在自然语言处理领域取得了巨大成功。",
    "The solar system consists of the sun and the planets that orbit it.",
    "Volcanoes erupt when molten rock rises from deep within the Earth.",
    "The human brain contains billions of neurons that communicate with each other.",
    "Electric cars are becoming increasingly popular around the world.",
    "The Nile River is the longest river in Africa and flows through many countries.",
]
PROMPTS = [
    "Spiking neural networks are",
    "The future of artificial intelligence",
    "脉冲神经网络是一种",
    "Machine learning models can",
    "深度学习在自然语言处理中",
]


class BitNetSpikeLinear(nn.Module):
    """BitNet 三元权重 + SpikingBrain 脉冲激活。"""

    def __init__(self, in_features, out_features, bias=False, k=K):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        # BitNet 三元量化缓存
        self.register_buffer("w_ternary", torch.zeros(out_features, in_features))
        self.register_buffer("alpha", torch.ones(out_features))

    def quantize_weights(self):
        """BitNet 三元量化：s=1/mean|W|，W_ternary=round(W*s).clamp(-1,1)。"""
        s = 1.0 / self.weight.abs().mean().clamp_(min=1e-5)
        w_ternary = (self.weight * s).round().clamp(-1, 1)
        self.w_ternary.copy_(w_ternary)
        self.alpha.copy_(torch.full((self.out_features,), (1.0 / s).item()))

    def forward(self, x):
        # SpikingBrain 脉冲化激活
        vth = x.abs().mean([-1], keepdim=True).float() / self.k
        vth = vth.clamp(min=1e-5, max=1e4)
        spikes_int = (x / vth).round()
        spikes_int = spike_fake_quant(spikes_int, SpikeCountBitwiseNode(is_bidirectional=True))
        x_spike = (spikes_int * vth).to(x.dtype)
        # BitNet 三元权重 matmul
        out = nn.functional.linear(x_spike, self.w_ternary, None)
        out = out * self.alpha.unsqueeze(0)
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_attn(model, k=K):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "self_attn" in name:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = BitNetSpikeLinear(module.in_features, module.out_features,
                                    bias=module.bias is not None, k=k)
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            new.quantize_weights()
            setattr(parent, attr, new)
            count += 1
    model.eval()
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
    """测量 激活稀疏 + 权重稀疏 + 联合跳过率。"""
    act_sparsities = []

    def hook(module, input, output):
        x = input[0]
        vth = x.abs().mean([-1], keepdim=True).float() / module.k
        vth = vth.clamp(min=1e-5, max=1e4)
        spikes_int = (x / vth).round()
        act_sparsities.append((spikes_int == 0).float().mean().item())

    handles = [m.register_forward_hook(hook)
               for m in model.modules() if isinstance(m, BitNetSpikeLinear)]
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    act_sp = sum(act_sparsities) / len(act_sparsities) if act_sparsities else 0.0

    # 权重稀疏
    w_zeros, w_total = 0, 0
    for m in model.modules():
        if isinstance(m, BitNetSpikeLinear):
            w_zeros += (m.w_ternary == 0).sum().item()
            w_total += m.w_ternary.numel()
    w_sp = w_zeros / w_total if w_total else 0.0

    # 联合跳过率（功耗降低潜力）
    combined = 1 - (1 - act_sp) * (1 - w_sp)
    return act_sp, w_sp, combined


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=32)

    # 原始模型 PPL
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    ppl_ref = compute_perplexity(ref, tokenizer, EVAL_TEXTS)
    del ref

    # 组合模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    n = convert_attn(model, k=K)
    ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS)
    act_sp, w_sp, combined = measure_sparsity(model, inputs)

    print(f"转换 {n} 个注意力层为 BitNet三元+SpikingBrain脉冲")
    print(f"\n=== 精度 ===")
    print(f"  原始 float32 PPL = {ppl_ref:.2f}")
    print(f"  组合模型 PPL     = {ppl:.2f}")
    print(f"\n=== 稀疏度（功耗降低潜力）===")
    print(f"  激活稀疏度 = {act_sp*100:.1f}%")
    print(f"  权重稀疏度 = {w_sp*100:.1f}%")
    print(f"  联合跳过率 = {combined*100:.1f}%  (1-(1-{act_sp*100:.0f}%)*(1-{w_sp*100:.0f}%))")
    print(f"  int2 权重内存降低 = 16x")


if __name__ == "__main__":
    main()
