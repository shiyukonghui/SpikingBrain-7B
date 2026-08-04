# -*- coding: utf-8 -*-
"""
把 AVX2 LUT 核集成到组合方案（BitNet 三元 + SpikingBrain 脉冲）的完整前向。
对比：稠密组合 vs LUT 组合 的端到端前向时间与精度。
"""
import ctypes
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode

MODEL_PATH = "./models/Qwen3-0.6B"
K = 4.0
EVAL_TEXTS = [
    "Spiking neural networks are inspired by the brain.",
    "The future of artificial intelligence is bright.",
    "脉冲神经网络是一种受大脑启发的计算模型。",
    "Machine learning models can learn from large amounts of data.",
    "深度学习在自然语言处理领域取得了巨大成功。",
]
PROMPTS = [
    "Spiking neural networks are",
    "The future of artificial intelligence",
    "脉冲神经网络是一种",
    "Machine learning models can",
    "深度学习在自然语言处理中",
]

# 加载 AVX2 LUT 核
LIB = os.path.join("rust_lut", "target", "release", "rust_lut.dll")
lib = ctypes.CDLL(LIB)
lib.lut_matmul_avx2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.lut_matmul_avx2.restype = None


def spike_activation(x, k=K):
    """SpikingBrain 脉冲化激活。"""
    vth = x.abs().mean([-1], keepdim=True).float() / k
    vth = vth.clamp(min=1e-5, max=1e4)
    spikes_int = (x / vth).round()
    spikes_int = spike_fake_quant(spikes_int, SpikeCountBitwiseNode(is_bidirectional=True))
    return (spikes_int * vth).to(x.dtype)


def pack_ternary_2bit(w_ternary):
    """三元权重 -> 2-bit 码，返回 (n_blocks, n) 布局（AVX2 核要求）。"""
    n, k = w_ternary.shape
    codes = torch.where(w_ternary == -1, 0, torch.where(w_ternary == 0, 1, 2)).to(torch.int64)
    codes = codes.reshape(n, -1, 4)
    packed = torch.zeros(n, k // 4, dtype=torch.uint8, device=w_ternary.device)
    for i in range(4):
        packed |= (codes[:, :, i] << (2 * i)).to(torch.uint8)
    return packed.t().contiguous()  # (k/4, n)


class BitNetSpikeDenseLinear(nn.Module):
    """稠密组合：BitNet 三元权重 + SpikingBrain 脉冲（F.linear）。"""

    def __init__(self, in_features, out_features, bias=False, k=K):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.register_buffer("w_ternary", torch.zeros(out_features, in_features))
        self.register_buffer("alpha", torch.ones(out_features))

    def quantize_weights(self):
        s = 1.0 / self.weight.abs().mean().clamp_(min=1e-5)
        self.w_ternary.copy_((self.weight * s).round().clamp(-1, 1))
        self.alpha.copy_(torch.full((self.out_features,), (1.0 / s).item()))

    def forward(self, x):
        x_spike = spike_activation(x, self.k)
        out = nn.functional.linear(x_spike, self.w_ternary, None)
        out = out * self.alpha.unsqueeze(0)
        if self.bias is not None:
            out = out + self.bias
        return out


class BitNetSpikeLUTLinear(nn.Module):
    """LUT 组合：BitNet 三元权重 + SpikingBrain 脉冲 + AVX2 LUT matmul。"""

    def __init__(self, in_features, out_features, bias=False, k=K):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.register_buffer("w_code", torch.zeros(in_features // 4, out_features, dtype=torch.uint8))
        self.register_buffer("alpha", torch.ones(out_features))

    def quantize_weights(self):
        s = 1.0 / self.weight.abs().mean().clamp_(min=1e-5)
        w_ternary = (self.weight * s).round().clamp(-1, 1)
        self.w_code.copy_(pack_ternary_2bit(w_ternary))
        self.alpha.copy_(torch.full((self.out_features,), (1.0 / s).item()))

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)  # (m, k)
        x_spike = spike_activation(x, self.k)
        # 量化到 int8
        scale = 127.0 / x_spike.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x_spike * scale).round().clamp(-128, 127).to(torch.int8)
        # AVX2 LUT matmul
        m, k = x_q.shape
        n = self.out_features
        out = torch.zeros(m, n, dtype=torch.float32)
        lib.lut_matmul_avx2(x_q.data_ptr(), self.w_code.data_ptr(), out.data_ptr(), m, k, n)
        # 反量化 + 权重 scale
        out = out / scale * self.alpha.unsqueeze(0)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], n)


def convert_attn(model, cls, k=K):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "self_attn" in name:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = cls(module.in_features, module.out_features,
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


def measure_forward(model, inputs, n=5):
    with torch.no_grad():
        model(**inputs)
    wall = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        wall.append(time.perf_counter() - t0)
    return sum(wall) / n


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=32)

    # 稠密组合
    model_dense = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    n_dense = convert_attn(model_dense, BitNetSpikeDenseLinear, k=K)
    ppl_dense = compute_perplexity(model_dense, tokenizer, EVAL_TEXTS)
    t_dense = measure_forward(model_dense, inputs)
    del model_dense

    # LUT 组合
    model_lut = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    n_lut = convert_attn(model_lut, BitNetSpikeLUTLinear, k=K)
    ppl_lut = compute_perplexity(model_lut, tokenizer, EVAL_TEXTS)
    t_lut = measure_forward(model_lut, inputs)

    print(f"转换 {n_dense} 个注意力层")
    print(f"\n{'指标':<16}{'稠密组合':>14}{'LUT组合':>14}{'加速':>10}")
    print("-" * 54)
    print(f"{'端到端前向':<16}{t_dense*1000:>12.1f}ms{t_lut*1000:>12.1f}ms{t_dense/t_lut:>9.2f}x")
    print(f"{'留出PPL':<16}{ppl_dense:>14.2f}{ppl_lut:>14.2f}{'':>10}")


if __name__ == "__main__":
    main()
