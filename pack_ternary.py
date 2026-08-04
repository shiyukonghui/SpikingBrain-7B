# -*- coding: utf-8 -*-
"""
将三元权重真正存储为 int8 / int2 打包格式，验证内存占用降低。
流程：QAT 微调（三元+脉冲）-> 打包存储 -> 测量内存 + 验证精度。
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
EVAL_TEXTS = [
    "Spiking neural networks are inspired by the brain.",
    "The future of artificial intelligence is bright.",
    "脉冲神经网络是一种受大脑启发的计算模型。",
    "Machine learning models can learn from large amounts of data.",
    "深度学习在自然语言处理领域取得了巨大成功。",
]


class TernarySpikeLinearSTE(nn.Module):
    """三元权重 + 脉冲激活（STE），用于 QAT 微调。"""

    def __init__(self, in_features, out_features, bias=False, k=K):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weight(self):
        alpha = self.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        w_ternary = (self.weight / alpha).round().clamp(-1, 1)
        return w_ternary * alpha

    def forward(self, x):
        w_q = self.quantize_weight()
        w_ste = self.weight + (w_q - self.weight).detach()
        vth = x.abs().mean([-1], keepdim=True).float() / self.k
        vth = vth.clamp(min=1e-5, max=1e4)
        spikes_int = (x / vth).round()
        spikes_int = spike_fake_quant(spikes_int, SpikeCountBitwiseNode(is_bidirectional=True))
        x_ste = x + (spikes_int * vth - x).detach()
        return nn.functional.linear(x_ste, w_ste, self.bias)


class PackedTernaryLinear(nn.Module):
    """三元权重打包存储（int8 或 int2），前向解包计算。"""

    def __init__(self, in_features, out_features, bias=False, pack_bits=8, k=K):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pack_bits = pack_bits
        self.k = k
        if pack_bits == 8:
            self.register_buffer("w_packed", torch.zeros(out_features, in_features, dtype=torch.int8))
        elif pack_bits == 2:
            n_packed = (in_features + 3) // 4
            self.register_buffer("w_packed", torch.zeros(out_features, n_packed, dtype=torch.uint8))
        else:
            raise ValueError(pack_bits)
        self.register_buffer("alpha", torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def pack(self, w_ternary):
        """w_ternary: {-1,0,1} float -> 打包存储。"""
        if self.pack_bits == 8:
            self.w_packed.copy_(w_ternary.to(torch.int8))
        else:
            codes = (w_ternary + 1).to(torch.int64)  # -1->0, 0->1, 1->2
            pad = (-self.in_features) % 4
            if pad:
                codes = torch.cat([codes, torch.zeros(codes.shape[0], pad, dtype=torch.int64,
                                                      device=codes.device)], dim=1)
            codes = codes.reshape(codes.shape[0], -1, 4)
            packed = (codes[:, :, 0] | (codes[:, :, 1] << 2)
                      | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)).to(torch.uint8)
            self.w_packed.copy_(packed)

    def unpack(self):
        if self.pack_bits == 8:
            w = self.w_packed.float()
        else:
            p = self.w_packed.to(torch.int64)
            c0 = p & 3
            c1 = (p >> 2) & 3
            c2 = (p >> 4) & 3
            c3 = (p >> 6) & 3
            codes = torch.stack([c0, c1, c2, c3], dim=-1).reshape(self.out_features, -1)
            w = (codes[:, :self.in_features] - 1).float()
        return w * self.alpha.unsqueeze(1)

    def forward(self, x):
        w = self.unpack()
        return nn.functional.linear(x, w, self.bias)


def convert_attn(model, cls, **kw):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "self_attn" in name:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = cls(module.in_features, module.out_features,
                      bias=module.bias is not None, **kw)
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


def fine_tune(model, tokenizer, steps=60):
    for name, p in model.named_parameters():
        if "self_attn" not in name:
            p.requires_grad = False
    enc = tokenizer(TRAIN_TEXTS, return_tensors="pt", padding=True,
                    truncation=True, max_length=64)
    input_ids, labels = enc["input_ids"], enc["input_ids"].clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    model.eval()


def attn_weight_memory(model):
    """注意力层权重内存（字节）。"""
    total = 0
    for m in model.modules():
        if isinstance(m, (TernarySpikeLinearSTE, PackedTernaryLinear)):
            total += m.weight.numel() * m.weight.element_size() if hasattr(m, "weight") else 0
    return total


def packed_weight_memory(model):
    """打包后注意力层权重内存（字节）。"""
    total = 0
    for m in model.modules():
        if isinstance(m, PackedTernaryLinear):
            total += m.w_packed.numel() * m.w_packed.element_size()
            total += m.alpha.numel() * m.alpha.element_size()
            if m.bias is not None:
                total += m.bias.numel() * m.bias.element_size()
    return total


def convert_to_packed(model, pack_bits):
    """把微调后的三元 STE 层转换为打包存储。"""
    for name, module in list(model.named_modules()):
        if isinstance(module, TernarySpikeLinearSTE):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            out_f, in_f = module.weight.shape
            new = PackedTernaryLinear(in_f, out_f,
                                      bias=module.bias is not None, pack_bits=pack_bits, k=module.k)
            # 提取三元权重
            alpha = module.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            w_ternary = (module.weight / alpha).round().clamp(-1, 1)
            new.pack(w_ternary)
            new.alpha.copy_(alpha.squeeze(-1))
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(parent, attr, new)


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 1. QAT 微调
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True)
    convert_attn(model, TernarySpikeLinearSTE, k=K)
    print("QAT 微调中...")
    fine_tune(model, tokenizer)
    ppl_ft = compute_perplexity(model, tokenizer, EVAL_TEXTS)
    mem_float = attn_weight_memory(model)
    print(f"微调后留出PPL={ppl_ft:.2f}  注意力权重内存(float32)={mem_float/1024**2:.1f}MB")

    # 2. 打包存储
    for bits in [8, 2]:
        model_p = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, trust_remote_code=True)
        convert_attn(model_p, TernarySpikeLinearSTE, k=K)
        fine_tune(model_p, tokenizer)
        convert_to_packed(model_p, bits)
        ppl_packed = compute_perplexity(model_p, tokenizer, EVAL_TEXTS)
        mem_packed = packed_weight_memory(model_p)
        reduction = (1 - mem_packed / mem_float) * 100
        print(f"\n=== int{bits} 打包 ===")
        print(f"  留出PPL={ppl_packed:.2f} (微调后{ppl_ft:.2f})")
        print(f"  注意力权重内存: {mem_float/1024**2:.1f}MB -> {mem_packed/1024**2:.2f}MB  (降低{reduction:.1f}%)")
        del model_p


if __name__ == "__main__":
    main()
