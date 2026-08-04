# -*- coding: utf-8 -*-
"""
Step1: 扫描稀疏度系数 k，绘制 稀疏度-精度 权衡曲线。
转换一次模型，遍历不同 k 值，测量 PPL / 稀疏度 / 一致性。
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.quant_linear import QuantLinear
from convert_qwen3_spike import convert_to_spiking

MODEL_PATH = "./models/Qwen3-0.6B"
PPL_TEXTS = [
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
K_VALUES = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


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
    import W8ASpike.quant_linear as ql
    sparsities = []
    orig = ql.dynamic_spikes

    def patched(x, k=3.0):
        spikes_int, vth = orig(x, k)
        sparsities.append((spikes_int == 0).float().mean().item())
        return spikes_int, vth

    ql.dynamic_spikes = patched
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        ql.dynamic_spikes = orig
    return sum(sparsities) / len(sparsities) if sparsities else 0.0


def compare_logits(logits_ref, logits_spike):
    ref = logits_ref.float().reshape(-1, logits_ref.shape[-1])
    spk = logits_spike.float().reshape(-1, logits_spike.shape[-1])
    cos = nn.functional.cosine_similarity(ref, spk, dim=-1).mean().item()
    top1_ref = ref.argmax(-1)
    top1_spk = spk.argmax(-1)
    top1_acc = (top1_ref == top1_spk).float().mean().item()
    top5_ref = ref.topk(5, -1).indices
    top5_acc = (top5_ref == top1_spk.unsqueeze(-1)).any(-1).float().mean().item()
    return cos, top1_acc, top5_acc


def main():
    torch.manual_seed(0)
    print("=== 加载模型 ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()

    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=32)
    with torch.no_grad():
        logits_ref = model(**inputs).logits

    convert_to_spiking(model)
    model.eval()

    print(f"\n{'k':>6}{'PPL':>10}{'稀疏度':>10}{'cosine':>10}{'top1':>8}{'top5':>8}")
    print("-" * 52)
    results = []
    for k in K_VALUES:
        for m in model.modules():
            if isinstance(m, QuantLinear):
                m.k = k
        ppl = compute_perplexity(model, tokenizer, PPL_TEXTS)
        sparsity = measure_sparsity(model, inputs)
        with torch.no_grad():
            logits_spk = model(**inputs).logits
        cos, top1, top5 = compare_logits(logits_ref, logits_spk)
        results.append((k, ppl, sparsity, cos, top1, top5))
        print(f"{k:>6.1f}{ppl:>10.2f}{sparsity*100:>9.1f}%{cos:>10.4f}{top1:>8.3f}{top5:>8.3f}")

    print("\n=== 汇总 ===")
    for k, ppl, sparsity, cos, top1, top5 in results:
        print(f"k={k:>5.1f}  PPL={ppl:7.2f}  稀疏度={sparsity*100:5.1f}%  cosine={cos:.4f}  top1={top1:.3f}  top5={top5:.3f}")


if __name__ == "__main__":
    main()
