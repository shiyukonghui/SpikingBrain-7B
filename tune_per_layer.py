# -*- coding: utf-8 -*-
"""
Step3: 逐层阈值 k 优化（贪心）。
基于 Step2 最优配置 attn_only，为每个注意力层组贪心选择最大 k，
在 PPL 不超过阈值的前提下最大化稀疏度。
"""
import math
import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.quant_linear import QuantLinear
from convert_qwen3_spike import set_weight_scales

MODEL_PATH = "./models/Qwen3-0.6B"
BASE_K = 4.0
PPL_THRESHOLD = 49.0
# 升序 k = 降序稀疏度；贪心选满足 PPL 约束的最小 k（= 最大稀疏度）
K_CANDIDATES = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
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


def convert_attn_only(model, k=BASE_K, all_layers=False):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and (all_layers or "self_attn" in name):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = QuantLinear(module.in_features, module.out_features,
                              bias=module.bias is not None, w_group_size=128, dynamic_sfr=k)
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            set_weight_scales(new, 128, 8)
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


def set_layer_k(model, layer_idx, k):
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear) and f".{layer_idx}." in name and "self_attn" in name:
            m.k = k


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=32)

    all_layers = len(sys.argv) > 1 and sys.argv[1] == "all"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    n = convert_attn_only(model, k=BASE_K, all_layers=all_layers)
    print(f"{'all_layers' if all_layers else 'attn_only'} 转换完成，替换 {n} 层，初始 k={BASE_K}")

    n_layers = model.config.num_hidden_layers
    chosen = {}
    for i in range(n_layers):
        best_k = K_CANDIDATES[-1]  # 兜底：最大 k（最安全）
        for k in K_CANDIDATES:  # 从最小 k（最稀疏）开始
            set_layer_k(model, i, k)
            ppl = compute_perplexity(model, tokenizer, PPL_TEXTS)
            if ppl <= PPL_THRESHOLD:
                best_k = k  # 满足约束的最小 k = 最大稀疏度
                break
        set_layer_k(model, i, best_k)
        chosen[i] = best_k
        print(f"  layer {i:2d}: k={best_k:>4.1f}")

    ppl_final = compute_perplexity(model, tokenizer, PPL_TEXTS)
    sparsity = measure_sparsity(model, inputs)
    print(f"\n最终 PPL={ppl_final:.2f}  稀疏度={sparsity*100:.1f}%")
    print("逐层 k:", [chosen[i] for i in range(n_layers)])


if __name__ == "__main__":
    main()
