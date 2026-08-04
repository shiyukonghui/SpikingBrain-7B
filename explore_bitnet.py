# -*- coding: utf-8 -*-
"""
方向1探索：BitNet 三元权重 {-1,0,+1} × 脉冲激活。
对比：
  - original            : 原始模型
  - ternary_only        : 仅三元权重（无脉冲）
  - ternary+spike_attn  : 三元权重 + 注意力层脉冲化
  - ternary+spike_all   : 三元权重 + 全层脉冲化
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from bitnet_linear import convert_to_bitnet_spike

MODEL_PATH = "./models/Qwen3-0.6B"
K = 4.0
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


def weight_sparsity(model):
    """三元权重中零元素占比。"""
    from bitnet_linear import BitNetSpikeLinear
    zeros, total = 0, 0
    for m in model.modules():
        if isinstance(m, BitNetSpikeLinear):
            zeros += (m.w_ternary == 0).sum().item()
            total += m.w_ternary.numel()
    return zeros / total if total else 0.0


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=32)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    with torch.no_grad():
        logits_ref = ref_model(**inputs).logits
    ppl_ref = compute_perplexity(ref_model, tokenizer, PPL_TEXTS)
    del ref_model

    print(f"\n{'配置':<22}{'PPL':>10}{'激活稀疏':>10}{'权重稀疏':>10}{'cosine':>10}{'top1':>8}{'top5':>8}")
    print("-" * 78)

    configs = [
        ("ternary_only", dict(use_spike=False, spike_attn=False, spike_mlp=False)),
        ("ternary+spike_attn", dict(use_spike=True, spike_attn=True, spike_mlp=False)),
        ("ternary+spike_all", dict(use_spike=True, spike_attn=True, spike_mlp=True)),
    ]
    for name, cfg in configs:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
        convert_to_bitnet_spike(model, k=K, **cfg)
        ppl = compute_perplexity(model, tokenizer, PPL_TEXTS)
        sp_act = measure_sparsity(model, inputs)
        sp_w = weight_sparsity(model)
        with torch.no_grad():
            logits_spk = model(**inputs).logits
        cos, top1, top5 = compare_logits(logits_ref, logits_spk)
        print(f"{name:<22}{ppl:>10.2f}{sp_act*100:>9.1f}%{sp_w*100:>9.1f}%{cos:>10.4f}{top1:>8.3f}{top5:>8.3f}")
        del model


if __name__ == "__main__":
    main()
