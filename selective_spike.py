# -*- coding: utf-8 -*-
"""
Step2: 选择性脉冲化对比。
模式：
  - all      : 全部 Linear 层脉冲化（基线）
  - mlp_only : 仅 MLP 层（gate/up/down_proj）脉冲化
  - attn_only: 仅注意力层（q/k/v/o_proj）脉冲化
  - none     : 不脉冲化（仅权重量化）
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.quant_linear import QuantLinear
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
PROMPTS = [
    "Spiking neural networks are",
    "The future of artificial intelligence",
    "脉冲神经网络是一种",
    "Machine learning models can",
    "深度学习在自然语言处理中",
]


def should_spike(name, mode):
    if mode == "all":
        return True
    if mode == "mlp_only":
        return "mlp" in name
    if mode == "attn_only":
        return "self_attn" in name
    if mode == "none":
        return False
    raise ValueError(mode)


def convert_selective(model, mode, k=K):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if not should_spike(name, mode):
                continue
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

    # 原始模型参考 logits
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    with torch.no_grad():
        logits_ref = ref_model(**inputs).logits
    del ref_model

    print(f"\n{'模式':>10}{'PPL':>10}{'稀疏度':>10}{'cosine':>10}{'top1':>8}{'top5':>8}")
    print("-" * 56)
    for mode in ["all", "mlp_only", "attn_only", "none"]:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
        n = convert_selective(model, mode, k=K)
        ppl = compute_perplexity(model, tokenizer, PPL_TEXTS)
        sparsity = measure_sparsity(model, inputs)
        with torch.no_grad():
            logits_spk = model(**inputs).logits
        cos, top1, top5 = compare_logits(logits_ref, logits_spk)
        print(f"{mode:>10}{ppl:>10.2f}{sparsity*100:>9.1f}%{cos:>10.4f}{top1:>8.3f}{top5:>8.3f}  (替换{n}层)")
        del model


if __name__ == "__main__":
    main()
