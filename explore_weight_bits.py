# -*- coding: utf-8 -*-
"""
权重量化位宽探索：int8 / int4 / int2。
基于最优配置 attn_only + 逐层 k，仅改变权重量化位宽。
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
PER_LAYER_K = [2.0, 1.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 4.0, 1.5, 1.5, 1.5,
               1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
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


def convert_attn_only_perlayer(model, bits):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "self_attn" in name:
            layer_idx = int(name.split(".")[2])
            k = PER_LAYER_K[layer_idx]
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = QuantLinear(module.in_features, module.out_features,
                              bias=module.bias is not None, w_group_size=128, dynamic_sfr=k)
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            set_weight_scales(new, 128, bits)
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

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    with torch.no_grad():
        logits_ref = ref_model(**inputs).logits
    del ref_model

    print(f"\n{'位宽':>6}{'PPL':>10}{'稀疏度':>10}{'cosine':>10}{'top1':>8}{'top5':>8}")
    print("-" * 52)
    for bits in [8, 4, 2]:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
        convert_attn_only_perlayer(model, bits)
        ppl = compute_perplexity(model, tokenizer, PPL_TEXTS)
        sp = measure_sparsity(model, inputs)
        with torch.no_grad():
            logits_spk = model(**inputs).logits
        cos, top1, top5 = compare_logits(logits_ref, logits_spk)
        print(f"int{bits:<4}{ppl:>10.2f}{sp*100:>9.1f}%{cos:>10.4f}{top1:>8.3f}{top5:>8.3f}")
        del model


if __name__ == "__main__":
    main()
