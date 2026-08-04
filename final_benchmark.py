# -*- coding: utf-8 -*-
"""
最终综合基准：原始模型 vs 基线脉冲(all,k=3) vs 优化脉冲(attn_only+逐层k)。
指标：PPL / 稀疏度 / 一致性 / 前向CPU时间 / 解码速度 / 内存。
"""
import gc
import math
import os
import sys
import time

import psutil
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.quant_linear import QuantLinear
from convert_qwen3_spike import convert_to_spiking, set_weight_scales

MODEL_PATH = "./models/Qwen3-0.6B"
# Step3 贪心得到的逐层 k（attn_only）
PER_LAYER_K = [2.0, 1.5, 2.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0, 4.0, 1.5, 1.5, 1.5,
               1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
PROMPTS = [
    "Spiking neural networks are",
    "The future of artificial intelligence",
    "脉冲神经网络是一种",
    "Machine learning models can",
    "深度学习在自然语言处理中",
]
PPL_TEXTS = [
    "Spiking neural networks are inspired by the brain.",
    "The future of artificial intelligence is bright.",
    "脉冲神经网络是一种受大脑启发的计算模型。",
    "Machine learning models can learn from large amounts of data.",
    "深度学习在自然语言处理领域取得了巨大成功。",
]

process = psutil.Process()


def rss_mb():
    return process.memory_info().rss / 1024 ** 2


def convert_attn_only_perlayer(model):
    """attn_only + 逐层 k。"""
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


def measure_forward(model, inputs, n=3):
    with torch.no_grad():
        model(**inputs)
    gc.collect()
    mem_before = rss_mb()
    wall, cpu = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        c0 = time.process_time()
        with torch.no_grad():
            model(**inputs)
        wall.append(time.perf_counter() - t0)
        cpu.append(time.process_time() - c0)
    mem_after = rss_mb()
    return sum(cpu) / n, sum(wall) / n, mem_after - mem_before


def measure_decode(model, tokenizer, prompt, max_new=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    dt = time.perf_counter() - t0
    n_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    return n_tokens / dt


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

    # 原始模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    with torch.no_grad():
        logits_ref = model(**inputs).logits
    ppl_ref = compute_perplexity(model, tokenizer, PPL_TEXTS)
    cpu_ref, wall_ref, mem_ref = measure_forward(model, inputs)
    dec_ref = measure_decode(model, tokenizer, PROMPTS[0])
    del model

    # 基线脉冲（all, k=3）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    convert_to_spiking(model)
    model.eval()
    ppl_base = compute_perplexity(model, tokenizer, PPL_TEXTS)
    sp_base = measure_sparsity(model, inputs)
    with torch.no_grad():
        logits_base = model(**inputs).logits
    cpu_base, wall_base, mem_base = measure_forward(model, inputs)
    dec_base = measure_decode(model, tokenizer, PROMPTS[0])
    del model

    # 优化脉冲（attn_only + 逐层 k）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()
    n = convert_attn_only_perlayer(model)
    ppl_opt = compute_perplexity(model, tokenizer, PPL_TEXTS)
    sp_opt = measure_sparsity(model, inputs)
    with torch.no_grad():
        logits_opt = model(**inputs).logits
    cpu_opt, wall_opt, mem_opt = measure_forward(model, inputs)
    dec_opt = measure_decode(model, tokenizer, PROMPTS[0])

    cos_base, t1_base, t5_base = compare_logits(logits_ref, logits_base)
    cos_opt, t1_opt, t5_opt = compare_logits(logits_ref, logits_opt)

    print("\n" + "=" * 92)
    print(f"{'指标':<20}{'原始':>14}{'基线脉冲':>14}{'优化脉冲':>14}{'优化vs基线':>14}")
    print("-" * 92)
    rows = [
        ("PPL", f"{ppl_ref:.2f}", f"{ppl_base:.2f}", f"{ppl_opt:.2f}",
         f"{(ppl_opt/ppl_base-1)*100:+.1f}%"),
        ("稀疏度", "-", f"{sp_base*100:.1f}%", f"{sp_opt*100:.1f}%",
         f"{(sp_opt/sp_base-1)*100:+.1f}%"),
        ("cosine", "-", f"{cos_base:.4f}", f"{cos_opt:.4f}", ""),
        ("top1", "-", f"{t1_base:.3f}", f"{t1_opt:.3f}", ""),
        ("top5", "-", f"{t5_base:.3f}", f"{t5_opt:.3f}", ""),
        ("前向CPU", f"{cpu_ref:.2f}s", f"{cpu_base:.2f}s", f"{cpu_opt:.2f}s",
         f"{(cpu_opt/cpu_base-1)*100:+.1f}%"),
        ("解码速度", f"{dec_ref:.2f}t/s", f"{dec_base:.2f}t/s", f"{dec_opt:.2f}t/s",
         f"{(dec_opt/dec_base-1)*100:+.1f}%"),
        ("内存增量", f"{mem_ref:+.1f}MB", f"{mem_base:+.1f}MB", f"{mem_opt:+.1f}MB", ""),
    ]
    for name, r, b, o, chg in rows:
        print(f"{name:<20}{r:>14}{b:>14}{o:>14}{chg:>14}")
    print("=" * 92)
    print(f"优化脉冲替换 {n} 层（attn_only + 逐层 k）")


if __name__ == "__main__":
    main()
