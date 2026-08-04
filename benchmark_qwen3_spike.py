# -*- coding: utf-8 -*-
"""
Qwen3-0.6B 原始模型 vs 脉冲模型 综合性能对比
指标：模型规模 / 前向CPU时间 / 内存占用 / 解码速度 / 困惑度 / 一致性 / 激活稀疏度
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
from convert_qwen3_spike import convert_to_spiking

MODEL_PATH = "./models/Qwen3-0.6B"
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_forward(model, inputs, n=3):
    """前向推理：CPU 时间 + 峰值内存增量。"""
    with torch.no_grad():
        model(**inputs)  # warmup
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
    return {
        "wall_avg": sum(wall) / n,
        "cpu_avg": sum(cpu) / n,
        "mem_delta": mem_after - mem_before,
    }


def measure_decode(model, tokenizer, prompt, max_new=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    dt = time.perf_counter() - t0
    n_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    return n_tokens / dt, dt, n_tokens


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
    """脉冲模型激活稀疏度：统计 dynamic_spikes 产生的 spike 计数中零元素占比。"""
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
    mae = (ref - spk).abs().mean().item()
    top1_ref = ref.argmax(-1)
    top1_spk = spk.argmax(-1)
    top1_acc = (top1_ref == top1_spk).float().mean().item()
    top5_ref = ref.topk(5, -1).indices
    top5_acc = (top5_ref == top1_spk.unsqueeze(-1)).any(-1).float().mean().item()
    return {"cosine_sim": cos, "mae": mae, "top1_acc": top1_acc, "top5_acc": top5_acc}


def fmt(v, unit=""):
    return f"{v:.4f}{unit}"


def main():
    torch.manual_seed(0)
    print("=== 加载 Qwen3-0.6B ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True).eval()

    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=32)

    n_params = count_params(model)
    model_mem = n_params * 4 / 1024 ** 2  # float32 -> MB

    print("=== 原始模型基准 ===")
    fwd_ref = measure_forward(model, inputs)
    ppl_ref = compute_perplexity(model, tokenizer, PPL_TEXTS)
    dec_ref = measure_decode(model, tokenizer, PROMPTS[0])
    with torch.no_grad():
        logits_ref = model(**inputs).logits

    print("=== 转换为脉冲模型 ===")
    n_linear = convert_to_spiking(model)
    model.eval()
    print(f"已替换 {n_linear} 个 Linear 层")

    print("=== 脉冲模型基准 ===")
    fwd_spk = measure_forward(model, inputs)
    ppl_spk = compute_perplexity(model, tokenizer, PPL_TEXTS)
    dec_spk = measure_decode(model, tokenizer, PROMPTS[0])
    sparsity = measure_sparsity(model, inputs)
    with torch.no_grad():
        logits_spk = model(**inputs).logits

    acc = compare_logits(logits_ref, logits_spk)

    # ===== 输出对比表 =====
    print("\n" + "=" * 78)
    print(f"{'指标':<22}{'原始模型':>16}{'脉冲模型':>16}{'变化':>14}")
    print("-" * 78)
    rows = [
        ("参数量", f"{n_params/1e6:.1f}M", f"{n_params/1e6:.1f}M", "-"),
        ("权重内存", f"{model_mem:.0f}MB", f"{model_mem:.0f}MB", "-"),
        ("前向CPU时间", fmt(fwd_ref['cpu_avg'], 's'), fmt(fwd_spk['cpu_avg'], 's'),
         f"{(fwd_spk['cpu_avg']/fwd_ref['cpu_avg']):.2f}x"),
        ("前向墙钟时间", fmt(fwd_ref['wall_avg'], 's'), fmt(fwd_spk['wall_avg'], 's'),
         f"{(fwd_spk['wall_avg']/fwd_ref['wall_avg']):.2f}x"),
        ("前向内存增量", fmt(fwd_ref['mem_delta'], 'MB'), fmt(fwd_spk['mem_delta'], 'MB'),
         f"{(fwd_spk['mem_delta']-fwd_ref['mem_delta']):+.1f}MB"),
        ("解码速度", fmt(dec_ref[0], 'tok/s'), fmt(dec_spk[0], 'tok/s'),
         f"{(dec_spk[0]/dec_ref[0]):.2f}x"),
        ("困惑度(PPL)", fmt(ppl_ref), fmt(ppl_spk), f"{(ppl_spk/ppl_ref):.2f}x"),
        ("激活稀疏度", "-", f"{sparsity*100:.1f}%", "-"),
    ]
    for name, r, s, chg in rows:
        print(f"{name:<22}{r:>16}{s:>16}{chg:>14}")
    print("-" * 78)
    print("一致性对比（脉冲 vs 原始 logits）：")
    for k, v in acc.items():
        print(f"  {k:12s}: {v:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
