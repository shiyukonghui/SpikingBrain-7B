# -*- coding: utf-8 -*-
"""
对比 三元+QAT 与 int8+QAT：CPU / 内存 / 解码速度 / 精度 / 稀疏度。
两个模型用相同训练数据、相同步数微调，公平对比。
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

from W8ASpike.Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode

MODEL_PATH = "./models/Qwen3-0.6B"
K = 4.0
W_GROUP = 128
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
PROMPTS = [
    "Spiking neural networks are",
    "The future of artificial intelligence",
    "脉冲神经网络是一种",
    "Machine learning models can",
    "深度学习在自然语言处理中",
]

process = psutil.Process()


def rss_mb():
    return process.memory_info().rss / 1024 ** 2


class SpikeActSTE:
    """脉冲激活 STE 工具。"""
    act_sparsities = []

    @staticmethod
    def spike(x, k=K):
        vth = x.abs().mean([-1], keepdim=True).float() / k
        vth = vth.clamp(min=1e-5, max=1e4)
        spikes_int = (x / vth).round()
        spikes_int = spike_fake_quant(spikes_int, SpikeCountBitwiseNode(is_bidirectional=True))
        SpikeActSTE.act_sparsities.append((spikes_int == 0).float().mean().item())
        x_spike = spikes_int * vth
        return x + (x_spike - x).detach()


class Int8SpikeLinearSTE(nn.Module):
    """int8 分组权重 + 脉冲激活（STE）。"""

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

    def quantize_weight(self):
        w = self.weight
        wg = w.reshape(self.out_features, -1, W_GROUP)
        scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127
        w_q = (wg / scale).round().clamp(-128, 127) * scale
        return w_q.reshape(self.out_features, self.in_features)

    def forward(self, x):
        w_q = self.quantize_weight()
        w_ste = self.weight + (w_q - self.weight).detach()
        x_ste = SpikeActSTE.spike(x, self.k)
        return nn.functional.linear(x_ste, w_ste, self.bias)


class TernarySpikeLinearSTE(nn.Module):
    """三元权重 + 脉冲激活（STE）。"""

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
        x_ste = SpikeActSTE.spike(x, self.k)
        return nn.functional.linear(x_ste, w_ste, self.bias)


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


def measure_sparsity(model, inputs):
    SpikeActSTE.act_sparsities = []
    with torch.no_grad():
        model(**inputs)
    act_sp = (sum(SpikeActSTE.act_sparsities) / len(SpikeActSTE.act_sparsities)
              if SpikeActSTE.act_sparsities else 0.0)
    w_zeros, w_total = 0, 0
    for m in model.modules():
        if isinstance(m, (Int8SpikeLinearSTE, TernarySpikeLinearSTE)):
            wq = m.quantize_weight()
            w_zeros += (wq == 0).sum().item()
            w_total += wq.numel()
    w_sp = w_zeros / w_total if w_total else 0.0
    return act_sp, w_sp


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


def run_pipeline(name, cls, tokenizer, inputs):
    print(f"\n=== {name} ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, trust_remote_code=True)
    n = convert_attn(model, cls, k=K)
    print(f"转换 {n} 层，开始微调...")
    fine_tune(model, tokenizer)
    ppl = compute_perplexity(model, tokenizer, EVAL_TEXTS)
    act_sp, w_sp = measure_sparsity(model, inputs)
    cpu, wall, mem = measure_forward(model, inputs)
    dec = measure_decode(model, tokenizer, PROMPTS[0])
    del model
    gc.collect()
    return {"ppl": ppl, "act_sp": act_sp, "w_sp": w_sp,
            "cpu": cpu, "wall": wall, "mem": mem, "dec": dec}


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True,
                       truncation=True, max_length=32)

    res_int8 = run_pipeline("int8+QAT", Int8SpikeLinearSTE, tokenizer, inputs)
    res_tern = run_pipeline("ternary+QAT", TernarySpikeLinearSTE, tokenizer, inputs)

    print("\n" + "=" * 78)
    print(f"{'指标':<18}{'int8+QAT':>16}{'ternary+QAT':>16}{'三元vs int8':>16}")
    print("-" * 78)
    rows = [
        ("留出PPL", f"{res_int8['ppl']:.2f}", f"{res_tern['ppl']:.2f}",
         f"{(res_tern['ppl']/res_int8['ppl']-1)*100:+.1f}%"),
        ("激活稀疏", f"{res_int8['act_sp']*100:.1f}%", f"{res_tern['act_sp']*100:.1f}%",
         f"{(res_tern['act_sp']/res_int8['act_sp']-1)*100:+.1f}%"),
        ("权重稀疏", f"{res_int8['w_sp']*100:.1f}%", f"{res_tern['w_sp']*100:.1f}%",
         f"{(res_tern['w_sp']/res_int8['w_sp']-1)*100:+.1f}%"),
        ("前向CPU", f"{res_int8['cpu']:.2f}s", f"{res_tern['cpu']:.2f}s",
         f"{(res_tern['cpu']/res_int8['cpu']-1)*100:+.1f}%"),
        ("前向墙钟", f"{res_int8['wall']:.2f}s", f"{res_tern['wall']:.2f}s",
         f"{(res_tern['wall']/res_int8['wall']-1)*100:+.1f}%"),
        ("内存增量", f"{res_int8['mem']:+.1f}MB", f"{res_tern['mem']:+.1f}MB", ""),
        ("解码速度", f"{res_int8['dec']:.2f}t/s", f"{res_tern['dec']:.2f}t/s",
         f"{(res_tern['dec']/res_int8['dec']-1)*100:+.1f}%"),
    ]
    for name, a, b, chg in rows:
        print(f"{name:<18}{a:>16}{b:>16}{chg:>16}")
    print("=" * 78)


if __name__ == "__main__":
    main()
