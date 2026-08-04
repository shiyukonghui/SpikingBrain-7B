# -*- coding: utf-8 -*-
"""
Qwen3-0.6B -> SpikingBrain 脉冲模型转换 + 前向一致性对比

复用 SpikingBrain 的 W8ASpike 转换技术：
  1. 激活脉冲化：dynamic_spikes(x, k)  ->  round(x/vth) 经 SpikeCountBitwiseNode 编码
  2. 权重量化：  Quantizer 分组对称 int8 量化
  3. 用 QuantLinear 替换所有 nn.Linear
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer

from W8ASpike.quant_linear import QuantLinear

MODEL_PATH = "./models/Qwen3-0.6B"
W_GROUP_SIZE = 128   # 权重分组大小
W_BITS = 8           # 权重量化位宽
DYNAMIC_SFR = 3.0    # 动态脉冲阈值系数 k


def set_weight_scales(quant_linear: QuantLinear, w_group_size: int = W_GROUP_SIZE, bits: int = W_BITS):
    """按分组对称量化计算权重 scale，并写入 Quantizer。"""
    w = quant_linear.weight.detach()
    out_f, in_f = w.shape
    wg = w.reshape(out_f, -1, w_group_size)
    scale = wg.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)
    scale = scale.clamp(min=1e-8)
    quant_linear.weight_quantizer.scales.copy_(scale)


def convert_to_spiking(model: nn.Module, w_group_size: int = W_GROUP_SIZE,
                       bits: int = W_BITS, dynamic_sfr: float = DYNAMIC_SFR) -> int:
    """递归替换所有 nn.Linear 为 QuantLinear，返回替换数量。"""
    count = 0
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            new = QuantLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                w_group_size=w_group_size, dynamic_sfr=dynamic_sfr,
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            set_weight_scales(new, w_group_size, bits)
            setattr(model, name, new)
            count += 1
        else:
            count += convert_to_spiking(module, w_group_size, bits, dynamic_sfr)
    return count


def compare_logits(logits_ref: torch.Tensor, logits_spike: torch.Tensor):
    """计算原始与脉冲模型 logits 的一致性指标。"""
    ref = logits_ref.float().reshape(-1, logits_ref.shape[-1])
    spk = logits_spike.float().reshape(-1, logits_spike.shape[-1])

    cos = torch.nn.functional.cosine_similarity(ref, spk, dim=-1).mean().item()
    mae = (ref - spk).abs().mean().item()
    max_abs = (ref - spk).abs().max().item()

    # 预测 token 一致性（argmax）
    top1_ref = ref.argmax(dim=-1)
    top1_spk = spk.argmax(dim=-1)
    top1_acc = (top1_ref == top1_spk).float().mean().item()

    # top-5 命中率：脉冲模型预测 token 是否落在原始模型 top-5 内
    top5_ref = ref.topk(5, dim=-1).indices  # (N, 5)
    hit = (top5_ref == top1_spk.unsqueeze(-1)).any(dim=-1).float().mean().item()
    top5_acc = hit

    return {
        "cosine_sim": cos,
        "mae": mae,
        "max_abs_diff": max_abs,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
    }


def main():
    torch.manual_seed(0)
    device = "cpu"

    print("=== 加载 Qwen3-0.6B ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()

    # 测试输入
    texts = [
        "Spiking neural networks are",
        "The future of artificial intelligence",
        "脉冲神经网络是一种",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)

    print("=== 原始模型前向 ===")
    with torch.no_grad():
        out_ref = model(**inputs).logits

    print("=== 转换为脉冲模型 ===")
    n = convert_to_spiking(model)
    model.eval()  # 新替换的 QuantLinear 默认 training=True，需切回 eval
    print(f"已替换 {n} 个 Linear 层为 QuantLinear")

    print("=== 脉冲模型前向 ===")
    with torch.no_grad():
        out_spike = model(**inputs).logits

    print("=== 一致性对比 ===")
    metrics = compare_logits(out_ref, out_spike)
    for k, v in metrics.items():
        print(f"  {k:14s}: {v:.6f}")

    # 打印每个文本的预测 token
    print("\n=== 预测 token 对比 ===")
    for i, text in enumerate(texts):
        tok_ref = tokenizer.decode(out_ref[i, -1].argmax(-1).item())
        tok_spk = tokenizer.decode(out_spike[i, -1].argmax(-1).item())
        print(f"  [{text[:20]}...] 原始={tok_ref!r}  脉冲={tok_spk!r}")


if __name__ == "__main__":
    main()
