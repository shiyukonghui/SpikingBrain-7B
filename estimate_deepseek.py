# -*- coding: utf-8 -*-
"""
预估 DeepSeek-V4-Flash-0731 应用组合方案（BitNet三元量化 + SpikingBrain脉冲化）的优化潜力。

基于 config.json 计算参数分布，结合 Qwen3 实验实测的稀疏度，估算：
  1. 权重内存降低（int2 三元 vs 现有 fp8/fp4）
  2. 计算量降低（激活稀疏 + 权重稀疏 + MoE 稀疏）
  3. 综合优化
"""
import json
import os

# 实测稀疏度（来自 Qwen3 实验，方向11）
ACT_SPARSITY = 0.147   # 激活稀疏度 14.7%
W_SPARSITY = 0.346     # 权重稀疏度 34.6%
COMBINED_SKIP = 1 - (1 - ACT_SPARSITY) * (1 - W_SPARSITY)  # 44.3%

# 现有量化位宽（DeepSeek-V4-Flash）
ATTN_BITS = 8    # fp8 注意力权重
EXPERT_BITS = 4  # fp4 专家权重
# 组合方案位宽
TERNARY_BITS = 2  # int2 三元

# 模型配置
cfg = json.load(open(os.path.join("dir", "config.json"), encoding="utf-8"))
H = cfg["hidden_size"]            # 4096
L = cfg["num_hidden_layers"]      # 43
N_HEADS = cfg["num_attention_heads"]  # 64
HEAD_DIM = cfg["head_dim"]        # 512
N_KV = cfg["num_key_value_heads"] # 1
Q_DIM = N_HEADS * HEAD_DIM        # 32768
KV_DIM = N_KV * HEAD_DIM          # 512
Q_LORA = cfg["q_lora_rank"]       # 1024
O_LORA = cfg["o_lora_rank"]       # 1024
MOE_INT = cfg["moe_intermediate_size"]  # 2048
N_EXPERTS = cfg["n_routed_experts"]     # 256
N_SHARED = cfg["n_shared_experts"]      # 1
N_ACTIVE = cfg["num_experts_per_tok"]   # 6
VOCAB = cfg["vocab_size"]               # 129280


def attn_params_per_layer():
    # q: LoRA (H×Q_LORA + Q_LORA×Q_DIM)
    q = H * Q_LORA + Q_LORA * Q_DIM
    # k, v: H×KV_DIM each
    kv = 2 * H * KV_DIM
    # o: LoRA (Q_DIM×O_LORA + O_LORA×H)
    o = Q_DIM * O_LORA + O_LORA * H
    return q + kv + o


def expert_params_per_layer():
    # 每个专家: gate(H×MOE_INT) + up(H×MOE_INT) + down(MOE_INT×H)
    per_expert = 3 * H * MOE_INT
    routed = per_expert * N_EXPERTS
    shared = per_expert * N_SHARED
    return routed + shared


def main():
    attn_per = attn_params_per_layer()
    expert_per = expert_params_per_layer()
    attn_total = attn_per * L
    expert_total = expert_per * L
    embed = VOCAB * H * 2  # 输入+输出嵌入（tie_word_embeddings=false）
    total = attn_total + expert_total + embed

    print("=" * 70)
    print("DeepSeek-V4-Flash-0731 参数分布")
    print("=" * 70)
    print(f"  注意力参数: {attn_total/1e9:.1f}B ({attn_total/total*100:.1f}%)")
    print(f"  专家参数:   {expert_total/1e9:.1f}B ({expert_total/total*100:.1f}%)")
    print(f"  嵌入参数:   {embed/1e9:.1f}B ({embed/total*100:.1f}%)")
    print(f"  总参数:     {total/1e9:.1f}B")
    print(f"  每token激活专家: {N_ACTIVE}/{N_EXPERTS} = {N_ACTIVE/N_EXPERTS*100:.1f}%")

    print("\n" + "=" * 70)
    print("权重内存降低（int2 三元 vs 现有 fp8/fp4）")
    print("=" * 70)
    # 现有内存
    mem_attn_now = attn_total * ATTN_BITS / 8
    mem_expert_now = expert_total * EXPERT_BITS / 8
    mem_embed_now = embed * 16 / 8  # bf16 嵌入
    mem_now = mem_attn_now + mem_expert_now + mem_embed_now
    # 组合方案内存
    mem_attn_new = attn_total * TERNARY_BITS / 8
    mem_expert_new = expert_total * TERNARY_BITS / 8
    mem_new = mem_attn_new + mem_expert_new + mem_embed_now
    print(f"  现有: 注意力fp8+专家fp4 = {mem_now/1e9:.1f}GB")
    print(f"  组合: 全部int2三元       = {mem_new/1e9:.1f}GB")
    print(f"  权重内存降低: {(1-mem_new/mem_now)*100:.1f}%")
    print(f"  注意力权重: fp8→int2 = {8/TERNARY_BITS:.0f}x 降低")
    print(f"  专家权重:   fp4→int2 = {4/TERNARY_BITS:.0f}x 降低")

    print("\n" + "=" * 70)
    print("计算量降低（脉冲化稀疏 + MoE 稀疏）")
    print("=" * 70)
    print(f"  激活稀疏度(实测): {ACT_SPARSITY*100:.1f}%")
    print(f"  权重稀疏度(实测): {W_SPARSITY*100:.1f}%")
    print(f"  联合跳过率:       {COMBINED_SKIP*100:.1f}%")
    print(f"  MoE 激活率:       {N_ACTIVE/N_EXPERTS*100:.1f}%")
    # 有效计算 = 激活专家比例 × (1-联合跳过率)
    eff_compute = (N_ACTIVE / N_EXPERTS) * (1 - COMBINED_SKIP)
    print(f"  有效计算占比: {N_ACTIVE/N_EXPERTS*100:.1f}% × { (1-COMBINED_SKIP)*100:.1f}% = {eff_compute*100:.2f}%")
    print(f"  相对稠密全量: 计算降低 {(1-eff_compute)*100:.2f}%")

    print("\n" + "=" * 70)
    print("综合优化预估")
    print("=" * 70)
    print(f"  权重内存: 降低 {(1-mem_new/mem_now)*100:.1f}%")
    print(f"  计算量:   降低 {(1-eff_compute)*100:.2f}%（含MoE+脉冲稀疏）")
    print(f"  说明: 内存降低是确定性的（位宽决定）；计算降低需专用硬件兑现稀疏加法")


if __name__ == "__main__":
    main()
