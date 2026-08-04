# -*- coding: utf-8 -*-
"""
专用稀疏算法：LUT（查找表）matmul，参考 BitNet 的 three_lut_ctor。

原理：
  1. 三元权重 {-1,0,+1} 打包为 2-bit 码（-1→0, 0→1, +1→2）
  2. 激活量化到 int8
  3. 按权重块（B 个权重）预计算查找表 LUT：LUT[code] = 块内激活与权重的点积
  4. matmul 变成查表 + 累加，避免稠密乘法

LUT 每 token 构建一次（K/B 块 × 2^(2B) 项），复用于所有输出神经元。
"""
import time

import torch

torch.manual_seed(0)


def decode_2bit(code):
    """2-bit 码 -> 三元值：0→-1, 1→0, 2→+1, 3→0。"""
    return torch.where(code == 0, -1.0, torch.where(code == 2, 1.0, 0.0))


def pack_ternary_2bit(w_ternary, B):
    """三元权重 {-1,0,+1} -> 2-bit 码，shape (N, K/B)。"""
    N, K = w_ternary.shape
    codes = torch.where(w_ternary == -1, 0, torch.where(w_ternary == 0, 1, 2)).to(torch.int64)
    codes = codes.reshape(N, -1, B)
    packed = torch.zeros(N, K // B, dtype=torch.int64, device=w_ternary.device)
    for i in range(B):
        packed |= codes[:, :, i] << (2 * i)
    return packed


def build_lut(x_q, B):
    """构建 LUT：shape (M, K/B, 2^(2B))。x_q: (M, K) int8 激活。"""
    M, K = x_q.shape
    n_blocks = K // B
    n_codes = 1 << (2 * B)
    codes = torch.arange(n_codes, device=x_q.device, dtype=torch.int64)  # (2^(2B),)
    # 每个位置的 2-bit 值
    pos_vals = torch.stack([(codes >> (2 * i)) & 3 for i in range(B)], dim=0)  # (B, 2^(2B))
    pos_weights = decode_2bit(pos_vals)  # (B, 2^(2B))
    # 激活分块
    x_blocks = x_q.reshape(M, n_blocks, B)  # (M, K/B, B)
    # LUT[b][code] = sum_i x[m, b, i] * w_i(code)
    lut = torch.einsum("mbi,ic->mbc", x_blocks, pos_weights)  # (M, K/B, 2^(2B))
    return lut


def lut_matmul(x, w_ternary, B=4):
    """LUT matmul：x (M,K) @ w_ternary (N,K) -> (M,N)。"""
    M, K = x.shape
    N = w_ternary.shape[0]
    # 激活量化到 int8
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    x_q = (x * scale).round().clamp(-128, 127)
    # 打包权重
    w_code = pack_ternary_2bit(w_ternary, B)  # (N, K/B)
    # 构建 LUT
    lut = build_lut(x_q, B)  # (M, K/B, 2^(2B))
    # 查表：out[m, n] = sum_b lut[m, b, w_code[n, b]]（向量化）
    idx = w_code.t()  # (K/B, N)
    gathered = lut.gather(2, idx.unsqueeze(0).expand(M, -1, -1))  # (M, K/B, N)
    out = gathered.sum(1)  # (M, N)
    # 反量化
    out = out / scale
    return out


def bench(fn, n=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    for (M, K, N) in [(1, 1024, 1024), (8, 1024, 1024), (1, 4096, 4096)]:
        x = torch.randn(M, K)
        # 三元权重（34.6% 零，模拟实测）
        w = torch.randn(N, K)
        alpha = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        w_ternary = (w / alpha).round().clamp(-1, 1)

        def dense():
            return torch.nn.functional.linear(x, w_ternary)

        def lut():
            return lut_matmul(x, w_ternary, B=4)

        # 正确性
        out_dense = dense()
        out_lut = lut()
        err = (out_dense - out_lut).abs().mean().item()

        t_dense = bench(dense)
        t_lut = bench(lut)
        print(f"M={M} K={K} N={N}: 稠密={t_dense*1000:.3f}ms LUT={t_lut*1000:.3f}ms "
              f"加速={t_dense/t_lut:.2f}x 误差={err:.4f}")


if __name__ == "__main__":
    main()
