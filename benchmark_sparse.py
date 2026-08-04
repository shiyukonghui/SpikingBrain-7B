# -*- coding: utf-8 -*-
"""
方向4探索：稀疏 matmul 能否在 CPU 上利用脉冲稀疏度加速。
对比稠密 matmul 与 torch 稀疏 matmul，在不同稀疏度下测速。
"""
import time

import torch

torch.manual_seed(0)


def bench(fn, n=20):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    # 模拟 Qwen3 典型层：B=1, L=32, K=1024, N=1024
    B, L, K, N = 1, 32, 1024, 1024
    x = torch.randn(B, L, K)
    w = torch.randn(N, K)

    print(f"形状: x=({B},{L},{K}), w=({N},{K})")
    print(f"{'稀疏度':>8}{'稠密(ms)':>12}{'稀疏(ms)':>12}{'加速比':>10}")
    print("-" * 44)

    for sparsity in [0.0, 0.2, 0.32, 0.5, 0.7, 0.9]:
        mask = (torch.rand_like(x) > sparsity).float()
        x_sp = x * mask

        def dense():
            return torch.nn.functional.linear(x_sp, w)

        # 稀疏 matmul：把 x 转成稀疏，与 w.t() 相乘
        x_flat = x_sp.reshape(-1, K)
        x_sparse = x_flat.to_sparse()

        def sparse():
            return torch.sparse.mm(x_sparse, w.t())

        t_dense = bench(dense)
        t_sparse = bench(sparse)
        speedup = t_dense / t_sparse
        print(f"{sparsity*100:>7.0f}%{t_dense*1000:>12.3f}{t_sparse*1000:>12.3f}{speedup:>10.2f}x")


if __name__ == "__main__":
    main()
