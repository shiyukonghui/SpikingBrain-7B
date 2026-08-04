# -*- coding: utf-8 -*-
"""
基准测试：Rust LUT 核 vs PyTorch 稠密 matmul。
通过 ctypes 调用 rust_lut.dll。
"""
import ctypes
import os
import time

import torch

torch.manual_seed(0)

LIB = os.path.join("rust_lut", "target", "release", "rust_lut.dll")
lib = ctypes.CDLL(LIB)

# 定义函数签名
for fn in ["lut_matmul", "lut_matmul_opt", "lut_matmul_mt",
           "lut_matmul_blocked", "lut_matmul_avx2"]:
    f = getattr(lib, fn)
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    f.restype = None

# 紧凑 LUT 签名（多 3 个参数）
lib.lut_matmul_compact.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.lut_matmul_compact.restype = None


def precompute_codes(w_code, n_blocks, n, max_unique=256):
    """预计算每块唯一权重码。w_code: (n_blocks, n)。"""
    unique_codes = torch.zeros(n_blocks, max_unique, dtype=torch.uint8)
    unique_counts = torch.zeros(n_blocks, dtype=torch.int32)
    code_to_idx = torch.full((n_blocks, 256), -1, dtype=torch.int32)
    for bb in range(n_blocks):
        uniq = torch.unique(w_code[bb])
        cnt = uniq.numel()
        unique_counts[bb] = cnt
        unique_codes[bb, :cnt] = uniq
        code_to_idx[bb, uniq.long()] = torch.arange(cnt, dtype=torch.int32)
    return unique_codes, unique_counts, code_to_idx


def pack_ternary_2bit(w_ternary):
    """三元权重 {-1,0,+1} -> 2-bit 码，shape (N, K/4)。"""
    N, K = w_ternary.shape
    codes = torch.where(w_ternary == -1, 0, torch.where(w_ternary == 0, 1, 2)).to(torch.int64)
    codes = codes.reshape(N, -1, 4)
    packed = torch.zeros(N, K // 4, dtype=torch.uint8, device=w_ternary.device)
    for i in range(4):
        packed |= (codes[:, :, i] << (2 * i)).to(torch.uint8)
    return packed


def rust_lut(x_q, w_code, scale, m, k, n, opt=False):
    """调用 Rust LUT 核。x_q: (M,K) int8, w_code: (N,K/4) uint8, scale: (M,1)。"""
    out = torch.zeros(m, n, dtype=torch.float32)
    fn = lib.lut_matmul_opt if opt else lib.lut_matmul
    fn(x_q.data_ptr(), w_code.data_ptr(), out.data_ptr(), m, k, n)
    return out / scale  # 反量化


def bench(fn, n=50):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    for (M, K, N) in [(1, 1024, 1024), (8, 1024, 1024), (1, 4096, 4096), (1, 4096, 8192)]:
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        alpha = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        w_ternary = (w / alpha).round().clamp(-1, 1)

        # 激活量化到 int8
        scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * scale).round().clamp(-128, 127).to(torch.int8)
        w_code = pack_ternary_2bit(w_ternary)

        def dense():
            return torch.nn.functional.linear(x, w_ternary)

        def lut_basic():
            return rust_lut(x_q, w_code, scale, M, K, N, opt=False)

        def lut_opt():
            # 优化版用 (n_blocks, N) 布局
            w_code_t = w_code.t().contiguous()
            return rust_lut(x_q, w_code_t, scale, M, K, N, opt=True)

        def lut_mt():
            # 多线程版用 (n_blocks, N) 布局
            w_code_t = w_code.t().contiguous()
            out = torch.zeros(M, N, dtype=torch.float32)
            lib.lut_matmul_mt(x_q.data_ptr(), w_code_t.data_ptr(), out.data_ptr(), M, K, N)
            return out / scale

        # 紧凑 LUT：预计算唯一码
        n_blocks = K // 4
        w_code_t = w_code.t().contiguous()  # (n_blocks, N)
        unique_codes, unique_counts, code_to_idx = precompute_codes(w_code_t, n_blocks, N)
        max_unique = unique_codes.shape[1]
        avg_unique = unique_counts.float().mean().item()

        def lut_compact():
            out = torch.zeros(M, N, dtype=torch.float32)
            lib.lut_matmul_compact(
                x_q.data_ptr(), w_code_t.data_ptr(),
                unique_codes.data_ptr(), unique_counts.data_ptr(), code_to_idx.data_ptr(),
                out.data_ptr(), M, K, N, max_unique)
            return out / scale

        def lut_blocked():
            out = torch.zeros(M, N, dtype=torch.float32)
            lib.lut_matmul_blocked(x_q.data_ptr(), w_code_t.data_ptr(), out.data_ptr(), M, K, N)
            return out / scale

        def lut_avx2():
            out = torch.zeros(M, N, dtype=torch.float32)
            lib.lut_matmul_avx2(x_q.data_ptr(), w_code_t.data_ptr(), out.data_ptr(), M, K, N)
            return out / scale

        # 正确性（对比稠密，考虑 int8 量化误差）
        out_dense = dense()
        out_lut = lut_basic()
        err = (out_dense - out_lut).abs().mean().item()

        t_dense = bench(dense)
        t_basic = bench(lut_basic)
        t_blocked = bench(lut_blocked)
        t_avx2 = bench(lut_avx2)
        print(f"M={M} K={K} N={N}: 稠密={t_dense*1000:.3f}ms "
              f"完整LUT={t_basic*1000:.3f}ms({t_dense/t_basic:.2f}x) "
              f"blocked={t_blocked*1000:.3f}ms({t_dense/t_blocked:.2f}x) "
              f"AVX2={t_avx2*1000:.3f}ms({t_dense/t_avx2:.2f}x) 误差={err:.4f}")


if __name__ == "__main__":
    main()
