//! Rust LUT (Lookup Table) 稀疏 matmul 核。
//!
//! 参考 BitNet 的 three_lut_ctor：
//!   1. 三元权重 {-1,0,+1} 打包为 2-bit 码（-1→0, 0→1, +1→2）
//!   2. 激活为 int8
//!   3. 按权重块（B=4）预计算 LUT：LUT[code] = 块内激活与权重点积
//!   4. matmul 变成查表 + 累加

/// 解码 2-bit 码为三元值：0→-1, 1→0, 2→+1, 3→0
#[inline(always)]
fn decode_2bit(code: u32) -> i32 {
    match code {
        0 => -1,
        2 => 1,
        _ => 0,
    }
}

/// LUT matmul。
///
/// # 参数
/// - `x`: (M, K) int8 激活，行主序
/// - `w_code`: (N, K/4) 打包的 2-bit 三元权重码
/// - `out`: (M, N) f32 输出
/// - `m, k, n`: 维度
#[no_mangle]
pub extern "C" fn lut_matmul(
    x: *const i8,
    w_code: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let n_blocks = k / 4;
    let x = unsafe { std::slice::from_raw_parts(x, m * k) };
    let w_code = unsafe { std::slice::from_raw_parts(w_code, n * n_blocks) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m * n) };

    for mm in 0..m {
        for bb in 0..n_blocks {
            // 构建 LUT（256 项）
            let base = mm * k + bb * 4;
            let x0 = x[base] as i32;
            let x1 = x[base + 1] as i32;
            let x2 = x[base + 2] as i32;
            let x3 = x[base + 3] as i32;
            let mut lut = [0i32; 256];
            for code in 0..256usize {
                let c = code as u32;
                let sum = x0 * decode_2bit(c & 3)
                    + x1 * decode_2bit((c >> 2) & 3)
                    + x2 * decode_2bit((c >> 4) & 3)
                    + x3 * decode_2bit((c >> 6) & 3);
                lut[code] = sum;
            }
            // 查表累加
            for nn in 0..n {
                let code = w_code[nn * n_blocks + bb] as usize;
                out[mm * n + nn] += lut[code] as f32;
            }
        }
    }
}

/// 优化版 LUT matmul：两阶段（先构建全部 LUT，再 gather），改善缓存局部性。
/// w_code 布局为 (n_blocks, N)，使 gather 连续访问。
#[no_mangle]
pub extern "C" fn lut_matmul_opt(
    x: *const i8,
    w_code: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let n_blocks = k / 4;
    let x = unsafe { std::slice::from_raw_parts(x, m * k) };
    // w_code 布局: (n_blocks, N)
    let w_code = unsafe { std::slice::from_raw_parts(w_code, n_blocks * n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m * n) };

    for mm in 0..m {
        // 阶段1：构建全部 LUT（n_blocks × 256）
        let mut lut_all = vec![0i32; n_blocks * 256];
        for bb in 0..n_blocks {
            let base = mm * k + bb * 4;
            let x0 = x[base] as i32;
            let x1 = x[base + 1] as i32;
            let x2 = x[base + 2] as i32;
            let x3 = x[base + 3] as i32;
            let lut = &mut lut_all[bb * 256..(bb + 1) * 256];
            for code in 0..256usize {
                let c = code as u32;
                lut[code] = x0 * decode_2bit(c & 3)
                    + x1 * decode_2bit((c >> 2) & 3)
                    + x2 * decode_2bit((c >> 4) & 3)
                    + x3 * decode_2bit((c >> 6) & 3);
            }
        }
        // 阶段2：gather（连续访问 w_code）
        for nn in 0..n {
            let mut acc = 0i32;
            for bb in 0..n_blocks {
                let code = w_code[bb * n + nn] as usize;
                acc += lut_all[bb * 256 + code];
            }
            out[mm * n + nn] = acc as f32;
        }
    }
}

/// 紧凑 LUT matmul：只构建每块实际用到的唯一权重码，避免完整 256 项 LUT。
///
/// # 参数
/// - `x`: (m, k) int8 激活
/// - `w_code`: (n_blocks, n) 权重码
/// - `unique_codes`: (n_blocks, max_unique) 每块唯一权重码
/// - `unique_counts`: (n_blocks,) 每块唯一码数量
/// - `code_to_idx`: (n_blocks, 256) 码 -> 唯一码索引（-1 表示未用）
/// - `out`: (m, n) f32 输出
#[no_mangle]
pub extern "C" fn lut_matmul_compact(
    x: *const i8,
    w_code: *const u8,
    unique_codes: *const u8,
    unique_counts: *const u32,
    code_to_idx: *const i32,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    max_unique: usize,
) {
    let n_blocks = k / 4;
    let x = unsafe { std::slice::from_raw_parts(x, m * k) };
    let w_code = unsafe { std::slice::from_raw_parts(w_code, n_blocks * n) };
    let unique_codes = unsafe { std::slice::from_raw_parts(unique_codes, n_blocks * max_unique) };
    let unique_counts = unsafe { std::slice::from_raw_parts(unique_counts, n_blocks) };
    let code_to_idx = unsafe { std::slice::from_raw_parts(code_to_idx, n_blocks * 256) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m * n) };

    for mm in 0..m {
        // 构建紧凑 LUT：每块只算唯一码
        let mut lut = vec![0i32; n_blocks * max_unique];
        for bb in 0..n_blocks {
            let base = mm * k + bb * 4;
            let x0 = x[base] as i32;
            let x1 = x[base + 1] as i32;
            let x2 = x[base + 2] as i32;
            let x3 = x[base + 3] as i32;
            let cnt = unique_counts[bb] as usize;
            for u in 0..cnt {
                let code = unique_codes[bb * max_unique + u] as u32;
                lut[bb * max_unique + u] = x0 * decode_2bit(code & 3)
                    + x1 * decode_2bit((code >> 2) & 3)
                    + x2 * decode_2bit((code >> 4) & 3)
                    + x3 * decode_2bit((code >> 6) & 3);
            }
        }
        // gather：用 code_to_idx 映射到紧凑 LUT
        for nn in 0..n {
            let mut acc = 0i32;
            for bb in 0..n_blocks {
                let code = w_code[bb * n + nn] as usize;
                let idx = code_to_idx[bb * 256 + code];
                acc += lut[bb * max_unique + idx as usize];
            }
            out[mm * n + nn] = acc as f32;
        }
    }
}

/// 多线程版 LUT matmul：gather 按 nn 并行。
#[no_mangle]
pub extern "C" fn lut_matmul_mt(
    x: *const i8,
    w_code: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let n_blocks = k / 4;
    let x = unsafe { std::slice::from_raw_parts(x, m * k) };
    let w_code = unsafe { std::slice::from_raw_parts(w_code, n_blocks * n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m * n) };

    for mm in 0..m {
        // 阶段1：构建全部 LUT
        let mut lut_all = vec![0i32; n_blocks * 256];
        for bb in 0..n_blocks {
            let base = mm * k + bb * 4;
            let x0 = x[base] as i32;
            let x1 = x[base + 1] as i32;
            let x2 = x[base + 2] as i32;
            let x3 = x[base + 3] as i32;
            let lut = &mut lut_all[bb * 256..(bb + 1) * 256];
            for code in 0..256usize {
                let c = code as u32;
                lut[code] = x0 * decode_2bit(c & 3)
                    + x1 * decode_2bit((c >> 2) & 3)
                    + x2 * decode_2bit((c >> 4) & 3)
                    + x3 * decode_2bit((c >> 6) & 3);
            }
        }
        // 阶段2：gather 并行（用 chunks_mut 获取不相交切片）
        let n_threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
        let chunk = (n + n_threads - 1) / n_threads;
        let lut_all = &lut_all;
        let w_code = &w_code;
        let out = &mut out[mm * n..(mm + 1) * n];
        std::thread::scope(|s| {
            for (t, out_slice) in out.chunks_mut(chunk).enumerate() {
                let start = t * chunk;
                s.spawn(move || {
                    for (i, nn) in (start..start + out_slice.len()).enumerate() {
                        let mut acc = 0i32;
                        for bb in 0..n_blocks {
                            let code = w_code[bb * n + nn] as usize;
                            acc += lut_all[bb * 256 + code];
                        }
                        out_slice[i] = acc as f32;
                    }
                });
            }
        });
    }
}

/// 优化1：缓存局部性。块外循环 + nn 内循环，w_code 连续访问，LUT 块驻留 L1 缓存。
/// w_code 布局: (n_blocks, n)。
#[no_mangle]
pub extern "C" fn lut_matmul_blocked(
    x: *const i8,
    w_code: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let n_blocks = k / 4;
    let x = unsafe { std::slice::from_raw_parts(x, m * k) };
    let w_code = unsafe { std::slice::from_raw_parts(w_code, n_blocks * n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, m * n) };

    for mm in 0..m {
        // 构建全部 LUT
        let mut lut_all = vec![0i32; n_blocks * 256];
        for bb in 0..n_blocks {
            let base = mm * k + bb * 4;
            let x0 = x[base] as i32;
            let x1 = x[base + 1] as i32;
            let x2 = x[base + 2] as i32;
            let x3 = x[base + 3] as i32;
            let lut = &mut lut_all[bb * 256..(bb + 1) * 256];
            for code in 0..256usize {
                let c = code as u32;
                lut[code] = x0 * decode_2bit(c & 3)
                    + x1 * decode_2bit((c >> 2) & 3)
                    + x2 * decode_2bit((c >> 4) & 3)
                    + x3 * decode_2bit((c >> 6) & 3);
            }
        }
        // gather：块外循环，nn 内循环（w_code 连续，LUT 块驻留缓存）
        for bb in 0..n_blocks {
            let lut = &lut_all[bb * 256..(bb + 1) * 256];
            let wc = &w_code[bb * n..(bb + 1) * n];
            let out_row = &mut out[mm * n..(mm + 1) * n];
            for nn in 0..n {
                out_row[nn] += lut[wc[nn] as usize] as f32;
            }
        }
    }
}

/// 优化2：AVX2 SIMD gather。每块用 AVX2 一次处理 8 个输出。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lut_matmul_avx2_impl(
    x: *const i8,
    w_code: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let n_blocks = k / 4;
    let x = std::slice::from_raw_parts(x, m * k);
    let w_code = std::slice::from_raw_parts(w_code, n_blocks * n);
    let out = std::slice::from_raw_parts_mut(out, m * n);

    for mm in 0..m {
        // 构建全部 LUT
        let mut lut_all = vec![0i32; n_blocks * 256];
        for bb in 0..n_blocks {
            let base = mm * k + bb * 4;
            let x0 = x[base] as i32;
            let x1 = x[base + 1] as i32;
            let x2 = x[base + 2] as i32;
            let x3 = x[base + 3] as i32;
            let lut = &mut lut_all[bb * 256..(bb + 1) * 256];
            for code in 0..256usize {
                let c = code as u32;
                lut[code] = x0 * decode_2bit(c & 3)
                    + x1 * decode_2bit((c >> 2) & 3)
                    + x2 * decode_2bit((c >> 4) & 3)
                    + x3 * decode_2bit((c >> 6) & 3);
            }
        }
        // gather：AVX2 一次 8 个输出
        for bb in 0..n_blocks {
            let lut = &lut_all[bb * 256..(bb + 1) * 256];
            let wc = &w_code[bb * n..(bb + 1) * n];
            let out_row = &mut out[mm * n..(mm + 1) * n];
            let lut_ptr = lut.as_ptr() as *const i32;
            let mut nn = 0;
            while nn + 8 <= n {
                // 加载 8 个权重码字节，零扩展为 8 个 i32 索引
                let codes = _mm_loadu_si128(wc[nn..].as_ptr() as *const __m128i);
                let idx = _mm256_cvtepu8_epi32(codes);
                // gather 8 个 LUT 值
                let gathered = _mm256_i32gather_epi32(lut_ptr, idx, 4);
                // 累加到 out
                let out_vec = _mm256_loadu_ps(out_row[nn..].as_ptr());
                let gathered_f = _mm256_cvtepi32_ps(gathered);
                let new_out = _mm256_add_ps(out_vec, gathered_f);
                _mm256_storeu_ps(out_row[nn..].as_mut_ptr(), new_out);
                nn += 8;
            }
            // 处理剩余
            for i in nn..n {
                out_row[i] += lut[wc[i] as usize] as f32;
            }
        }
    }
}

/// AVX2 版入口：运行时检测 AVX2，不支持则回退到 blocked 版。
#[no_mangle]
pub extern "C" fn lut_matmul_avx2(
    x: *const i8,
    w_code: *const u8,
    out: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { lut_matmul_avx2_impl(x, w_code, out, m, k, n) };
            return;
        }
    }
    lut_matmul_blocked(x, w_code, out, m, k, n);
}
