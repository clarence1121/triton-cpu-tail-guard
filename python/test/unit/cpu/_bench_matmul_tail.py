"""
2D tail-guard benchmark for matmul.

Run:
    conda run -n triton-cpu python python/test/unit/cpu/_bench_matmul_tail.py

Compares:
  - disabled  : TRITON_CPU_TAIL_GUARD=0  (all tiles run masked corner kernel)
  - enabled   : TRITON_CPU_TAIL_GUARD=1  (quad dispatch: main/m_tail/n_tail/corner)
"""
import os
import time

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active_to_cpu()

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run_matmul(A, B, C, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness():
    print("=== Correctness ===")
    cases = [
        (64,  64,  64,  "aligned"),
        (100, 100, 100, "both tails"),
        (128, 100, 64,  "N-tail only"),
        (100, 128, 64,  "M-tail only"),
        (65,  65,  33,  "all tails"),
        (256, 256, 256, "large aligned"),
        (300, 300, 300, "large both tails"),
    ]
    for M, N, K, label in cases:
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        C = torch.zeros(M, N, dtype=torch.float32)
        run_matmul(A, B, C)
        ref = A @ B
        err = (C - ref).abs().max().item()
        ok = err < 1e-3
        print(f"  {label:20s} ({M}x{K}x{N}): {'PASS' if ok else 'FAIL'}  err={err:.2e}")
    print()


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def benchmark():
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    # Cases: (M, N, K, label)
    # "tail fraction" = fraction of tiles that are NOT the main variant
    cases = [
        (64,   64,   64,   "aligned (0% tail)"),
        (320,  320,  320,  "small aligned (0%)"),
        (300,  300,  300,  "small tail (~6%)"),
        (512,  512,  512,  "mid aligned (0%)"),
        (500,  500,  500,  "mid tail (~4%)"),
        (1024, 1024, 1024, "large aligned (0%)"),
        (1000, 1000, 1000, "large tail (~2%)"),
        (1000, 900,  800,  "asymmetric tail"),
    ]

    print(f"{'shape':>28}  {'enabled µs':>12}  {'disabled µs':>13}  {'speedup':>8}  {'TFLOPS on':>10}")
    print("-" * 80)

    for M, N, K, label in cases:
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)

        # Enabled
        os.environ["TRITON_CPU_TAIL_GUARD"] = "1"
        C_on = torch.zeros(M, N, dtype=torch.float32)
        run_matmul(A, B, C_on, BLOCK_M, BLOCK_N, BLOCK_K)  # warm compile
        t_on = bench(lambda: run_matmul(A, B, C_on, BLOCK_M, BLOCK_N, BLOCK_K))

        # Disabled — needs a fresh kernel (different cache key via env var)
        os.environ["TRITON_CPU_TAIL_GUARD"] = "0"
        C_off = torch.zeros(M, N, dtype=torch.float32)
        run_matmul(A, B, C_off, BLOCK_M, BLOCK_N, BLOCK_K)  # warm compile
        t_off = bench(lambda: run_matmul(A, B, C_off, BLOCK_M, BLOCK_N, BLOCK_K))

        os.environ["TRITON_CPU_TAIL_GUARD"] = "1"

        speedup = t_off / t_on
        flops = 2 * M * N * K
        tflops_on = flops / (t_on * 1e-6) / 1e12

        shape = f"({M}x{K}x{N})"
        print(f"  {label:26s} {shape:>10}  {t_on:12.1f}  {t_off:13.1f}  {speedup:8.3f}x  {tflops_on:10.3f}")


if __name__ == "__main__":
    check_correctness()
    benchmark()
