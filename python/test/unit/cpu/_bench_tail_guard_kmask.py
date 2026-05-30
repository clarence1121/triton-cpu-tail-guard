"""
Controlled experiment to isolate WHY removing K mask causes regression.

Two competing hypotheses:
  H1: removing the K mask from load/store hurts LLVM vectorization
  H2: changing loop range from range(0,K,BK) to range(0,_k_full,BK) hurts LLVM

We test 5 variants of matmul (K always divisible by BK, so masks are always-true):
  orig       : all masks, range(0,K,BK)
  no_mn      : M/N masks removed, K mask kept, range(0,K,BK)   <-- our working fast variant
  no_k       : M/N masks removed, K mask removed, range(0,K,BK)   isolates H1
  peeled_km  : M/N removed, K mask KEPT, range(0,_k_full,BK)     isolates H2
  peeled_all : all masks removed, range(0,_k_full,BK)            original peeled variant

If no_k << no_mn  → H1 confirmed (mask removal hurts)
If peeled_km << no_mn → H2 confirmed (range change hurts)
"""
import os, time, torch, triton, triton.language as tl
triton.runtime.driver.set_active_to_cpu()

BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32


@triton.jit
def matmul_orig(A, B, C, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_k = offs_k[None, :] + k < K
        a = tl.load(A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
                    mask=mask_m & mask_k, other=0.0)
        b = tl.load(B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
                    mask=(offs_k[:, None] + k < K) & mask_n, other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=mask_m & mask_n)


@triton.jit
def matmul_no_mn(A, B, C, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """M/N masks removed, K mask kept."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_k = offs_k[None, :] + k < K
        a = tl.load(A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
                    mask=mask_k, other=0.0)
        b = tl.load(B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
                    mask=(offs_k[:, None] + k < K), other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


@triton.jit
def matmul_no_k(A, B, C, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """M/N AND K masks removed, range still (0,K,BK). K must be divisible by BLOCK_K."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak))
        b = tl.load(B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn))
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


@triton.jit
def matmul_peeled_kmask(A, B, C, M, N, K,
                        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """M/N removed, K mask KEPT, but range changed to (0,_k_full,BK). K divisible."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    _k_full = (K // BLOCK_K) * BLOCK_K
    for k in range(0, _k_full, BLOCK_K):
        mask_k = offs_k[None, :] + k < K
        a = tl.load(A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
                    mask=mask_k, other=0.0)
        b = tl.load(B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
                    mask=(offs_k[:, None] + k < K), other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


@triton.jit
def matmul_peeled_all(A, B, C, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """All masks removed, range=(0,_k_full,BK). K divisible - the original peeled variant."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    _k_full = (K // BLOCK_K) * BLOCK_K
    for k in range(0, _k_full, BLOCK_K):
        a = tl.load(A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak))
        b = tl.load(B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn))
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


def run(fn, A, B, C):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and K % BLOCK_K == 0 and M % BLOCK_M == 0 and N % BLOCK_N == 0
    grid = (M // BLOCK_M, N // BLOCK_N)
    fn[grid](A, B, C, M, N, K,
             A.stride(0), A.stride(1), B.stride(0), B.stride(1),
             C.stride(0), C.stride(1),
             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)


def bench(fn, warmup=30, iters=300):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters * 1e6


VARIANTS = [
    ("orig",        matmul_orig,        "all masks, range(0,K,BK)"),
    ("no_mn",       matmul_no_mn,       "M/N removed, K kept, range(0,K,BK)   [fast baseline]"),
    ("no_k",        matmul_no_k,        "ALL removed, range(0,K,BK)            [isolates K mask effect]"),
    ("peeled_km",   matmul_peeled_kmask,"M/N removed, K kept, range(_k_full)   [isolates range change]"),
    ("peeled_all",  matmul_peeled_all,  "ALL removed, range(_k_full)           [original peeled variant]"),
]

print("Hypothesis H1: K mask removal hurts  →  no_k should be slow")
print("Hypothesis H2: range change hurts    →  peeled_km should be slow")
print()

for sz in [256, 512, 1024]:
    M = N = K = sz
    A = torch.randn(M, K)
    B = torch.randn(K, N)

    print(f"=== {M}×{N}×{K} ===")
    print(f"  {'variant':<14}  {'µs':>8}  {'vs orig':>9}  {'vs no_mn':>9}  notes")
    print(f"  {'-'*14}  {'-'*8}  {'-'*9}  {'-'*9}  -----")

    times = {}
    for name, fn, notes in VARIANTS:
        C = torch.zeros(M, N)
        t = bench(lambda fn=fn, A=A, B=B, C=C: run(fn, A, B, C))
        times[name] = t

    t_orig  = times["orig"]
    t_no_mn = times["no_mn"]
    for name, fn, notes in VARIANTS:
        t = times[name]
        vs_orig  = t / t_orig
        vs_no_mn = t / t_no_mn
        marker = ""
        if name == "no_k"      and vs_no_mn > 1.05: marker = "  ← H1 confirmed"
        if name == "peeled_km" and vs_no_mn > 1.05: marker = "  ← H2 confirmed"
        if name == "no_k"      and vs_no_mn < 1.05: marker = "  ← H1 NOT confirmed"
        if name == "peeled_km" and vs_no_mn < 1.05: marker = "  ← H2 NOT confirmed"
        print(f"  {name:<14}  {t:8.1f}  {vs_orig:8.3f}x  {vs_no_mn:8.3f}x  {notes}{marker}")
    print()
