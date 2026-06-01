"""Three-way comparison.

  baseline    — always-masked (kernel writes mask on every load/store)
  manual_if   — programmer-written `if (full_tile): no_mask else: mask` in kernel
  quad        — our 4-variant tail-guard dispatch

Bench all three vs baseline (= 1.0x). interleaved + min-of-many.
"""
import os, time, torch, triton, triton.language as tl
triton.runtime.driver.set_active_to_cpu()

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
N_WARMUP = 50
N_ROUNDS = 300


@triton.jit
def matmul_baseline(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
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
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def matmul_manual_if(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                     stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    full_m = (pid_m + 1) * BLOCK_M <= M
    full_n = (pid_n + 1) * BLOCK_N <= N
    if full_m and full_n:
        # main path: no M/N mask, keep K mask
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)
    else:
        # tail path: full mask
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(kernel, A, B, C):
    M, K = A.shape; K2, N = B.shape
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](A, B, C, M, N, K,
                 A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                 C.stride(0), C.stride(1),
                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)


def bench3(fn_base, fn_man, fn_quad):
    for _ in range(N_WARMUP):
        fn_base(); fn_man(); fn_quad()
    bt, mt, qt = [], [], []
    for _ in range(N_ROUNDS):
        t0 = time.perf_counter(); fn_base(); t1 = time.perf_counter()
        t2 = time.perf_counter(); fn_man(); t3 = time.perf_counter()
        t4 = time.perf_counter(); fn_quad(); t5 = time.perf_counter()
        bt.append((t1 - t0) * 1e6)
        mt.append((t3 - t2) * 1e6)
        qt.append((t5 - t4) * 1e6)
    return bt, mt, qt


def stats(times):
    s = sorted(times)
    n = len(s)
    return s[0], s[n // 10], s[n // 2]


cases = [
    (300, 300, 300, 'small tail'),
    (512, 512, 512, 'mid aligned'),
    (500, 500, 500, 'mid tail'),
    (1024, 1024, 1024, 'large aligned'),
    (1000, 1000, 1000, 'large tail'),
    (1000, 800, 900, 'asymm tail'),
    (2048, 2048, 2048, 'xlarge aligned'),
    (2000, 2000, 2000, 'xlarge tail'),
]

print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}  rounds={N_ROUNDS}  warmup={N_WARMUP}  BLOCK={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
print(f"{'label':16s} {'shape':>14s}     "
      f"{'baseline med':>13s}     "
      f"{'manual_if':>10s}  {'mfif/base':>9s}     "
      f"{'quad':>9s}  {'quad/base':>9s}")
print('-' * 110)
for M, N, K, lbl in cases:
    A = torch.randn(M, K); B = torch.randn(K, N)

    os.environ['TRITON_CPU_TAIL_GUARD'] = '0'
    Cbase = torch.zeros(M, N); run(matmul_baseline, A, B, Cbase)
    Cman  = torch.zeros(M, N); run(matmul_manual_if, A, B, Cman)
    os.environ['TRITON_CPU_TAIL_GUARD'] = '1'
    Cquad = torch.zeros(M, N); run(matmul_baseline, A, B, Cquad)

    def fn_base():
        os.environ['TRITON_CPU_TAIL_GUARD'] = '0'
        run(matmul_baseline, A, B, Cbase)
    def fn_man():
        os.environ['TRITON_CPU_TAIL_GUARD'] = '0'
        run(matmul_manual_if, A, B, Cman)
    def fn_quad():
        os.environ['TRITON_CPU_TAIL_GUARD'] = '1'
        run(matmul_baseline, A, B, Cquad)

    bt, mt, qt = bench3(fn_base, fn_man, fn_quad)
    b_min, b_p10, b_med = stats(bt)
    m_min, m_p10, m_med = stats(mt)
    q_min, q_p10, q_med = stats(qt)
    sp_m = b_med / m_med
    sp_q = b_med / q_med
    print(f'{lbl:16s} {f"{M}x{N}x{K}":>14s}     '
          f'{b_med:13.1f}     '
          f'{m_med:10.1f}  {sp_m:8.3f}x     '
          f'{q_med:9.1f}  {sp_q:8.3f}x')
