"""
Causal Attention PID-Variant Benchmark for Triton-CPU
======================================================

Motivating example for heterogeneous pid-class execution.

Tile classification by (pid_q, pid_k) coordinate:
  Full tile     pid_k < pid_q  →  dense dot product, no causal mask
  Diagonal tile pid_k == pid_q →  dot product + lower-triangular mask
  Invalid tile  pid_k > pid_q  →  zero useful work, should be skipped

Four execution strategies:
  A  Generalized   all T² tiles dispatched; element-level causal mask throughout
  B  PID-branch    all T² tiles dispatched; pid-level early return for invalid tiles
  C  Bucketed      separate kernels per class; invalid tiles never dispatched
  D  Bucketed-masked  valid tiles only (like C), but all use the same masked body

Usage:
  pytest test_pid_variants.py --device cpu            # correctness only
  python test_pid_variants.py                          # correctness + benchmark
  python test_pid_variants.py --no-bench               # correctness only
"""

import argparse
import math
import sys
import time

import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime.driver import driver
from triton.compiler.cpu_pid_region import RegionPlan, RegionSpec, region_dispatch

# ---------------------------------------------------------------------------
# Tile geometry helpers
# ---------------------------------------------------------------------------

def tile_stats(N: int, B: int) -> dict:
    T = N // B
    total   = T * T
    invalid = T * (T - 1) // 2   # upper triangle excl. diagonal
    diag    = T
    full    = T * (T - 1) // 2   # lower triangle excl. diagonal
    assert full + diag + invalid == total
    return dict(T=T, total=total, full=full, diag=diag, invalid=invalid)


def effective_flops(N: int, B: int, D: int) -> int:
    """FLOPs for valid lower-triangular tiles (multiply-add counts as 2)."""
    s = tile_stats(N, B)
    return 2 * (s["full"] + s["diag"]) * B * B * D


# ---------------------------------------------------------------------------
# Version A: Generalized masked kernel  (GPU-portable baseline)
# ---------------------------------------------------------------------------

@triton.jit
def _kernel_A(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)

    acc = tl.dot(Q_tile, tl.trans(K_tile))

    # Element-level causal mask — present in ALL tiles, including invalid ones.
    # This inhibits clean SIMD code generation for the store path.
    row_idx = offs_q[:, None]
    col_idx = offs_k[None, :]
    acc = tl.where(col_idx <= row_idx, acc, -1e9)

    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def launch_A(Q, K, S, N: int, B: int, D: int):
    T = N // B
    _kernel_A[(T, T)](
        Q, K, S,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        S.stride(0), S.stride(1),
        BQ=B, BK=B, HEAD_DIM=D,
    )


# ---------------------------------------------------------------------------
# Version B: Single-body with pid-level branching
# ---------------------------------------------------------------------------

@triton.jit
def _kernel_B(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)

    # pid-level early exit: invalid tiles do no work.
    # On CPU this is a real scalar branch (no warp divergence concept).
    if pid_k > pid_q:
        return

    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)

    acc = tl.dot(Q_tile, tl.trans(K_tile))

    if pid_k == pid_q:
        # Diagonal tile: apply lower-triangular mask.
        row_idx = offs_q[:, None]
        col_idx = offs_k[None, :]
        acc = tl.where(col_idx <= row_idx, acc, -1e9)
    # else: full tile — acc is correct, no mask needed.

    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def launch_B(Q, K, S, N: int, B: int, D: int):
    T = N // B
    _kernel_B[(T, T)](
        Q, K, S,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        S.stride(0), S.stride(1),
        BQ=B, BK=B, HEAD_DIM=D,
    )


# ---------------------------------------------------------------------------
# Version F: Compiler-automated pid-region dispatch
# ---------------------------------------------------------------------------
# Same source as Version B.  When enable_pid_region_dispatch=True the compiler
# detects the `if pid_k > pid_q: return` / `if pid_k == pid_q: ...` pattern,
# synthesizes two branch-free variants (full + diag), and dispatches via the
# C-level launch_lower_tri_2d / launch_diagonal_2d — no manual bucket tables,
# no manually written variant kernels.
# ---------------------------------------------------------------------------

def launch_F(Q, K, S, N: int, B: int, D: int):
    T = N // B
    _kernel_B[(T, T)](
        Q, K, S,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        S.stride(0), S.stride(1),
        BQ=B, BK=B, HEAD_DIM=D,
        enable_pid_region_dispatch=True,
    )


# ---------------------------------------------------------------------------
# Version G: Explicit @region_dispatch decorator API
# ---------------------------------------------------------------------------
# Same body as Version B/F, but opts in via the decorator rather than a
# per-call flag.  The compiler detects the plan at first compilation and
# dispatches via C-level launchers for every subsequent call — no flag needed.
# ---------------------------------------------------------------------------

_causal_plan = RegionPlan(
    pid_vars=["pid_q", "pid_k"],
    regions=[
        RegionSpec("full", "pid_k < pid_q", "region_2d_lt"),
        RegionSpec("diag", "pid_k == pid_q", "region_2d_eq"),
    ],
)


@region_dispatch(_causal_plan)
@triton.jit
def _kernel_G(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k > pid_q:
        return
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)
    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q_tile, tl.trans(K_tile))
    if pid_k == pid_q:
        row_idx = offs_q[:, None]
        col_idx = offs_k[None, :]
        acc = tl.where(col_idx <= row_idx, acc, -1e9)
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def launch_G(Q, K, S, N: int, B: int, D: int):
    T = N // B
    _kernel_G[(T, T)](
        Q, K, S,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        S.stride(0), S.stride(1),
        BQ=B, BK=B, HEAD_DIM=D,
    )


# ---------------------------------------------------------------------------
# Version C: Bucketed per-variant execution
# ---------------------------------------------------------------------------

@triton.jit
def _kernel_C_full(
    Q_ptr, K_ptr, S_ptr,
    pq_ptr, pk_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Full-tile variant.  Invariant: pid_k < pid_q established by bucket construction.
    Zero masking — compiler sees a pure dense dot + store, enabling clean SIMD."""
    idx   = tl.program_id(0)
    pid_q = tl.load(pq_ptr + idx)
    pid_k = tl.load(pk_ptr + idx)

    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)

    acc = tl.dot(Q_tile, tl.trans(K_tile))   # no predicate anywhere in this function

    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


@triton.jit
def _kernel_C_diagonal(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Diagonal-tile variant.  Invariant: pid_k == pid_q established by dispatch.
    Mask pattern is a *static* lower-triangular BQ×BK block (j <= i within tile),
    allowing the compiler to treat it as a compile-time constant structure."""
    pid_q = tl.program_id(0)
    pid_k = pid_q   # guaranteed equal by dispatch

    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)

    acc = tl.dot(Q_tile, tl.trans(K_tile))

    # Within-tile lower-triangular mask: col <= row iff j_idx <= i_idx.
    # This is a constexpr-shaped predicate, not a data-dependent one.
    i_idx = tl.arange(0, BQ)[:, None]
    j_idx = tl.arange(0, BK)[None, :]
    acc = tl.where(j_idx <= i_idx, acc, -1e9)

    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


class TileBuckets:
    """Pre-computed tile classification.  Built once, reused across launches."""

    def __init__(self, T: int, B: int, device='cpu'):
        full_pq, full_pk = [], []
        valid_pq, valid_pk = [], []
        for pq in range(T):
            valid_pq.append(pq); valid_pk.append(pq)   # diagonal tile
            for pk in range(pq):                        # full tiles (pk < pq)
                full_pq.append(pq);  full_pk.append(pk)
                valid_pq.append(pq); valid_pk.append(pk)
        self.T  = T
        self.B  = B
        self.num_full  = len(full_pq)
        self.num_valid = len(valid_pq)
        if self.num_full > 0:
            self.full_pq  = torch.tensor(full_pq,  dtype=torch.int32, device=device)
            self.full_pk  = torch.tensor(full_pk,  dtype=torch.int32, device=device)
        else:
            self.full_pq = self.full_pk = None
        self.valid_pq = torch.tensor(valid_pq, dtype=torch.int32, device=device)
        self.valid_pk = torch.tensor(valid_pk, dtype=torch.int32, device=device)

    @property
    def build_cost_ns(self) -> float:
        t0 = time.perf_counter_ns()
        TileBuckets(self.T, self.B)
        return time.perf_counter_ns() - t0


def launch_C(Q, K, S, N: int, B: int, D: int, buckets: TileBuckets):
    T = N // B
    assert buckets.T == T and buckets.B == B

    # Full-tile variant: mask-free dense dot product per tile
    if buckets.num_full > 0:
        _kernel_C_full[(buckets.num_full,)](
            Q, K, S,
            buckets.full_pq, buckets.full_pk,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            S.stride(0), S.stride(1),
            BQ=B, BK=B, HEAD_DIM=D,
        )

    # Diagonal-tile variant: static lower-triangular mask, only T tiles
    _kernel_C_diagonal[(T,)](
        Q, K, S,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        S.stride(0), S.stride(1),
        BQ=B, BK=B, HEAD_DIM=D,
    )
    # Invalid tiles (pid_k > pid_q): never dispatched — zero arithmetic work.


# ---------------------------------------------------------------------------
# Version E: Launcher-mapped regions without bucket lookup
# ---------------------------------------------------------------------------
# This is the OpenMP-side MVP for the target architecture:
#   - invalid upper-triangle tasks are never assigned
#   - full/diagonal tasks use separate branch-free kernels
#   - compact task_id -> logical (pid_q, pid_k) mapping happens in C launcher
# ---------------------------------------------------------------------------

@triton.jit
def _kernel_E_full(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)

    acc = tl.dot(Q_tile, tl.trans(K_tile))
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


@triton.jit
def _kernel_E_diagonal(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)

    acc = tl.dot(Q_tile, tl.trans(K_tile))

    i_idx = tl.arange(0, BQ)[:, None]
    j_idx = tl.arange(0, BK)[None, :]
    acc = tl.where(j_idx <= i_idx, acc, -1e9)

    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


class RegionKernels:

    def __init__(self, Q, K, S, N: int, B: int, D: int):
        T = N // B
        grid = (T, T)
        base_args = (
            Q, K, S,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            S.stride(0), S.stride(1),
        )
        meta = dict(BQ=B, BK=B, HEAD_DIM=D)
        self.T = T
        self.args = base_args + (B, B, D)
        self.full = _kernel_E_full.warmup(*base_args, grid=grid, **meta)
        self.diag = _kernel_E_diagonal.warmup(*base_args, grid=grid, **meta)

    def launch(self):
        stream = driver.active.get_current_stream(driver.active.get_current_device())
        self.full.run_lower_tri_2d(self.T, self.T, 1, stream, None, knobs.runtime.launch_enter_hook,
                                   knobs.runtime.launch_exit_hook, *self.args)
        self.diag.run_diagonal_2d(self.T, self.T, 1, stream, None, knobs.runtime.launch_enter_hook,
                                  knobs.runtime.launch_exit_hook, *self.args)


# ---------------------------------------------------------------------------
# Version D: Bucketed-masked  (valid tiles only, but all use masked body)
# ---------------------------------------------------------------------------
# Controls for arithmetic work (same as C: no invalid tiles).
# Intentionally does NOT specialize variants — full tiles still use tl.where.
# D vs A isolates: pure skip-invalid-tiles benefit
# D vs C isolates: pure mask-free SIMD codegen benefit
# ---------------------------------------------------------------------------

@triton.jit
def _kernel_D(
    Q_ptr, K_ptr, S_ptr,
    pq_ptr, pk_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Valid-only dispatch (skip invalid tiles), but uses element-level tl.where mask
    for ALL tiles — including full tiles that don't need it.
    This is the 'all-masked with smart dispatch' baseline."""
    idx   = tl.program_id(0)
    pid_q = tl.load(pq_ptr + idx)
    pid_k = tl.load(pk_ptr + idx)

    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)

    acc = tl.dot(Q_tile, tl.trans(K_tile))

    # Always apply causal mask — even for full tiles where it's always True.
    # Compiler cannot eliminate this because pid_q/pid_k are runtime values.
    row_idx = offs_q[:, None]
    col_idx = offs_k[None, :]
    acc = tl.where(col_idx <= row_idx, acc, -1e9)

    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def launch_D(Q, K, S, N: int, B: int, D: int, buckets: TileBuckets):
    T = N // B
    assert buckets.T == T and buckets.B == B
    # Dispatch all valid tiles (full + diagonal) with the single masked kernel.
    _kernel_D[(buckets.num_valid,)](
        Q, K, S,
        buckets.valid_pq, buckets.valid_pk,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        S.stride(0), S.stride(1),
        BQ=B, BK=B, HEAD_DIM=D,
    )


# ---------------------------------------------------------------------------
# Micro-benchmark: isolated mask overhead on full tiles
# ---------------------------------------------------------------------------
# Both kernels process the SAME tiles with the SAME lookup table.
# The only difference is whether tl.where is present in the body.
# This isolates pure "mask predicate" cost from dispatch/launch effects.
# Note: for full tiles (pid_k < pid_q) the mask is always True, so both
# produce identical outputs; the overhead is purely computational.
# ---------------------------------------------------------------------------

@triton.jit
def _kernel_full_unmasked(
    Q_ptr, K_ptr, S_ptr,
    pq_ptr, pk_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    idx   = tl.program_id(0)
    pid_q = tl.load(pq_ptr + idx)
    pid_k = tl.load(pk_ptr + idx)
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)
    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q_tile, tl.trans(K_tile))
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


@triton.jit
def _kernel_full_masked(
    Q_ptr, K_ptr, S_ptr,
    pq_ptr, pk_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    idx   = tl.program_id(0)
    pid_q = tl.load(pq_ptr + idx)
    pid_k = tl.load(pk_ptr + idx)
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, HEAD_DIM)
    Q_tile = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K_tile = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q_tile, tl.trans(K_tile))
    # mask present even though it is always True for full tiles (pid_k < pid_q)
    row_idx = offs_q[:, None]
    col_idx = offs_k[None, :]
    acc = tl.where(col_idx <= row_idx, acc, -1e9)
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def run_mask_overhead_bench(
    N_vals=(512, 1024, 2048),
    B: int = 32,
    D: int = 64,
    warmup: int = 5,
    rep: int = 60,
) -> list[dict]:
    triton.runtime.driver.set_active_to_cpu()
    rows = []
    for N in N_vals:
        T = N // B
        Q = torch.randn(N, D, dtype=torch.float32, device='cpu')
        K = torch.randn(N, D, dtype=torch.float32, device='cpu')
        S = torch.zeros(N, N, dtype=torch.float32, device='cpu')
        buckets = TileBuckets(T, B)
        if buckets.num_full == 0:
            continue

        def _launch(kernel):
            kernel[(buckets.num_full,)](
                Q, K, S,
                buckets.full_pq, buckets.full_pk,
                Q.stride(0), Q.stride(1),
                K.stride(0), K.stride(1),
                S.stride(0), S.stride(1),
                BQ=B, BK=B, HEAD_DIM=D,
            )

        ms_u = _bench_one(lambda: _launch(_kernel_full_unmasked), warmup=warmup, rep=rep)
        ms_m = _bench_one(lambda: _launch(_kernel_full_masked),   warmup=warmup, rep=rep)
        rows.append(dict(
            N=N, T=T, num_full=buckets.num_full,
            ms_unmasked=ms_u, ms_masked=ms_m,
            overhead_pct=100 * (ms_m - ms_u) / ms_u,
        ))
    return rows


def print_mask_overhead_table(rows: list[dict]):
    print(f"\nMask overhead on full tiles only (identical dispatch, body differs by one tl.where)")
    print(f"{'N':>6} {'T':>4} {'#full':>6} | {'unmasked':>10} {'masked':>10} | {'overhead':>10}")
    print("-" * 55)
    for r in rows:
        sign = "+" if r['overhead_pct'] >= 0 else ""
        print(f"{r['N']:>6} {r['T']:>4} {r['num_full']:>6} | "
              f"{r['ms_unmasked']:>10.3f} {r['ms_masked']:>10.3f} | "
              f"{sign}{r['overhead_pct']:>9.1f}%")


# ---------------------------------------------------------------------------
# Reference implementation (PyTorch CPU)
# ---------------------------------------------------------------------------

def causal_attn_ref(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Lower-triangular causal attention score: S[i,j] = Q[i]·K[j] if j<=i else -1e9."""
    N = Q.shape[0]
    S = torch.matmul(Q.float(), K.float().T)
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=Q.device))
    return torch.where(mask, S, torch.full_like(S, -1e9))


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def _check(got: torch.Tensor, ref: torch.Tensor, name: str, atol: float = 1e-2):
    max_err = torch.max(torch.abs(got - ref)).item()
    ok = torch.allclose(got, ref, atol=atol)
    if not ok:
        raise AssertionError(
            f"{name}: max_err={max_err:.4f} (atol={atol})\n"
            f"  ref[0,:8]={ref[0,:8].tolist()}\n"
            f"  got[0,:8]={got[0,:8].tolist()}"
        )
    return max_err


def _run_correctness(N: int, B: int, D: int, device: str, seed: int = 42):
    torch.manual_seed(seed)
    triton.runtime.driver.set_active_to_cpu()

    Q = torch.randn(N, D, dtype=torch.float32, device=device)
    K = torch.randn(N, D, dtype=torch.float32, device=device)
    T = N // B

    ref = causal_attn_ref(Q, K)

    # Version A: all T² tiles, element-level mask
    S_A = torch.full((N, N), -1e9, dtype=torch.float32, device=device)
    launch_A(Q, K, S_A, N, B, D)

    # Version B: pid-branch, still T² dispatched
    S_B = torch.full((N, N), -1e9, dtype=torch.float32, device=device)
    launch_B(Q, K, S_B, N, B, D)

    buckets = TileBuckets(T, B, device=device)

    # Version C: bucketed variants, only valid tiles dispatched
    S_C = torch.full((N, N), -1e9, dtype=torch.float32, device=device)
    launch_C(Q, K, S_C, N, B, D, buckets)

    # Version D: bucketed-masked, valid tiles only but all use masked body
    S_D = torch.full((N, N), -1e9, dtype=torch.float32, device=device)
    launch_D(Q, K, S_D, N, B, D, buckets)

    # Version E: launcher-mapped regions, no bucket lookup
    S_E = torch.full((N, N), -1e9, dtype=torch.float32, device=device)
    RegionKernels(Q, K, S_E, N, B, D).launch()

    # Version F: compiler-automated pid-region dispatch (same source as B)
    S_F = torch.full((N, N), -1e9, dtype=torch.float32, device=device)
    launch_F(Q, K, S_F, N, B, D)

    # Version G: explicit @region_dispatch decorator API
    S_G = torch.full((N, N), -1e9, dtype=torch.float32, device=device)
    launch_G(Q, K, S_G, N, B, D)

    err_A = _check(S_A, ref, "Version A")
    err_B = _check(S_B, ref, "Version B")
    err_C = _check(S_C, ref, "Version C")
    err_D = _check(S_D, ref, "Version D")
    err_E = _check(S_E, ref, "Version E")
    err_F = _check(S_F, ref, "Version F")
    err_G = _check(S_G, ref, "Version G")

    return dict(err_A=err_A, err_B=err_B, err_C=err_C, err_D=err_D, err_E=err_E,
                err_F=err_F, err_G=err_G)


@pytest.fixture(autouse=True)
def _set_cpu(request):
    triton.runtime.driver.set_active_to_cpu()


@pytest.mark.parametrize("N,B,D", [
    (128, 32, 64),
    (256, 32, 64),
    (512, 32, 64),
    (256, 32, 128),
    (512, 64, 64),
])
def test_correctness_all_versions(N, B, D, device):
    if device != "cpu":
        pytest.skip("CPU-only test")
    errs = _run_correctness(N, B, D, device="cpu")
    for name, err in errs.items():
        assert err < 1e-2, f"{name}: {err}"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _bench_one(
    fn, *, warmup: int = 5, rep: int = 50,
) -> float:
    """Return median kernel time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rep):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) * 1e-6)
    times.sort()
    return times[len(times) // 2]


def _trimmed_mean(vals: list[float]) -> float:
    vals = sorted(vals)
    if len(vals) <= 2:
        return sum(vals) / len(vals)
    return sum(vals[1:-1]) / (len(vals) - 2)


def _bench_rounds(fn, *, warmup: int = 5, rep: int = 50, rounds: int = 1) -> float:
    return _trimmed_mean([_bench_one(fn, warmup=warmup, rep=rep) for _ in range(rounds)])


def run_benchmark(
    N_vals=(512, 1024, 2048),
    B: int = 32,
    D: int = 64,
    warmup: int = 5,
    rep: int = 50,
    rounds: int = 1,
) -> list[dict]:
    triton.runtime.driver.set_active_to_cpu()
    rows = []

    for N in N_vals:
        assert N % B == 0, f"N={N} must be divisible by B={B}"
        T = N // B
        Q = torch.randn(N, D, dtype=torch.float32, device='cpu')
        K = torch.randn(N, D, dtype=torch.float32, device='cpu')
        stats = tile_stats(N, B)
        flops = effective_flops(N, B, D)

        # Pre-build Version C buckets (amortised; excluded from kernel timing)
        buckets = TileBuckets(T, B, device='cpu')
        bucket_us = buckets.build_cost_ns * 1e-3

        S = torch.full((N, N), -1e9, dtype=torch.float32, device='cpu')
        region_kernels = RegionKernels(Q, K, S, N, B, D)

        ms_A   = _bench_rounds(lambda: launch_A(Q, K, S, N, B, D), warmup=warmup, rep=rep, rounds=rounds)
        ms_B   = _bench_rounds(lambda: launch_B(Q, K, S, N, B, D), warmup=warmup, rep=rep, rounds=rounds)
        ms_C   = _bench_rounds(lambda: launch_C(Q, K, S, N, B, D, buckets), warmup=warmup, rep=rep, rounds=rounds)
        ms_D   = _bench_rounds(lambda: launch_D(Q, K, S, N, B, D, buckets), warmup=warmup, rep=rep, rounds=rounds)
        ms_E   = _bench_rounds(region_kernels.launch, warmup=warmup, rep=rep, rounds=rounds)
        ms_F   = _bench_rounds(lambda: launch_F(Q, K, S, N, B, D), warmup=warmup, rep=rep, rounds=rounds)
        ms_G   = _bench_rounds(lambda: launch_G(Q, K, S, N, B, D), warmup=warmup, rep=rep, rounds=rounds)
        ms_ref = _bench_rounds(lambda: causal_attn_ref(Q, K), warmup=warmup, rep=rep, rounds=rounds)

        def gflops(ms):
            return flops / (ms * 1e-3) * 1e-9

        rows.append(dict(
            N=N, B=B, D=D, T=T,
            pct_invalid=100 * stats["invalid"] / stats["total"],
            ms_A=ms_A, ms_B=ms_B, ms_C=ms_C, ms_D=ms_D, ms_E=ms_E,
            ms_F=ms_F, ms_G=ms_G, ms_ref=ms_ref,
            gflops_A=gflops(ms_A), gflops_B=gflops(ms_B),
            gflops_C=gflops(ms_C), gflops_D=gflops(ms_D),
            gflops_E=gflops(ms_E), gflops_F=gflops(ms_F), gflops_G=gflops(ms_G),
            # skip-invalid gain: D vs A  (same masked body, different dispatch)
            speedup_DvA=ms_A / ms_D,
            # SIMD-quality gain: C vs D  (same dispatch, different codegen)
            speedup_CvD=ms_D / ms_C,
            # task-mapping gain over native Triton pid-level branch/return
            speedup_CvB=ms_B / ms_C,
            # launcher-region MVP: no bucket lookup, compact OpenMP task mapping
            speedup_EvB=ms_B / ms_E,
            speedup_EvC=ms_C / ms_E,
            # compiler-automated region dispatch (same source as B, auto-transformed)
            speedup_FvB=ms_B / ms_F,
            speedup_FvE=ms_E / ms_F,
            # decorator-based region dispatch (G vs F should be ~1.0x; both are the same compiled code)
            speedup_GvB=ms_B / ms_G,
            speedup_GvF=ms_F / ms_G,
            # combined gain:     C vs A
            speedup_CvA=ms_A / ms_C,
            speedup_EvA=ms_A / ms_E,
            speedup_FvA=ms_A / ms_F,
            speedup_GvA=ms_A / ms_G,
            bucket_build_us=bucket_us,
        ))
    return rows


def print_benchmark_table(rows: list[dict]):
    hdr = (
        f"{'N':>6} {'T':>4} {'inv%':>5} | "
        f"{'A(ms)':>7} {'B(ms)':>7} {'D(ms)':>7} {'C(ms)':>7} "
        f"{'E(ms)':>7} {'F(ms)':>7} {'G(ms)':>7} | "
        f"{'GF/s A':>7} {'GF/s C':>7} {'GF/s F':>7} {'GF/s G':>7} | "
        f"{'D/A':>5} {'C/D':>5} {'C/B':>5} {'E/B':>5} "
        f"{'F/B':>5} {'G/B':>5} {'G/F':>5} {'G/A':>5} | "
        f"{'bkt µs':>7}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['N']:>6} {r['T']:>4} {r['pct_invalid']:>4.0f}% | "
            f"{r['ms_A']:>7.2f} {r['ms_B']:>7.2f} {r['ms_D']:>7.2f} {r['ms_C']:>7.2f} "
            f"{r['ms_E']:>7.2f} {r['ms_F']:>7.2f} {r['ms_G']:>7.2f} | "
            f"{r['gflops_A']:>7.1f} {r['gflops_C']:>7.1f} "
            f"{r['gflops_F']:>7.1f} {r['gflops_G']:>7.1f} | "
            f"{r['speedup_DvA']:>5.2f} {r['speedup_CvD']:>5.2f} "
            f"{r['speedup_CvB']:>5.2f} {r['speedup_EvB']:>5.2f} "
            f"{r['speedup_FvB']:>5.2f} {r['speedup_GvB']:>5.2f} "
            f"{r['speedup_GvF']:>5.2f} {r['speedup_GvA']:>5.2f} | "
            f"{r['bucket_build_us']:>7.1f}"
        )
    print()
    print("D/A = skip-invalid gain   (same masked body, fewer tiles dispatched)")
    print("C/D = mask-free SIMD gain (same dispatch,    variant specialization)")
    print("C/B = task-mapping gain over native Triton pid-level if/else")
    print("E/B = launcher-region MVP over native Triton pid-level if/else")
    print("F/B = compiler-automated region dispatch (flag-based, same source as B)")
    print("G/B = compiler-automated region dispatch (@region_dispatch decorator)")
    print("G/F = decorator vs flag-based dispatch (should be ~1.0x)")
    print("G/A = combined gain for decorator-based region dispatch")


def print_tile_analysis(N_vals, B):
    print(f"\nTile analysis  (B={B})")
    print(f"{'N':>6} {'T':>4} {'total':>7} {'full':>7} {'diag':>6} {'invalid':>8} {'invalid%':>9}")
    print("-" * 55)
    for N in N_vals:
        s = tile_stats(N, B)
        pct = 100 * s["invalid"] / s["total"]
        print(f"{N:>6} {s['T']:>4} {s['total']:>7} {s['full']:>7} {s['diag']:>6} {s['invalid']:>8} {pct:>8.1f}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bench", action="store_true",
                        help="run correctness checks only, skip benchmark")
    parser.add_argument("--N", nargs="+", type=int, default=[512, 1024, 2048],
                        help="sequence lengths to sweep")
    parser.add_argument("--tile", type=int, default=32, help="tile size B")
    parser.add_argument("--head-dim", type=int, default=64, help="head dimension D")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--rounds", type=int, default=1,
                        help="benchmark rounds; if >2, drop min/max and average")
    args = parser.parse_args()

    B, D = args.tile, args.head_dim
    N_vals = args.N

    # ---- correctness ----
    print("=" * 60)
    print("Correctness checks")
    print("=" * 60)
    for N in N_vals:
        errs = _run_correctness(N, B, D, device="cpu")
        status = "OK" if all(v < 1e-2 for v in errs.values()) else "FAIL"
        print(f"  N={N:4d}, B={B}, D={D}  [{status}]  "
              f"err: A={errs['err_A']:.2e}  B={errs['err_B']:.2e}  "
              f"C={errs['err_C']:.2e}  D={errs['err_D']:.2e}  "
              f"E={errs['err_E']:.2e}  F={errs['err_F']:.2e}  G={errs['err_G']:.2e}")

    if args.no_bench:
        sys.exit(0)

    # ---- tile breakdown ----
    print_tile_analysis(N_vals, B)

    # ---- main benchmark ----
    print(f"\nBenchmark  (B={B}, D={D}, rep={args.rep}, rounds={args.rounds})")
    rows = run_benchmark(N_vals, B=B, D=D, rep=args.rep, rounds=args.rounds)
    print_benchmark_table(rows)

    # ---- isolated mask overhead ----
    mask_rows = run_mask_overhead_bench(N_vals, B=B, D=D, rep=args.rep)
    print_mask_overhead_table(mask_rows)
    print()
