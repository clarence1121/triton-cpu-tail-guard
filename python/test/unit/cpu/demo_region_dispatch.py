"""
demo_region_dispatch.py  –  Generalized @region_dispatch demo
==============================================================

Demonstrates that ANY kernel with simple pid-pairwise conditions
can use @region_dispatch to get automatic branch-free variant
compilation and C-level geometric tile dispatch.

Three examples with the same Q @ K^T attention score computation:

  Example 1  Causal (lower-tri + diagonal)
               pid_k <  pid_q → full tile, dense dot
               pid_k == pid_q → diagonal tile, upper-tri mask
               pid_k >  pid_q → skip (invalid)

  Example 2  Anti-causal (upper-tri + diagonal)
               pid_k >  pid_q → full tile, dense dot
               pid_k == pid_q → diagonal tile, lower-tri mask
               pid_k <  pid_q → skip (invalid)

  Example 3  Diagonal-only
               pid_k == pid_q → only the T diagonal tiles, no mask
               everything else → skip

Each example shows:
  naive    – all T² tiles dispatched, pid-branch guard in kernel body
  dispatch – @region_dispatch: only valid tiles dispatched, branch-free variants

Usage
-----
  python demo_region_dispatch.py               # correctness + benchmark
  python demo_region_dispatch.py --no-bench    # correctness only
  python demo_region_dispatch.py --N 512 1024 2048 --tile 32 --head-dim 64
"""

import argparse
import time

import torch
import triton
import triton.language as tl
import triton.runtime

from triton.compiler.cpu_pid_region import RegionPlan, RegionSpec, region_dispatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bench(fn, warmup: int = 5, rep: int = 50) -> float:
    """Return median wall-clock time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rep):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) * 1e-6)
    times.sort()
    return times[len(times) // 2]


def _tile_args(Q, K, S, B):
    return (Q, K, S,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            S.stride(0), S.stride(1))


# ---------------------------------------------------------------------------
# Example 1: Causal attention
#   Valid: lower-triangle (pid_k < pid_q) + diagonal (pid_k == pid_q)
#   Skip:  upper-triangle (pid_k > pid_q)
# ---------------------------------------------------------------------------

@triton.jit
def _causal_naive(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k > pid_q:                          # skip invalid upper-tri tiles
        return
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    Q = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q, tl.trans(K))
    if pid_k == pid_q:                         # diagonal: apply causal mask
        acc = tl.where(offs_k[None, :] <= offs_q[:, None], acc, -1e9)
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


_causal_plan = RegionPlan(
    pid_vars=["pid_q", "pid_k"],
    regions=[
        RegionSpec("full", "pid_k < pid_q",  "region_2d_lt"),  # lower-tri: no mask
        RegionSpec("diag", "pid_k == pid_q", "region_2d_eq"),  # diagonal: mask
    ],
)

@region_dispatch(_causal_plan)
@triton.jit
def _causal_dispatch(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
):
    """Identical body to _causal_naive — compiler synthesizes branch-free variants."""
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k > pid_q:
        return
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    Q = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q, tl.trans(K))
    if pid_k == pid_q:
        acc = tl.where(offs_k[None, :] <= offs_q[:, None], acc, -1e9)
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def ref_causal(Q, K):
    S = torch.matmul(Q.float(), K.float().T)
    mask = torch.tril(torch.ones(Q.shape[0], Q.shape[0], dtype=torch.bool, device=Q.device))
    return torch.where(mask, S, torch.full_like(S, -1e9))


# ---------------------------------------------------------------------------
# Example 2: Anti-causal attention
#   Valid: upper-triangle (pid_k > pid_q) + diagonal (pid_k == pid_q)
#   Skip:  lower-triangle (pid_k < pid_q)
# ---------------------------------------------------------------------------

@triton.jit
def _anticausal_naive(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k < pid_q:                          # skip invalid lower-tri tiles
        return
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    Q = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q, tl.trans(K))
    if pid_k == pid_q:                         # diagonal: apply anti-causal mask
        acc = tl.where(offs_k[None, :] >= offs_q[:, None], acc, -1e9)
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


_anticausal_plan = RegionPlan(
    pid_vars=["pid_q", "pid_k"],
    regions=[
        RegionSpec("upper", "pid_k > pid_q",  "region_2d_gt"),  # upper-tri: no mask
        RegionSpec("diag",  "pid_k == pid_q", "region_2d_eq"),  # diagonal: mask
    ],
)

@region_dispatch(_anticausal_plan)
@triton.jit
def _anticausal_dispatch(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
):
    """Identical body to _anticausal_naive — compiler synthesizes branch-free variants."""
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k < pid_q:
        return
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    Q = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q, tl.trans(K))
    if pid_k == pid_q:
        acc = tl.where(offs_k[None, :] >= offs_q[:, None], acc, -1e9)
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def ref_anticausal(Q, K):
    S = torch.matmul(Q.float(), K.float().T)
    mask = torch.triu(torch.ones(Q.shape[0], Q.shape[0], dtype=torch.bool, device=Q.device))
    return torch.where(mask, S, torch.full_like(S, -1e9))


# ---------------------------------------------------------------------------
# Example 3: Diagonal-only
#   Valid: only the T diagonal tiles (pid_k == pid_q)
#   Skip:  everything else (pid_k != pid_q)
#
#   For diagonal tiles there is no masking — every element within the tile
#   is valid.  The dispatch alone saves ~(T-1)/T of the work vs naive.
# ---------------------------------------------------------------------------

@triton.jit
def _diag_naive(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k != pid_q:                         # skip all off-diagonal tiles
        return
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    Q = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q, tl.trans(K))
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


_diag_plan = RegionPlan(
    pid_vars=["pid_q", "pid_k"],
    regions=[
        RegionSpec("diag", "pid_k == pid_q", "region_2d_eq"),
    ],
)

@region_dispatch(_diag_plan)
@triton.jit
def _diag_dispatch(
    Q_ptr, K_ptr, S_ptr,
    stride_qn, stride_qd, stride_kn, stride_kd, stride_sn, stride_sk,
    BQ: tl.constexpr, BK: tl.constexpr, D: tl.constexpr,
):
    """Identical body to _diag_naive — compiler removes the != guard entirely."""
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k != pid_q:
        return
    offs_q = pid_q * BQ + tl.arange(0, BQ)
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    Q = tl.load(Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
    K = tl.load(K_ptr + offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    acc = tl.dot(Q, tl.trans(K))
    tl.store(S_ptr + offs_q[:, None] * stride_sn + offs_k[None, :] * stride_sk, acc)


def ref_diag(Q, K, B):
    N = Q.shape[0]
    S = torch.full((N, N), 0.0, device=Q.device)
    T = N // B
    for t in range(T):
        lo, hi = t * B, (t + 1) * B
        S[lo:hi, lo:hi] = Q[lo:hi] @ K[lo:hi].T
    return S


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def _launch(kernel, Q, K, S, N, B, D):
    T = N // B
    kernel[(T, T)](*_tile_args(Q, K, S, B), BQ=B, BK=B, D=D)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def run_correctness(N: int, B: int, D: int):
    torch.manual_seed(42)
    triton.runtime.driver.set_active_to_cpu()
    Q = torch.randn(N, D)
    K = torch.randn(N, D)

    results = {}
    for name, naive, dispatch, ref_fn, init_val in [
        ("causal",     _causal_naive,     _causal_dispatch,     lambda: ref_causal(Q, K),     -1e9),
        ("anticausal", _anticausal_naive, _anticausal_dispatch, lambda: ref_anticausal(Q, K), -1e9),
        ("diag",       _diag_naive,       _diag_dispatch,       lambda: ref_diag(Q, K, B),     0.0),
    ]:
        ref = ref_fn()
        S_n = torch.full((N, N), init_val); _launch(naive,    Q, K, S_n, N, B, D)
        S_d = torch.full((N, N), init_val); _launch(dispatch, Q, K, S_d, N, B, D)
        err_naive    = (S_n - ref).abs().max().item()
        err_dispatch = (S_d - ref).abs().max().item()
        ok = err_naive < 1e-2 and err_dispatch < 1e-2 and (S_n - S_d).abs().max().item() < 1e-5
        results[name] = (ok, err_naive, err_dispatch)
    return results


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(N: int, B: int, D: int, warmup: int = 5, rep: int = 50):
    triton.runtime.driver.set_active_to_cpu()
    T = N // B
    Q = torch.randn(N, D)
    K = torch.randn(N, D)

    rows = []
    for name, naive, dispatch, valid_tiles, init_val in [
        ("causal",     _causal_naive,     _causal_dispatch,     T * (T + 1) // 2, -1e9),
        ("anticausal", _anticausal_naive, _anticausal_dispatch, T * (T + 1) // 2, -1e9),
        ("diag",       _diag_naive,       _diag_dispatch,       T,                 0.0),
    ]:
        S = torch.full((N, N), init_val)
        ms_n = _bench(lambda: _launch(naive,    Q, K, S, N, B, D), warmup=warmup, rep=rep)
        ms_d = _bench(lambda: _launch(dispatch, Q, K, S, N, B, D), warmup=warmup, rep=rep)
        flops = 2 * valid_tiles * B * B * D
        rows.append(dict(
            name=name, N=N, T=T,
            valid_tiles=valid_tiles, pct_valid=100 * valid_tiles / (T * T),
            ms_naive=ms_n, ms_dispatch=ms_d,
            gflops_naive=flops / (ms_n * 1e-3) * 1e-9,
            gflops_dispatch=flops / (ms_d * 1e-3) * 1e-9,
            speedup=ms_n / ms_d,
        ))
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_correctness(N_vals, B, D):
    print("=" * 60)
    print("Correctness checks")
    print("=" * 60)
    for N in N_vals:
        res = run_correctness(N, B, D)
        status = "OK" if all(r[0] for r in res.values()) else "FAIL"
        parts = "  ".join(f"{k}={r[1]:.2e}/{r[2]:.2e}" for k, r in res.items())
        print(f"  N={N:4d}  [{status}]  (naive/dispatch err)  {parts}")
    print()


def print_benchmark(N_vals, B, D, warmup, rep):
    print(f"Benchmark  (B={B}, D={D}, warmup={warmup}, rep={rep})")
    hdr = (f"  {'example':<12} {'N':>5} {'T':>4} {'valid%':>7} | "
           f"{'naive(ms)':>10} {'dispatch(ms)':>13} {'speedup':>8} | "
           f"{'GF/s naive':>11} {'GF/s disp':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    prev_name = None
    for N in N_vals:
        rows = run_benchmark(N, B, D, warmup=warmup, rep=rep)
        for r in rows:
            if r['name'] != prev_name and prev_name is not None:
                print()
            print(f"  {r['name']:<12} {r['N']:>5} {r['T']:>4} {r['pct_valid']:>6.0f}% | "
                  f"{r['ms_naive']:>10.3f} {r['ms_dispatch']:>13.3f} {r['speedup']:>8.2f}x | "
                  f"{r['gflops_naive']:>11.1f} {r['gflops_dispatch']:>10.1f}")
            prev_name = r['name']
    print()
    print("  speedup = naive_ms / dispatch_ms  (higher is better for dispatch)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--N",        nargs="+", type=int,  default=[512, 1024, 2048])
    parser.add_argument("--tile",     type=int,  default=32)
    parser.add_argument("--head-dim", type=int,  default=64)
    parser.add_argument("--warmup",   type=int,  default=5)
    parser.add_argument("--rep",      type=int,  default=50)
    args = parser.parse_args()

    B, D = args.tile, args.head_dim

    print_correctness(args.N, B, D)

    if not args.no_bench:
        print_benchmark(args.N, B, D, warmup=args.warmup, rep=args.rep)
