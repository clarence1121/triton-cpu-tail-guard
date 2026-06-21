"""
CPU causal FlashAttention microbenchmark.

Compares two single-kernel forward variants:
  baseline: every Q tile scans every K/V tile and applies causal mask each step
  split:    skips future K/V tiles, uses mask-free full tiles, masks only diagonal

This is intentionally small and CPU-focused.  It validates whether causal tile
classification remains profitable after softmax and V accumulation, not just QK
score materialization.
"""

import argparse
import time

import torch
import triton
import triton.language as tl


triton.runtime.driver.set_active_to_cpu()


@triton.jit
def _attn_baseline(Q, K, V, O, stride_n, stride_d, N_CTX: tl.constexpr, BLOCK: tl.constexpr,
                   HEAD_DIM: tl.constexpr, SM_SCALE: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d)
    m_i = tl.full((BLOCK, ), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK, ), 0.0, tl.float32)
    acc = tl.zeros((BLOCK, HEAD_DIM), tl.float32)

    for start_n in range(0, N_CTX, BLOCK):
        k = tl.load(K + (start_n + offs_n)[:, None] * stride_n + offs_d[None, :] * stride_d)
        qk = tl.dot(q, tl.trans(k)) * SM_SCALE
        causal = (start_n + offs_n[None, :]) <= offs_m[:, None]
        qk = tl.where(causal, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V + (start_n + offs_n)[:, None] * stride_n + offs_d[None, :] * stride_d)
        acc += tl.dot(p.to(tl.float32), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d, acc)


@triton.jit
def _attn_split(Q, K, V, O, stride_n, stride_d, N_CTX: tl.constexpr, BLOCK: tl.constexpr,
                HEAD_DIM: tl.constexpr, SM_SCALE: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    q_start = pid_m * BLOCK

    q = tl.load(Q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d)
    m_i = tl.full((BLOCK, ), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK, ), 0.0, tl.float32)
    acc = tl.zeros((BLOCK, HEAD_DIM), tl.float32)

    for start_n in range(0, N_CTX, BLOCK):
        if start_n <= q_start:
            k = tl.load(K + (start_n + offs_n)[:, None] * stride_n + offs_d[None, :] * stride_d)
            qk = tl.dot(q, tl.trans(k)) * SM_SCALE
            if start_n == q_start:
                causal = offs_n[None, :] <= tl.arange(0, BLOCK)[:, None]
                qk = tl.where(causal, qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v = tl.load(V + (start_n + offs_n)[:, None] * stride_n + offs_d[None, :] * stride_d)
            acc += tl.dot(p.to(tl.float32), v)
            m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d, acc)


def ref_attention(q, k, v):
    scale = q.shape[1] ** -0.5
    scores = q @ k.T * scale
    mask = torch.tril(torch.ones(scores.shape, dtype=torch.bool))
    probs = torch.softmax(torch.where(mask, scores, torch.full_like(scores, -float("inf"))), dim=-1)
    return probs @ v


def bench_one(fn, warmup=5, rep=30):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rep):
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) * 1e-6)
    times.sort()
    return times[len(times) // 2]


def trimmed_mean(vals):
    vals = sorted(vals)
    if len(vals) <= 2:
        return sum(vals) / len(vals)
    return sum(vals[1:-1]) / (len(vals) - 2)


def run_case(n, block, head_dim, rep, rounds):
    assert n % block == 0
    torch.manual_seed(0)
    q = torch.randn(n, head_dim, dtype=torch.float32)
    k = torch.randn(n, head_dim, dtype=torch.float32)
    v = torch.randn(n, head_dim, dtype=torch.float32)
    o_base = torch.empty_like(q)
    o_split = torch.empty_like(q)
    scale = head_dim ** -0.5
    grid = (n // block, )

    def baseline():
        _attn_baseline[grid](q, k, v, o_base, q.stride(0), q.stride(1), N_CTX=n, BLOCK=block,
                             HEAD_DIM=head_dim, SM_SCALE=scale)

    def split():
        _attn_split[grid](q, k, v, o_split, q.stride(0), q.stride(1), N_CTX=n, BLOCK=block,
                          HEAD_DIM=head_dim, SM_SCALE=scale)

    baseline()
    split()
    ref = ref_attention(q, k, v)
    base_err = torch.max(torch.abs(o_base - ref)).item()
    split_err = torch.max(torch.abs(o_split - ref)).item()

    base_times = []
    split_times = []
    for _ in range(rounds):
        base_times.append(bench_one(baseline, rep=rep))
        split_times.append(bench_one(split, rep=rep))

    base_ms = trimmed_mean(base_times)
    split_ms = trimmed_mean(split_times)
    tiles = n // block
    skipped = tiles * (tiles - 1) // 2
    total = tiles * tiles
    return {
        "N": n,
        "T": tiles,
        "skip_pct": 100.0 * skipped / total,
        "base_ms": base_ms,
        "split_ms": split_ms,
        "speedup": base_ms / split_ms,
        "base_err": base_err,
        "split_err": split_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    print(f"rounds={args.rounds}, trim=min/max, BLOCK={args.block}, D={args.head_dim}")
    print(f"{'N':>6} {'T':>4} {'skip%':>6} | {'baseline':>9} {'split':>9} {'speedup':>7} | {'err base':>9} {'err split':>9}")
    print("-" * 82)
    for n in args.N:
        r = run_case(n, args.block, args.head_dim, args.rep, args.rounds)
        print(f"{r['N']:>6} {r['T']:>4} {r['skip_pct']:>5.1f}% | "
              f"{r['base_ms']:>9.3f} {r['split_ms']:>9.3f} {r['speedup']:>7.2f} | "
              f"{r['base_err']:>9.2e} {r['split_err']:>9.2e}")


if __name__ == "__main__":
    main()
