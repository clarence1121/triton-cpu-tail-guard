# triton-cpu — CPU dispatch optimizations

> Fork of [triton-lang/triton-cpu](https://github.com/triton-lang/triton-cpu)
> with two CPU-backend optimizations added on top of upstream.

## What's in this fork

**2-D Tail Guard** — automatic 4-variant compilation of matmul-style kernels
(`main` / `m_tail` / `n_tail` / `corner`) with a custom C dispatcher.  The
compiler walks the kernel AST, drops boundary `< M` / `< N` mask
comparisons for interior tiles, and keeps the `< K` K-loop mask intact —
the K mask is required to prevent LLVM from scalarizing the vector load
(verified down to assembly: removing it produces 1024 `vbroadcastss`
memory broadcasts instead of 127 register-to-register broadcasts).

**PID Region Dispatch** — automatic per-region kernel specialization for
kernels with heterogeneous pid classes.  The motivating case is causal
attention, where `(pid_q, pid_k)` tiles split into full / diagonal /
invalid regions that benefit from independently optimized variants.

Both features are **automatic** (no kernel changes required) and
controlled by env vars (`TRITON_CPU_TAIL_GUARD`, `TRITON_CPU_PID_REGION`).

## Headline result

3-way comparison at `OMP_NUM_THREADS=8`, BLOCK 64×64×32, interleaved 300
rounds, median time per kernel launch:

| Workload                  | baseline (µs) | manual `if/else` (µs)     | **4-variant quad (µs)** |
|---------------------------|---------------|---------------------------|-------------------------|
| 300×300×300   (tail)      |    289        |    300  (0.97× ↓)         |   **283  (1.02×)**      |
| 512×512×512   (aligned)   |   1 139       |  1 178  (0.97× ↓)         | **1 115  (1.02×)**      |
| 500×500×500   (tail)      |   1 154       |  1 197  (0.96× ↓)         | **1 132  (1.02×)**      |
| 1024×1024×1024 (aligned)  |   9 354       |  9 686  (0.97× ↓)         | **9 112  (1.03×)**      |
| 1000×1000×1000 (tail)     |   9 169       |  9 547  (0.96× ↓)         | **9 000  (1.02×)**      |
| 1000×800×900  (asymm.)    |   6 669       |  6 947  (0.96× ↓)         | **6 565  (1.02×)**      |
| 2048×2048×2048 (aligned)  |  75 029       | 71 254  (1.05×)           | **67 489 (1.11×)**      |
| 2000×2000×2000 (tail)     |  73 765       | 72 258  (1.02×)           | **67 366 (1.10×)**      |

Three implementations measured:

- **baseline** — the unmodified Triton kernel: every load/store pays the
  `(offs_m < M) & (offs_k + k < K) & (offs_n < N)` mask cost on every tile.
- **manual `if/else`** — the kernel itself runtime-branches on
  `(pid_m + 1) * BLOCK_M <= M && (pid_n + 1) * BLOCK_N <= N`, picking a
  no-M/N-mask path for full tiles and the masked path otherwise.  This is
  the obvious thing a programmer would write by hand.
- **4-variant quad** — this fork's automatic compile-time specialization:
  four separately compiled kernels (`main` / `m_tail` / `n_tail` / `corner`)
  dispatched per-tile by `launch_quad` in C.

Quad **strictly dominates** the manual `if/else` rewrite at every size —
the manual rewrite *regresses* 3-4 % on small/mid sizes because both
branches share a single kernel (I-cache pressure plus `scf.if` region
barriers blocking cross-branch LLVM optimization).  Compile-time variant
separation avoids both costs.

## Full technical writeup

→ **[CPU_DISPATCH_README.md](CPU_DISPATCH_README.md)** ←

Covers architecture (compile-time + runtime), file map, K-mask evidence
chain (MLIR → LLVM IR → x86 assembly → InstCombine root cause), build
instructions, reproduction benchmarks, debugging tips, and known caveats.

## Files

Implementation:

- `python/triton/compiler/cpu_tail.py` — tail-guard AST analysis + rewriter
- `python/triton/compiler/cpu_pid_region.py` — pid-region dispatch framework
- `third_party/cpu/backend/driver.py` — `launch_quad` / `launch_dual` /
  `run_region_2d` C launchers
- `python/triton/compiler/compiler.py`, `python/triton/runtime/jit.py` —
  wiring

Tests and benchmarks (`python/test/unit/cpu/`):

- `test_tail_guard.py` — unit tests with IR-level assertions
- `_bench_tail_guard_kmask.py` — K-mask isolation bench (proves the K mask
  must stay)
- `_bench_tail_guard_3way.py` — baseline / manual-if / quad three-way bench
- `_bench_matmul_tail.py` — 2-D matmul end-to-end bench
- `test_pid_variants.py`, `demo_region_dispatch.py`,
  `_bench_causal_flash_attention.py` — pid-region tests and demo

## Upstream

This is a fork.  The original project README and build instructions live
at upstream: [triton-lang/triton-cpu](https://github.com/triton-lang/triton-cpu).
