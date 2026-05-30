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

3-way comparison at OMP=8 (interleaved 300 rounds, median):

| Workload                        | manual `if/else` | **4-variant quad** |
|---------------------------------|------------------|--------------------|
| 512×512×512  (aligned)          | 0.97× ↓          | **1.02×**          |
| 1024×1024×1024 (aligned)        | 0.97× ↓          | **1.03×**          |
| 1000×1000×1000 (tail)           | 0.96× ↓          | **1.02×**          |
| 2048×2048×2048 (aligned)        | 1.05×            | **1.11×**          |
| 2000×2000×2000 (tail)           | 1.02×            | **1.10×**          |

Baseline = always-masked kernel (the same kernel run on every tile).

Quad **strictly dominates** the obvious manual `if/else` rewrite at every
size — the manual rewrite *regresses* 3-4 % on small/mid sizes because
both branches share a single kernel (I-cache pressure plus `scf.if`
region barriers blocking cross-branch LLVM optimization).  Compile-time
variant separation in quad dispatch avoids both costs.

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
