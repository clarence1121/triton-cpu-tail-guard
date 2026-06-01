# CPU Dispatch Optimizations — Branch `jopperm/tensor-desc`

This branch adds two CPU-backend optimizations on top of upstream triton-cpu:

| Feature | Default | Env var | Purpose |
|---|---|---|---|
| **2-D Tail Guard** | **ON** | `TRITON_CPU_TAIL_GUARD` | Strip M/N boundary masks from full tiles in matmul-style kernels |
| **PID Region Dispatch** | OFF | `TRITON_CPU_PID_REGION` | Auto-detect causal-attention pid patterns and only dispatch valid tiles |

Both are *automatic* (no kernel changes required) — the compiler analyzes the
Python AST and emits multiple kernel variants, then a custom C launcher picks
which variant to call for each tile.

Most of this document is about **Tail Guard** (the main effort and the only
one with end-to-end bench evidence in this branch).  PID Region Dispatch is
described briefly in §6.

---

## Table of contents

1. [Problem and headline result](#1-problem-and-headline-result)
2. [Key insight: K mask must stay](#2-key-insight-k-mask-must-stay)
3. [Architecture](#3-architecture)
4. [File map (everything that changed on this branch)](#4-file-map)
5. [Build + run instructions](#5-build--run-instructions)
6. [PID Region Dispatch (Feature 2)](#6-pid-region-dispatch-feature-2)
7. [Debugging / troubleshooting](#7-debugging--troubleshooting)
8. [Known caveats](#8-known-caveats)

---

## 1. Problem and headline result

### Problem

In a typical Triton matmul kernel, every tile pays the same mask cost even
when the tile is fully inside the matrix:

```python
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
    acc = tl.dot(a, b, acc)
```

Only the rightmost column-strip and bottom row-strip of tiles actually need
the `< M`, `< N` masks.  The interior tiles (the vast majority for large
matmuls) waste cycles on those comparisons.

### Solution (2-D Tail Guard)

The compiler analyzes the kernel AST.  If it matches the matmul mask
pattern, it produces **four kernel variants** by AST rewriting:

| Variant | M-mask | N-mask | K-mask | Runs on |
|---|---|---|---|---|
| `main`   | ❌ removed | ❌ removed | ✅ kept | interior tiles (`pid_m < ⌊M/BLOCK_M⌋ ∧ pid_n < ⌊N/BLOCK_N⌋`) |
| `m_tail` | ✅ kept    | ❌ removed | ✅ kept | bottom row strip   (`pid_m == ⌊M/BLOCK_M⌋`) |
| `n_tail` | ❌ removed | ✅ kept    | ✅ kept | right column strip (`pid_n == ⌊N/BLOCK_N⌋`) |
| `corner` | ✅ kept    | ✅ kept    | ✅ kept | the single bottom-right corner tile |

A custom C launcher (`launch_quad` in `third_party/cpu/backend/driver.py`)
picks the right variant per `(pid_m, pid_n)` at run time.

### Headline result (this machine — `Linux 6.8.0-90`, OMP=8)

Median across 300 interleaved rounds, BLOCK 64×64×32:

| Workload | baseline (always masked) | manual `if/else` in kernel | **4-variant quad** |
|---|---|---|---|
| 300×300×300 (tail)          | 1.000× | 0.965× ↓ | **1.021×** |
| 512×512×512 (aligned)       | 1.000× | 0.967× ↓ | **1.022×** |
| 500×500×500 (tail)          | 1.000× | 0.964× ↓ | **1.019×** |
| 1024×1024×1024 (aligned)    | 1.000× | 0.966× ↓ | **1.027×** |
| 1000×1000×1000 (tail)       | 1.000× | 0.960× ↓ | **1.019×** |
| 1000×800×900 (asymm. tail)  | 1.000× | 0.960× ↓ | **1.016×** |
| 2048×2048×2048 (aligned)    | 1.000× | 1.053×   | **1.112×** |
| 2000×2000×2000 (tail)       | 1.000× | 1.021×   | **1.095×** |

Headlines:

- **+1.6 % to +11.2 %** speedup over always-masked baseline, **no regression at any size**.
- Quad **strictly dominates** the obvious manual `if/else` rewrite — the
  manual rewrite regresses 3-4 % on small/mid sizes because both branches
  are compiled into one kernel (I-cache pressure + scf.if blocks
  cross-branch LLVM optimization).
- Speedup grows with problem size — at 2048×2048×2048 we get **~10 %**.

The bench script is `python/test/unit/cpu/_bench_tail_guard_3way.py`.

---

## 2. Key insight: K mask must stay

**This is the most surprising finding of the whole investigation.**  The K
mask (`offs_k[None, :] + k < K`) **cannot** be removed even when `K %
BLOCK_K == 0` — removing it causes a **37 % regression**.

### Evidence chain (top to bottom)

#### 2.1 Isolation bench (single thread, BLOCK 32×32×32)

`python/test/unit/cpu/_bench_tail_guard_kmask.py` runs 5 variants where
M/N are always divisible (so M/N masks are no-ops) and isolates two effects:

- **H1**: removing K mask → does LLVM stop vectorizing?
- **H2**: changing loop range `range(0, K, BK)` → `range(0, _k_full, BK)` → does that hurt?

Result at 512×512×512, single thread, 300 iters:

| Variant | M/N mask | K mask | Range | Time | vs `no_mn` |
|---|---|---|---|---|---|
| `orig`       | ✅ | ✅ | `K` | 897 µs | 1.28× |
| `no_mn`      | ❌ | ✅ | `K` | 699 µs | 1.00× (baseline) |
| `no_k`       | ❌ | ❌ | `K` | 1040 µs | **1.49×** ← **H1 confirmed** |
| `peeled_km`  | ❌ | ✅ | `_k_full` | 813 µs | 1.16× ← H2 marginal |
| `peeled_all` | ❌ | ❌ | `_k_full` | 1080 µs | 1.55× |

**H1 (removing K mask) costs +49 %.**  H2 (changing loop range) is only +16 %
and disappears when the K mask is kept.

#### 2.2 MLIR (`tttcir` after lowering)

- `no_mn` uses `vector.maskedload`
- `no_k`  uses `vector.load`

Both have identical surrounding code.  Only the load op differs.

#### 2.3 LLVM IR

- `no_mn`: `@llvm.masked.load.v32f32.p0` with a runtime mask
  `%90 = icmp slt <32 x i32> %89, %19` (where `%19` is the `K` function
  argument — *runtime* value).
- `no_k`:  plain `load <32 x float>` — and then *each element is extracted
  individually*.

#### 2.4 Assembly (Intel x86_64, AVX-512 host)

| Metric | `no_mn` | `no_k` |
|---|---|---|
| `vbroadcastss` total | 135 | **1024** |
| ... from register (free) | 127 | 1 |
| ... from memory | 8 | **1023** |

`no_k` issues a scalar memory broadcast for *every* `A[m, k]` element.

#### 2.5 Root cause

LLVM's InstCombine applies the fold

```
extractelement(load ptr, i) → load(ptr + i*4)
```

For `@llvm.masked.load` with a runtime mask this fold is *illegal* — LLVM
cannot prove `mask[i] == true`, so the original vector load survives, the A
row stays in a vector register, and downstream FMAs broadcast from register
to register (one µop each, no memory traffic).

For unconditional `load <32 x float>`, the fold fires whenever LLVM thinks
extract-elements outweigh the vector load — which is exactly what happens
in the outer-product FMA pattern emitted by `ConvertDotToFMA.cpp` (it
emits 32 `vector.to_elements` per A row).

### 2.6 What does NOT work (don't try these)

- **Compile-time `dense<true>` mask.**  LLVM InstCombine simplifies
  `masked.load(all_true)` back to a plain `load`, which then scalarizes
  the same way.  Verified.
- **Manual `if/else` in the kernel** (see §1 results).  Both branches live
  in one kernel — I-cache pressure and scf.if region barriers eat the win.

The only fix is *separately compiled* variants where the main variant has
a real runtime K mask preserved in its IR.  This is what 2-D Tail Guard does.

---

## 3. Architecture

> Want the line-by-line walkthrough? See [`TAIL_GUARD_INTERNALS.md`](TAIL_GUARD_INTERNALS.md)
> for the full implementation deep-dive: AST recognition, mask
> simplifier, JIT integration, C launcher dispatch, OpenMP scheduling,
> and timing methodology — every step with `file:line` citations and
> concrete code excerpts. This section is the executive summary.

### 3.1 Compile-time path

```
@triton.jit kernel source
         ↓
JITFunction.run()  (python/triton/runtime/jit.py)
         ↓
analyze_cpu_tail_2d(fn, src)   (python/triton/compiler/cpu_tail.py)
   - walks AST, looks for:
       pid_m = tl.program_id(0)
       pid_n = tl.program_id(1)
       offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
       offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
       any compare:  offs_m[...] < <some_name>  →  records M-bound
       any compare:  offs_n[...] < <some_name>  →  records N-bound
       any for-loop:  for k in range(0, K, BLOCK_K)   (constexpr step)
   - returns matched=True iff all four are found
         ↓
make_cpu_tail_2d_variant("main"|"m_tail"|"n_tail"|"corner")
   - clones the AST
   - walks every tl.load / tl.store
   - simplifies its mask=... keyword:
       AND-tree of Compare nodes
       drop Compare if it refs offs_m and keep_m=False
       drop Compare if it refs offs_n and keep_n=False
       KEEP everything else (this is what preserves the K mask)
   - if all Compares dropped, remove the mask= and other= kwargs entirely
         ↓
self.compile(src, ...) × 4 with cpu_tail_2d_variant=v
         ↓
QuadPathCompiledKernel  (python/triton/compiler/compiler.py)
   - holds main/m_tail/n_tail/corner kernels
   - .run() calls launch_quad(...) with all four function pointers + M, N, BM, BN argument indices
```

### 3.2 Runtime path

`launch_quad` in `third_party/cpu/backend/driver.py` (~120 lines of C
generated from a Python template):

```c
// 1. Read M, N, BLOCK_M, BLOCK_N from kernel args by index.
// 2. Compute full_m = M / BLOCK_M, full_n = N / BLOCK_N.
// 3. If grid shape matches exp = (ceil(M/BM), ceil(N/BN), 1) → use quad dispatch.
//    Otherwise fall back to single-kernel run_omp_kernels(corner_kernel).
// 4. For each (x, y) in the grid:
//      bool xt = (x == full_m), yt = (y == full_n);
//      kptr = (!xt && !yt) ? main
//           : ( xt && !yt) ? m_tail
//           : (!xt &&  yt) ? n_tail
//           :                corner;
//      (*kptr)(args..., x, y, 0, gridX, gridY, 1);
// 5. Parallelized with #pragma omp parallel for schedule(static).
```

The dispatch is **branchless per tile** in the parallel loop body (a chain
of conditional moves, not branches), so the cost is on the order of one
extra cmov per tile — negligible compared to BLOCK_M × BLOCK_N × BLOCK_K FMAs.

The fallback to single-kernel mode triggers when the user passes an
unexpected grid shape (e.g. tiled by hand), in which case we just run the
masked corner kernel everywhere — never wrong, just no speedup.

---

## 4. File map

### 4.1 Tail-guard implementation files (new)

| File | Lines | Role |
|---|---|---|
| `python/triton/compiler/cpu_tail.py` | 538 | AST analysis + variant generation (1-D and 2-D). |

The 1-D path (`analyze_cpu_tail` / `make_cpu_tail_main_variant`) handles
elementwise kernels (`mask = offs < n`); it produces a `main` variant and
uses `DualPathCompiledKernel` + `launch_dual` for dispatch.  The 2-D path
(`analyze_cpu_tail_2d` / `make_cpu_tail_2d_variant`) handles matmul-style
kernels and is what the benchmarks measure.  Both paths share helpers.

The critical function is `_simplify_mask_expr` (lines 347-370): it walks
an AND-tree, drops `Compare` nodes that reference `offs_m`/`offs_n` per the
`keep_m`/`keep_n` flags, **and leaves everything else (including K-mask
Compares) untouched** — see the inline docstring at line 499.

### 4.2 Tail-guard wiring (modified files)

| File | Hunks | What changed |
|---|---|---|
| `python/triton/runtime/jit.py` | +85 lines | In `JITFunction.run()`: call `analyze_cpu_tail_2d` → if matched, compile 4 variants and build `QuadPathCompiledKernel`; otherwise try 1-D analysis → `DualPathCompiledKernel`. Also adds option plumbing for `enable_tail_guard` and `enable_pid_region_dispatch`. |
| `python/triton/compiler/compiler.py` | +225 lines | Adds `cpu_tail_variant` and `cpu_tail_2d_variant` parameters to `ASTSource`, calls into `cpu_tail.py` to rewrite the tree before compilation. Defines `DualPathCompiledKernel` (1-D) and `QuadPathCompiledKernel` (2-D) classes that hold multiple kernels and dispatch to `launch_dual` / `launch_quad`. Also adds `RegionCompiledKernel` for pid-region. |
| `python/triton/compiler/code_generator.py` | +3 lines | Plumbing for the new ASTSource fields. |
| `python/triton/compiler/__init__.py` | +11 lines | Public re-exports. |
| `third_party/cpu/backend/compiler.py` | +6 lines | Adds `enable_tail_guard: bool = True` to `CPUOptions` + env-var fallback `TRITON_CPU_TAIL_GUARD`. Also `enable_pid_region_dispatch`. |
| `third_party/cpu/backend/driver.py` | +618 lines | Adds the C launcher functions `launch_dual`, `launch_quad`, and `run_region_2d` (for pid-region). These are emitted into the generated `*.so` for each kernel. |

### 4.3 Tests and benches (new)

| File | Lines | Role |
|---|---|---|
| `python/test/unit/cpu/test_tail_guard.py` | 120 | Unit tests: correctness, AST analysis matches/falls back correctly, env var on/off, IR-level assertions (`vector.maskedload` in corner kernel vs `vector.load` in main kernel). Runs under `pytest`. |
| `python/test/unit/cpu/_bench_matmul_tail.py` | 158 | End-to-end matmul bench: correctness check across 7 sizes, then enabled-vs-disabled timing across 8 production sizes. **This is the original headline bench** but uses only 20 iters → noisy. |
| `python/test/unit/cpu/_bench_tail_guard_kmask.py` | 185 | **Single-thread K-mask isolation bench** (5 variants, BLOCK 32×32×32, 300 iters). Confirms K mask must stay; see §2.1. |
| `python/test/unit/cpu/_bench_tail_guard_3way.py` | 130 | **Three-way comparison** (baseline / manual `if/else` / 4-variant quad), interleaved + median-of-300, BLOCK 64×64×32. Produces the table in §1. |

### 4.4 PID-region feature (new, parallel feature on same branch)

| File | Lines | Role |
|---|---|---|
| `python/triton/compiler/cpu_pid_region.py` | 527 | Generalized 2-D pid-region dispatch framework. |
| `python/test/unit/cpu/test_pid_variants.py` | 865 | Causal-attention pid-variant tests (4 strategies: A/B/C/D). |
| `python/test/unit/cpu/demo_region_dispatch.py` | 387 | Demo of `@region_dispatch` decorator. |
| `python/test/unit/cpu/_bench_causal_flash_attention.py` | 188 | CPU causal FlashAttention microbench. |

See §6 for details.

### 4.5 Files removed during cleanup

| Removed file | Reason |
|---|---|
| `python/test/unit/cpu/_bench_tail_guard.py` (52 lines) | Early 1-D add POC bench. Superseded by `_bench_matmul_tail.py` (2-D) + unit tests in `test_tail_guard.py`. |
| `python/test/unit/cpu/_bench_tail_guard2.py` (101 lines) | 1-D hook-based overhead-decomposition bench. Useful when first measuring Python-dispatch overhead, but not relevant to the 2-D production case we care about now. |

### 4.6 Unrelated (left alone)

- `third_party/nvidia/backend/lib/gsan.ll` — Nvidia backend artifact, not
  related to CPU tail-guard or pid-region. Untracked but irrelevant.

---

## 5. Build + run instructions

### 5.1 Environment

These benches were developed and measured in conda env `triton-cpu`:

```bash
source /local/clarence/miniconda3/bin/activate triton-cpu
# Confirm: should print 2.6.0+cu124 3.6.0
python -c "import torch, triton; print(torch.__version__, triton.__version__)"
```

On the target machine, you need:
- Python 3.10+
- PyTorch (CPU is enough)
- A C++ compiler with OpenMP (`g++ -fopenmp`)
- LLVM build deps (the triton-cpu submodule has its own LLVM)

### 5.2 Build triton-cpu

Standard editable install from the repo root:

```bash
cd /path/to/triton-cpu
pip install -e python   # builds C++ extensions, takes ~10-20 min first time
```

If a previous build is cached and you only changed `.py` files, no rebuild
is needed.  If you change `third_party/cpu/backend/driver.py` (the C launcher
template), the C launcher is regenerated at kernel-compile time — no rebuild
of triton itself is needed.  If you change C++ pass code under
`third_party/cpu/lib/`, you do need to rebuild.

### 5.3 Run the unit tests

```bash
# All tail-guard unit tests (1-D, correctness + IR assertions)
pytest python/test/unit/cpu/test_tail_guard.py -v

# All causal pid-variant tests
pytest python/test/unit/cpu/test_pid_variants.py -v
```

Expected: all pass.

### 5.4 Run the three headline benches

```bash
# A. K-mask isolation (single thread, proves K mask must stay)
python python/test/unit/cpu/_bench_tail_guard_kmask.py

# B. End-to-end matmul (multi-thread, default 8 cores recommended)
OMP_NUM_THREADS=8 python python/test/unit/cpu/_bench_matmul_tail.py

# C. Three-way comparison (baseline / manual_if / 4-variant quad) — THE headline
OMP_NUM_THREADS=8 python python/test/unit/cpu/_bench_tail_guard_3way.py
```

Bench (C) is the strongest single piece of evidence — it directly proves
that 4-variant quad **strictly dominates** both naive masked-everywhere and
the obvious manual `if/else` rewrite.

### 5.5 Expected results on a stable machine

This machine is documented in §1 as noisy.  On the new (stable) machine:

- (A) should show **~1.4×-1.5× regression** for `no_k` and `peeled_all` vs
  `no_mn` baseline at 512×512×512.  If `no_k ≈ no_mn`, something is off
  (different LLVM version? different vec width?  — see §7).
- (B) is noisy unless you run with many iters.  Trends matter more than
  individual numbers.
- (C) should show **quad ≥ 1.01× on every size, ≥ 1.05× on xlarge**, and
  **manual_if regressing 3-5 % on small/mid sizes**.

If results differ qualitatively (e.g. manual_if no longer regresses, or
quad shows regression), see §7.

---

## 6. PID Region Dispatch (Feature 2)

This is a **separate feature on the same branch** for handling kernels with
heterogeneous pid classes — the motivating case is causal attention, where
`(pid_q, pid_k)` tiles split into:

- `pid_k < pid_q`  →  *full* tile, dense dot product, no mask
- `pid_k == pid_q` →  *diagonal* tile, lower-triangular mask
- `pid_k > pid_q`  →  *invalid*, do not dispatch

### 6.1 Two ways to use it

**(a) Decorator (explicit)**:

```python
from triton.compiler.cpu_pid_region import RegionPlan, RegionSpec, region_dispatch

causal_plan = RegionPlan(
    pid_vars=["pid_q", "pid_k"],
    regions=[
        RegionSpec("full", "pid_k < pid_q",  "region_2d_lt"),
        RegionSpec("diag", "pid_k == pid_q", "region_2d_eq"),
    ],
)

@region_dispatch(causal_plan)
@triton.jit
def attn_kernel(...):
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    if pid_k > pid_q:        # statically removed in both variants
        return
    if pid_k == pid_q:       # kept only in "diag" variant, removed in "full"
        ... masked branch ...
    else:                    # kept only in "full" variant
        ... no-mask branch ...
```

The constraint solver in `cpu_pid_region.py` proves under which region each
`if` is dead/live and rewrites the AST per variant.

**(b) Auto-detection (set env var)**:

```bash
TRITON_CPU_PID_REGION=1 python ...
```

`analyze_pid_region` scans the kernel; if it sees a top-of-function early
`if pid_k > pid_q: return` plus an inner `if pid_k == pid_q` / `else`, it
constructs the plan automatically.

### 6.2 C launcher

`run_region_2d(grid, kernel, op_code)` in `third_party/cpu/backend/driver.py`:
loops only over `(x, y)` pairs satisfying the geometric predicate (`x < y`,
`x == y`, etc.).

Op codes:

| Code | Predicate (`y` is pid_axis1, `x` is pid_axis0) |
|---|---|
| 0 | `y < x` (lower triangle) |
| 1 | `y <= x` |
| 2 | `y == x` (diagonal) |
| 3 | `y >= x` |
| 4 | `y > x` (upper triangle) |
| 5 | `y != x` |

### 6.3 Files

- Implementation: `python/triton/compiler/cpu_pid_region.py`
- Tests: `python/test/unit/cpu/test_pid_variants.py`
- Demo: `python/test/unit/cpu/demo_region_dispatch.py`
- Bench: `python/test/unit/cpu/_bench_causal_flash_attention.py`

### 6.4 Status

PID region dispatch is **OFF by default** (`TRITON_CPU_PID_REGION=0`).  It
shares wiring with tail-guard in `jit.py` / `compiler.py` but the two
features are mutually exclusive per-kernel — tail-guard analysis runs first
and pid-region only runs if tail-guard didn't match.

No headline benchmark numbers in this branch — the bench
(`_bench_causal_flash_attention.py`) is functional but not yet run with the
same rigour as the tail-guard 3-way bench.

---

## 7. Debugging / troubleshooting

### 7.1 Toggle features

```bash
TRITON_CPU_TAIL_GUARD=1   # default ON
TRITON_CPU_TAIL_GUARD=0   # disable; falls back to single corner kernel
TRITON_CPU_PID_REGION=1   # enable pid-region auto-detection
TRITON_CPU_TAIL_GUARD_DEBUG=1   # print analysis diagnostic + variant info
TRITON_CPU_PID_REGION_DEBUG=1
```

The tail-guard env var also accepts a per-call override:

```python
my_kernel[grid](..., enable_tail_guard=False)
```

### 7.2 Dump compiled IR / assembly

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton_dump python ...
```

Then look in `/tmp/triton_dump/<hash>/`:

- `<kernel>.ttir`   — Triton IR
- `<kernel>.ttcir`  — Triton CPU IR (look for `vector.load` vs `vector.maskedload`)
- `<kernel>.tttcir` — Triton-to-target CPU IR (same)
- `<kernel>.llir`   — LLVM IR (look for `@llvm.masked.load` and `extractelement`)
- `<kernel>.asm`    — x86 assembly (look for `vbroadcastss (mem)` vs `vbroadcastss %xmm,%ymm`)
- `<kernel>.so`     — final shared object

Each variant compiles to its own hash directory.  Match by kernel name
prefix (the variant suffix is encoded in the kernel name).

### 7.3 Verify the K-mask invariant on a new machine

If §2 results don't reproduce, the LLVM scalarization story may be different
on that CPU/LLVM combo.  Quick check:

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump \
    python python/test/unit/cpu/_bench_tail_guard_kmask.py
grep -c vbroadcastss /tmp/dump/*matmul_no_mn*.asm
grep -c vbroadcastss /tmp/dump/*matmul_no_k*.asm
```

Expected ratio: `no_k` should have **~7× more `vbroadcastss`** than `no_mn`,
and most of `no_k`'s should be memory-source (`(%rip)` or `(%rax)` etc.)
while `no_mn`'s should be mostly register-source (`%xmm` / `%ymm` / `%zmm`).

If the ratio is closer to 1×, LLVM is not scalarizing the unmasked load on
this target — in which case the K-mask preservation is unnecessary (but
also not harmful).

### 7.4 Catch a wrong-variant dispatch

`test_tail_guard.py` asserts at the IR level:

```python
assert "vector.load" in compiled.main_asm["tttcir"]
assert "vector.maskedload" not in compiled.main_asm["tttcir"]
assert "vector.maskedload" in compiled.asm["tttcir"]   # corner kernel
```

If `pytest python/test/unit/cpu/test_tail_guard.py` fails on these
assertions after a code change, the AST rewriter broke.

### 7.5 Verify quad dispatch path was taken

Add print statements or hook the launch to verify `launch_quad` ran rather
than the single-kernel fallback.  The fallback triggers when:

- the grid shape doesn't match `(ceil(M/BM), ceil(N/BN), 1)`
- M, N, BLOCK_M, BLOCK_N arg-index lookups fail
- any of those values is negative or zero

These are surfaced as `use_quad = false` at `driver.py:889` in the generated C.

---

## 8. Known caveats

1. **Noisy multi-threaded measurements.** On this machine, the same matmul
   bench can vary ±30 % between runs.  Use `_bench_tail_guard_3way.py`
   (300 interleaved rounds, median) — single-shot bench scripts are not
   trustworthy.

2. **Tail-guard requires AST pattern match.** If your kernel uses an
   unusual mask form (e.g. `offsets + 1 < n`, or BLOCK passed as runtime
   arg instead of `constexpr`), analysis falls back gracefully — the
   feature simply doesn't apply.  Check
   `compiled.metadata.cpu_tail_guard` for the matched/reason diagnostic.

3. **K-mask preservation is LLVM-specific.**  See §7.3.  On a different CPU
   target this might not be needed — but it doesn't hurt to keep, so the
   code keeps it unconditionally.

4. **Pid-region feature has no end-to-end speedup numbers in this branch
   yet.**  Implementation is functional but not benchmarked at the same
   level of rigour as tail-guard.

5. **`_bench_matmul_tail.py` uses only 20 iters.** It is convenient for a
   quick smoke test but unreliable for measurement — use
   `_bench_tail_guard_3way.py` for numbers you trust.

6. **The cache key includes the env var values.** Changing
   `TRITON_CPU_TAIL_GUARD` between runs forces recompile (intentional, so
   you can A/B in one process).  Don't be surprised by long first-run
   latency when switching modes.
