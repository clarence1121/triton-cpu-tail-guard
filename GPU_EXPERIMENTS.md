# GPU Side-Project Experiments — Where the Wins Actually Are

> Five hardware-flavored optimization angles tried on Triton 3.2.0 +
> RTX 4090 (sm_89). **Four are negative results** (no clean win on this
> hardware), **one is a 2×–12× win** that lives one layer above the
> kernel author.
>
> The single positive finding: **CUDA Graphs** capture of a chained
> Triton kernel sequence skips per-launch driver overhead and gives
> 2× ... 12.7× speedup, growing with chain length. The other four
> attempts at the kernel-codegen layer (manual 128-bit loads,
> persistent kernels, dynamic atomic work-stealing, redux.sync warp
> reduction) showed that vanilla Triton's GPU codegen is already
> well-tuned for typical kernel shapes on Ada hardware.
>
> Branch: `gpu-ptx-experiments` (companion to `cpu-tail-guard` and
> `cpu-loop-peel`). Scripts: `python/test/unit/gpu/`.

## TL;DR

| Angle | Hypothesis | Reality | Outcome |
|---|---|---|---|
| **★ CUDA Graphs orchestration** | Capture a chain of Triton kernels into one driver-level graph; replay skips per-launch overhead | Per-launch overhead in Triton is ~17 µs of pure Python+driver work. For chained workloads, this dominates. CUDA Graphs amortizes it across the whole chain. | **2× ... 12.7× speedup** (grows with chain length) |
| **#6a** Force 128-bit `ld.global.v4.b32` via inline PTX when Triton uses 32-bit | Triton emits narrow loads at small BLOCK and leaves bandwidth on the table | Triton *does* use scalar loads at BLOCK ≤ 128 — but at those sizes, **launch overhead dominates**, not load width. Forcing 128-bit at small BLOCK can't move the wall. | **No clean win**; Triton's load codegen is well-tuned for typical kernel shapes. |
| **#4a** Persistent kernel (static stride) | Less launch overhead than grid-launch for many small tiles | For uniform workloads, **GPU hardware scheduler beats software persistent** at almost every workload size. Persistent costs 0.94×–1.13× of grid. | **Marginal**; only wins at very specific tile counts. |
| **#4b** Persistent + atomic counter for work-stealing | Should crush variable-length workloads (load imbalance) | **3–4× *slower* than grid** for variable-length reductions on this hardware. 128 SMs contend for a single global atomic counter, and the contention costs more than the imbalance it solves. | **Worse than baseline**. |
| **#6b** `redux.sync.add.f32` via inline PTX instead of Triton's 5-instruction `shfl.sync.bfly` tree | Single-instruction warp reduction is faster than the tree | **FP `redux.sync` requires sm_100+ (Blackwell)**. On Ada (sm_89, this 4090) and Hopper (sm_90), only integer redux exists. ptxas refuses `redux.sync.add.f32`. | **Hardware-blocked**, cannot test on this GPU. |

The CPU-side work on this fork (tail-guard, loop-peel) addresses
**CPU-specific codegen weaknesses** (LLVM scalarizing masked loads,
predicated loops missing vectorization). The analogous GPU codegen
weaknesses don't exist on this hardware/Triton combination — the
NVIDIA codegen path is mature and the predicate hardware on Ada is
essentially free.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (Ada Lovelace, sm_89, 128 SMs) |
| Triton | 3.2.0 (vanilla pip install, no modifications) |
| PyTorch | 2.6.0 + cu124 |
| CUDA | 12.4 |
| Timing | `triton.testing.do_bench(measure_time_with_hooks=False)` — kernel timing on GPU goes via CUDA events, which already exclude Python wrapper overhead |
| Statistical | `return_mode="min"` to filter cold/throttled iterations |

GPU bench uses CUDA events natively (`triton.testing.do_bench` on
device != "cpu"), so the Python-wrapper-dilution problem documented in
the CPU work (`TAIL_GUARD_INTERNALS.md` §8) doesn't apply here. Numbers
shown are kernel-only.

## Angle #6a — Manual 128-bit loads

### What was hypothesized

Triton's tile-to-thread mapping at small BLOCK degenerates into one
element per thread, which CUDA compiles to `ld.global.b32` (32-bit
scalar load) per thread. Forcing the codegen to issue
`ld.global.v4.b32` (128-bit vector) by hand-bundling 4 elements per
thread should saturate memory bandwidth even at small BLOCK.

### What Triton actually emits

`python/test/unit/gpu/_probe_load_ptx.py` dumps PTX for `vec_add`
across BLOCK sizes:

```
 BLOCK   # loads                          opcodes (unique)
    32         2                             ld.global.b32         ← scalar
    64         2                             ld.global.b32         ← scalar
   128         2                             ld.global.b32         ← scalar
   256         2                          ld.global.v2.b32         ← 64-bit, starts vectorizing
   512         2                          ld.global.v4.b32         ← 128-bit, full vector
  1024         4                          ld.global.v4.b32
  2048         8                          ld.global.v4.b32
  4096        16                          ld.global.v4.b32
```

**Crossover from scalar → 128-bit is BLOCK=512.** BLOCK=256 picks up
half-width vector loads. Below 256, scalar.

### Bandwidth measurements

`_bench_load_widths.py`, N = 1M (12 MB total) and N = 16M (201 MB total):

```
N = 16,777,216 (201 MB)
   BLOCK   time (us)     bandwidth
      32      275.46       730.9 GB/s   ← scalar load, but
      64      231.84       868.4 GB/s     bandwidth already approaches
     128      226.08       890.5 GB/s     ~900 GB/s (peak is ~1000)
     256      224.26       897.8 GB/s
     512      224.16       898.1 GB/s
    1024      225.28       893.7 GB/s
    2048      228.35       881.7 GB/s
```

The **bandwidth wall is at BLOCK = 64**, not at the scalar/vector
crossover (BLOCK = 256–512). Going from scalar (BLOCK = 64) to vector
(BLOCK = 512) only buys 30 GB/s (~3%) on the way to the ~900 GB/s
plateau. At BLOCK = 32 the cost is launch overhead (32× more programs
than at BLOCK = 1024), not load width.

### Why the obvious fix doesn't pay

A manual 128-bit load via the "wide layout" trick
(`BLOCK threads × 4 elements_per_thread`) was tried in
`_bench_load_widths.py` (early prototype, kept simple in final
version):

- `vec_add` with BLOCK=32 (1 warp, 1 elem/thread): 24.58 µs, 512 GB/s
- "Wide" `vec_add` with BLOCK=32 × ELEMS=4 (1 warp, 4 elems/thread): 17.41 µs, 723 GB/s

A 1.41× speedup — but the PTX shows **both kernels emit scalar
`ld.global.b32`**. The win comes from **fewer programs** (1/4 the
launches), not from wider loads. Equivalent to just using BLOCK=128.

`tl.inline_asm_elementwise` cannot fix this either: it's designed for
elementwise *compute*, not memory access. Loads are not "pure
elementwise" operations from the asm-helper's perspective.

The right tools for forcing wider loads (modifying Triton's layout
inference, or using `tl.max_contiguous` + `tl.multiple_of` hints) live
inside the compiler. From outside the compiler, on this hardware,
Triton's choice is fine.

## Angle #4 — Persistent kernel

### What was hypothesized

Replace `grid = (NUM_TILES,)` with `grid = (NUM_SMS,)` and have each
program loop over its share of tiles. Benefits:

1. Fewer kernel launches per call (NUM_SMS = 128 vs NUM_TILES, which
   can be much larger).
2. (For dynamic atomic-stealing) Better load balance under variable
   per-tile cost.

### Uniform vec_add — static persistent

`_bench_persistent.py` § 1, BLOCK=1024:

```
         N   tiles    grid us    pers us   speedup
     1,024       1       2.05       2.75     0.74x   ← 1 tile, grid wins
    16,384      16       3.07       2.72     1.13x   ← tiles < NUM_SMS, persistent wins slightly
   262,144     256       6.94       6.85     1.01x   ← tiles ≈ 2×NUM_SMS, tie
 1,048,576    1024      16.38      17.34     0.94x   ← persistent loses a hair
16,777,216   16384     225.28     230.40     0.98x   ← tie
67,108,864   65536     876.54     884.74     0.99x   ← tie
```

Persistent wins by 13% at exactly one workload size (N=16K, 16 tiles)
and loses or ties everywhere else. **The GPU hardware scheduler
distributes tiles across SMs very efficiently** — when grid is small
enough, you don't need software help; when grid is large, the
scheduler handles thousands of tiles without breaking a sweat.

### Variable-length — dynamic persistent (atomic stealing)

`_bench_persistent.py` § 2, BLOCK=128, random lengths in [1, 1024]:

```
 tiles   max_len    grid us   static us     dyn us   st/grid   dyn/grid
   128      1024       6.11        5.92       8.19     1.03x      0.75x
  1024      1024       9.22       23.55      25.44     0.39x      0.36x
  4096      1024      20.48       76.80      81.70     0.27x      0.25x
```

**Dynamic persistent (atomic counter) is 3–4× SLOWER than
grid-launch.** The atomic on a single global counter creates massive
SM-to-SM contention: every iteration, all 128 SMs serialize through
`tl.atomic_add(COUNTER, 1)`. Even when this avoids load imbalance, the
contention cost is much worse than the imbalance it would otherwise
incur.

For the 4-tile case (`128 tiles, max_len=1024`), grid even barely
edges out static persistent — because grid launching 128 programs is
exactly the right hardware mapping.

### When does persistent actually win on GPU?

Not in these tests. The real wins from persistent kernels on GPU
require shapes we couldn't test here:

| Scenario | Why persistent wins | Why we didn't test |
|---|---|---|
| Many tiny back-to-back kernels (e.g. RNN cell, LLM serving with batch=1) | Skip host-side launch latency between "logical kernels" by encoding them as iterations of one persistent kernel | Requires multi-kernel orchestration as part of the workload definition — outside scope |
| Producer-consumer with warp specialization | One warp issues `cp.async`, another consumes; persistent lets them coexist permanently | Triton 3.2.0 has no `tl.async_task` or `warp_specialize` API |
| Hopper CTA cluster + DSM | CTAs in a cluster directly read each other's shared memory | 4090 has no clusters (Hopper-only) |
| TMA-backed pipelined loads | Dedicated DMA engine for tiles | TMA is Hopper-only |

## Angle #6b — `redux.sync.add` warp reduction

### What was hypothesized

Triton's `tl.sum` over a warp lowers to a tree of 5
`shfl.sync.bfly.b32` instructions. NVIDIA introduced `redux.sync.<op>`
in sm_80 (Ampere) that does the same reduction in 1 instruction.
Inline PTX should beat the tree.

### What Triton actually emits

`_probe_redux_ptx.py` confirms:

```
PTX for tl.sum(x) on a warp-sized vector:
  shfl.sync.bfly.b32  %r5,  %r1,  16, 31, -1;
  shfl.sync.bfly.b32  %r7,  %r6,  8,  31, -1;
  shfl.sync.bfly.b32  %r9,  %r8,  4,  31, -1;
  shfl.sync.bfly.b32  %r11, %r10, 2,  31, -1;
  shfl.sync.bfly.b32  %r13, %r12, 1,  31, -1;
```

The tree IS there. Triton does NOT use `redux.sync` even though we're
on sm_89 — which would support the integer variant.

### What blocks the inline PTX fix

Tried:

```ptx
redux.sync.add.f32 dst, src, 0xffffffff;
```

`ptxas` rejects with:

```
ptxas /tmp/foo.ptx, line 45; error : Unexpected instruction types specified for 'redux'
ptxas fatal : Ptx assembly aborted due to errors
```

Looking at the NVIDIA PTX ISA:

| Compute capability | redux.sync support |
|---|---|
| sm_80 (Ampere) – sm_89 (Ada) | Integer only: `u32`, `s32`, `b32` |
| sm_90 (Hopper) | Integer only (still no FP redux for arbitrary ops) |
| **sm_100+ (Blackwell)** | **First adds FP redux** (`f16`, `bf16`, `f32`) |

So the inline PTX win for FP reductions is **Blackwell-only hardware**.
Ada and Hopper can't test it. The integer variants exist on Ada but
aren't useful for typical FP reduction workloads (sum, mean, max).

### Why Triton doesn't use redux for integer either

Triton's compiler currently emits the shfl tree uniformly for all
reductions, even integer ones where redux.sync is available. This
*is* a missed optimization for integer reductions on sm_80+ —
**that's a real bug for the upstream Triton codegen to fix**, not a
user-level workaround. The fix would be in the LLVM backend or
TritonGPU's reduction lowering pass, not in the kernel-author's
toolbox.

## ★ The win: CUDA Graphs

### Setup

Workload: a chain of `n_ops` tiny `tiny_kernel = vec += 1` kernels,
each operating on N elements. Each kernel is small (~µs of actual
GPU work). Per-launch overhead is the dominant cost.

Two execution modes:

1. **Eager**: each kernel launched separately via standard Triton
   call (`kernel[grid](args)`). Per-launch overhead pays each time.
2. **Graph**: capture the whole chain into a `torch.cuda.CUDAGraph`,
   then `g.replay()` runs the whole sequence as one driver-level
   dispatch. Per-launch overhead paid ONCE during capture, then ZERO
   per replay.

Script: `python/test/unit/gpu/_bench_cuda_graphs.py`.

### Numbers

BLOCK=1024, RTX 4090:

```
 chain len         N    eager us    graph us   speedup
        1       256       17.41        8.35     2.08x
        1      4096       16.61        8.19     2.03x
        1     65536       16.74        8.19     2.04x
        4       256       51.10        8.19     6.24x
        4      4096       50.56        8.19     6.17x
        4     65536       50.11        8.19     6.12x
       16       256      181.38       18.21     9.96x
       16      4096      179.10       18.43     9.72x
       16     65536      178.24       19.46     9.16x
       64       256      703.42       57.34    12.27x
       64      4096      696.64       60.42    11.53x
       64     65536      700.42       66.43    10.54x
      256       256     2757.79      216.77    12.72x
      256      4096     2797.57      228.35    12.25x
      256     65536     2791.42      250.50    11.14x
```

### Reading the table

| Pattern | Implication |
|---|---|
| Even at `n_ops = 1`, graphs give 2.0× | Per-launch overhead is ~8 µs (eager 17 µs − graph 8 µs); a single graph replay is ~8 µs (driver dispatch) |
| Speedup grows with `chain_len`: 1→2×, 4→6×, 16→10×, 64→12×, 256→12.7× | Per-launch overhead amortizes across the whole chain; ceiling is determined by how much the graph itself costs to dispatch |
| Speedup is invariant to N (within a column) | Per-launch overhead is independent of kernel work size; only how MANY launches matters |

For workloads with many chained kernels — LLM serving with batch=1
(thousands of decoder kernel launches per token), RNN cell rollouts,
ML pipelines with many small ops — **this is the single most important
GPU optimization to apply**. It's not a Triton kernel-level fix; it's
orchestration above the kernels.

### Why this is the only kernel-author-accessible GPU win we found

CUDA Graphs is exposed via PyTorch (`torch.cuda.CUDAGraph`) — no
Triton compiler work needed. You just wrap your launch sequence in
the capture context. The rest is NVIDIA driver + GPU hardware doing
the work.

The other four angles tried in this round attacked the *kernel*
layer (load width, persistent grid, warp reduction). All blocked
because vanilla Triton's kernel-layer codegen is already near-peak on
Ada. The remaining performance is at the **launch / orchestration
layer**, which is exactly what CUDA Graphs targets.

## What would actually win on this hardware?

Plausible angles not tried in this round, each requiring
significantly more infrastructure than the experiments above:

1. **L2 cache eviction control** via `cuStreamSetAttribute` /
   `cudaAccessPolicyWindow`. For streaming workloads, mark loads as
   `EVICT_FIRST` to avoid polluting L2 with one-time reads; for
   re-read workloads, pin data with `EVICT_LAST`. Triton has no
   awareness of these driver-level hints; user must call CUDA driver
   APIs directly around kernel launches.

2. **CUDA Graphs** to capture many small kernels into one launched
   unit, amortizing launch overhead across the whole graph. Triton
   kernels compose with `torch.cuda.CUDAGraph`; the optimization is
   in the orchestration layer, not the kernel.

3. **`cp.async` software pipelining for non-matmul kernels.** Triton's
   pipeline pass exists but is matmul-focused. Manually scheduling
   `cp.async.cg.shared.global` to overlap next-iteration loads with
   current-iteration compute could give 5–15% on reduction-heavy
   kernels. Requires either custom inline PTX or modifications to
   the Triton pipeline pass.

4. **Modify Triton's TritonGPU reduction lowering** to emit
   `redux.sync.add` for integer reductions on sm_80+. This is a clean
   one-line-ish patch to the TritonGPU codegen, but it's a Triton
   compiler change, not a user-side optimization.

5. **Hopper-specific work**: TMA, cluster + DSM, warp specialization
   via `wgmma`. All require sm_90+ hardware.

These are real angles for someone with the matching hardware and
willingness to either modify Triton or write CUDA C++ alongside their
Triton kernels.

## Conclusion

On RTX 4090 + Triton 3.2.0, vanilla Triton's GPU **kernel-level**
codegen is well-tuned for the standard kernel shapes we tried. The
optimizations that made sense on CPU (where LLVM's scalarization of
masked loads created big wins from mask removal) **do not have
kernel-level analogs on GPU** because:

- GPU mask is a hardware predicate, essentially free
- GPU load codegen already vectorizes at typical BLOCK sizes
- GPU hardware scheduler outperforms software persistent for typical
  workloads
- The remaining kernel-level attack surfaces (Blackwell FP redux,
  Hopper TMA, L2 hints) require hardware or infrastructure we don't
  have

**The win is one layer up.** CUDA Graphs orchestration gives 2× ...
12.7× on chained kernel workloads by amortizing launch overhead in
driver hardware. For any workload that does many small back-to-back
Triton kernel launches — exactly the shape of LLM inference, RNN
rollouts, ML pipelines — this is the single most important GPU
optimization to apply.

**Two takeaways**:

1. **Tail-guard and loop-peel on CPU are CPU-specific.** Their direct
   GPU translations buy nothing because the GPU codegen is mature.
2. **GPU performance work belongs at the orchestration layer**, not
   the kernel layer — at least until you reach Hopper-class hardware
   where TMA, clusters, and warp specialization open new kernel-level
   surfaces.

## Scripts

| File | What it does |
|---|---|
| `python/test/unit/gpu/_bench_cuda_graphs.py` | **★ Main win** — chained-launch eager vs CUDA Graphs (2×–12× speedup) |
| `python/test/unit/gpu/_bench_load_widths.py` | vec_add bandwidth across BLOCK sizes (saturates at BLOCK=64) |
| `python/test/unit/gpu/_probe_load_ptx.py` | dump load opcodes per BLOCK size (scalar→128-bit crossover at BLOCK=256) |
| `python/test/unit/gpu/_bench_persistent.py` | grid vs static persistent vs dynamic-atomic persistent |
| `python/test/unit/gpu/_probe_redux_ptx.py` | dump reduction PTX, check redux.sync FP availability |
