# CPU Triton "Graph" — Proof-of-Concept

> Inspired by NVIDIA CUDA Graphs (which gave 2× ... 12.7× on GPU
> chained workloads — see `GPU_EXPERIMENTS.md` on `gpu-ptx-experiments`).
> This is the CPU analogue: skip Triton's Python launcher wrapper for
> chained kernel invocations by running the chain in a C trampoline.
>
> Status: **PoC**. Demonstrates 15× ... 190× speedup is reachable in the
> ideal case (single-thread, grid=(1,), tiny kernel, same args
> repeated). Production version would need OMP support, per-call args,
> automatic capture API — see "Scope limits" at the end.
>
> Branch: `cpu-launcher-graph`.

## 1. What problem this attacks

`kernel[grid](args)` in Triton-CPU goes through:

```
   Python: __getitem__(grid) → runner closure
       │
       ├─ resolve constexpr values
       ├─ lookup specialization for current dtype/divisibility
       ├─ serialize signature, pack args into PyBind argv
       │
       ▼
   C launcher (mod.launch): PyArg_ParseTuple, extract pointers,
                            extract num_threads, fire hooks,
                            OMP parallel for over grid,
                            invoke kernel function pointer
       │
       ▼
   Kernel (JIT'd C function): the actual SIMD work
```

For a tiny kernel where the kernel work is small (~µs), the per-call
**Python wrapper cost is the dominant component of wall-clock time**.

Probed by `_bench_cpu_graph_overhead.py`:

```
eager  kernel[grid](args)  : 7.38 us/call
direct compiled.run(...)   : 0.62 us/call    <- bypass __getitem__/runner
Python wrapper overhead    : 6.76 us (92% of total)
```

92% of every call is Python wrapper. For chained workloads (many
back-to-back small kernels), this overhead pays per call. Amortizing
it across the whole chain — exactly what CUDA Graphs does on GPU — is
the attack.

## 2. The PoC

`_bench_cpu_graph_poc.py`.

1. **Compile** the kernel via Triton normally
2. **Extract** the raw kernel function pointer (`compiled.function` —
   an integer holding the C function address inside the JIT-compiled
   .so)
3. **Generate** a tiny C trampoline:
   ```c
   typedef void (*kernel_fn_t)(float*, float*, float*, int32_t,
                               int32_t, int32_t, int32_t,
                               int32_t, int32_t, int32_t);
   void replay_chain(void* fn_ptr, float* x, float* y, float* out,
                     int32_t N, int n_chain) {
       kernel_fn_t fn = (kernel_fn_t)fn_ptr;
       for (int i = 0; i < n_chain; i++) {
           fn(x, y, out, N, /*pid=*/0, 0, 0, /*grid=*/1, 1, 1);
       }
   }
   ```
4. **Compile** the trampoline with `gcc -O2 -shared -fPIC`
5. **Load** via `ctypes`
6. **Replay**: one Python call into `replay_chain` runs the whole
   chain in C; per-iteration Python wrapper is zero

The kernel function pointer's C signature is what Triton's own
launcher uses internally (see `make_launcher` in
`python/triton/backends/cpu/driver.py`, the call
`(*kernel_ptr)(args..., pid_x, pid_y, pid_z, gridX, gridY, gridZ)`).
We're not bypassing the kernel — we're bypassing only the Python
wrapper, exactly as CUDA Graph replay does on GPU.

## 3. Numbers

PoC bench, Triton 3.7.0 CPU backend, tiny vec_add of 1024 elements,
grid=(1,) (single-thread). Same kernel called N times with same args:

```
 chain  eager (us)  graph (us)   speedup
     1        7.80        0.52    14.95x
     4       29.76        0.66    45.14x
    16      113.97        1.12   101.76x
    64      454.90        2.93   155.18x
   256     1827.72       10.10   180.88x
  1024     7326.65       38.63   189.67x
```

**Correctness**: bit-exact vs eager output (`max abs err = 0.0`) — we
literally call the same kernel function pointer, just skipping the
Python wrapper.

### Reading the table

- `chain=1` already gives 15× because graph has ONE Python call
  (~0.5µs) vs eager's full wrapper (~7.5µs)
- Growth from 15× → 190× is asymptotic to the ratio of (eager µs /
  per-C-call µs). At chain=1024, per-C-call cost is ~38ns (the
  tightest gcc-optimized for loop)
- The 190× ceiling is the **wall** the CPU launch path could approach
  if Triton itself emitted such a trampoline; it would not be a more
  exotic optimization than that.

## 4. Why this works so well on CPU vs the GPU number (12×)

| | GPU CUDA Graphs | CPU "graph" PoC |
|---|---|---|
| Python wrapper cost per call | ~17 µs (Triton 3.2 GPU path) | ~7 µs (Triton 3.7 CPU path) |
| Kernel work for "tiny" kernel | ~2 µs (GPU is huge, kernel finishes fast) | ~0.5 µs (1024-elem vec_add on 1 CPU thread) |
| Ceiling = wrapper / kernel | ~17/2 ≈ 8.5× | ~7/0.04 ≈ 190× (per-iter in C is ~40ns) |
| Observed (chain=256, max in test) | 12.7× | 180× |

The CPU ceiling is higher because:
1. CPU's "kernel work" for the tiny benchmark is much smaller than
   GPU's per-kernel overhead floor
2. The C-to-C call has lower per-call cost than GPU's command-buffer
   dispatch

For **realistic kernels** (those doing 50–500µs of actual work), the
wrapper share drops dramatically and the achievable speedup shrinks
toward the 1.05–1.2× range — same shape as GPU's curve, just shifted.

## 5. Scope limits of this PoC

What this PoC deliberately **doesn't** handle, that a production
implementation would need:

1. **OMP-parallel kernels (grid > 1, multi-thread)**: first attempt
   segfaulted, likely due to thread-init / TLS interaction between
   the trampoline's OMP context and Triton's kernel. Needs the
   trampoline to inline the same OMP loop logic the existing
   launcher uses (see `run_omp_kernels` in
   `python/triton/backends/cpu/driver.py:272`).
2. **Different kernels in the chain** (not just same kernel N times):
   trampoline must take an array of (fn_ptr, args_blob) and
   dispatch in order.
3. **Different args per call** (e.g. ping-pong output buffers):
   trampoline must accept arg arrays per call position, or buffer
   addresses must be threaded through.
4. **Capture API**: a clean `with cpu_graph.capture(): ...` context
   that records launches automatically (vs. the manual setup in the
   PoC script). Mirror of `torch.cuda.graph()`.
5. **Constexpr handling**: the PoC's kernel takes BLOCK as constexpr,
   which is baked in at compile time. Different constexpr values
   would need different compiled trampolines.
6. **Tail-guard interaction**: PoC sets `TRITON_CPU_TAIL_GUARD=0`
   because tail-guard wraps the compiled object in
   `DualPathCompiledKernel` / `QuadPathCompiledKernel`, which don't
   expose `compiled.function` directly. Production would need to
   either disable tail-guard for graphed kernels or extract the
   main/m_tail/n_tail/corner function pointers and call the right
   one per tile (mirroring `launch_quad`'s dispatch in C).

## 6. Place in the portfolio

Three CPU optimizations on this fork attack three orthogonal layers of
"how does Triton-CPU spend time it doesn't need to":

| Layer | Branch | What it skips |
|---|---|---|
| Cross-kernel (spatial) — per-tile dispatch overhead | `cpu-tail-guard` | M/N mask compute on interior tiles |
| Intra-kernel (temporal) — per-iteration mask cost | `cpu-loop-peel` | Mask cmpi + maskedload on aligned loop iterations |
| **Cross-launch (temporal) — per-launch Python overhead** | **`cpu-launcher-graph` (PoC)** | **Python wrapper for chained kernel sequences** |

Each addresses a different scope, and the wins compose (a tail-guard
binary can be loop-peeled internally, and any of them can be replayed
via the graph trampoline).

## 7. Reproducibility

```bash
# Overhead breakdown
conda run -n triton-cpu python python/test/unit/cpu/_bench_cpu_graph_overhead.py

# PoC bench
conda run -n triton-cpu python python/test/unit/cpu/_bench_cpu_graph_poc.py
```

Both correct against eager output (verified in the script before
benching).
