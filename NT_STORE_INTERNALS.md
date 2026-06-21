# Non-Temporal Store Pass — Internals

> A late-stage LLVM-dialect pass that marks write-only store ops with
> `nontemporal=true` (and bumps their alignment so the LLVM backend can
> actually issue 256-bit `vmovntps` instead of falling back to scalar
> `movntiq`). Frees L1/L2 bandwidth for the data the kernel re-reads.
>
> Off by default; enable with `TRITON_CPU_NT_STORE=1`.
>
> Branch: `cpu-nt-store`.

## TL;DR

A small MLIR pass (~150 lines) that:

1. Walks every `LLVM::LLVMFuncOp` in the module.
2. Identifies pointer arguments the kernel never reads from (write-only
   buffers — typically the output tensor of an elementwise / pointwise op).
3. For every `llvm.store` whose address traces back to one of those
   arguments, sets `nontemporal=true` and bumps `alignment` to 32 bytes.

The LLVM x86 backend then emits `vmovntps` (256-bit AVX2 non-temporal
store) instead of `vmovups`, bypassing L1/L2 cache on writes.

### Numbers (i7-13700K, single thread, vec_add)

```
       N      OFF (us)    ON (us)   speedup
   16,384       9.79       10.41    0.94x   ← small N, NT slightly loses
   65,536      14.99       16.64    0.90x   ← still fits L2 (2MB), no pressure
  262,144      64.51       51.89    1.24x   ← exceeds L2, NT relieves pressure
1,048,576     237.61      198.42    1.20x
4,194,304    1471.07     1033.42    1.42x   ← sweet spot, fits L3 (30MB)
16,777,216   9927.27     7267.46    1.37x   ← exceeds L3, DRAM-bound
```

Correctness: bit-exact against torch in all cases.

## How it works

### What "non-temporal" means

A normal store (`vmovups`) goes through the CPU's cache hierarchy: the
target cache line is read into L1 (if not present), modified, and
later evicted to memory. For workloads where the kernel writes data
that it (or any soon-running kernel) won't re-read — typical "output
of an elementwise op" — that cache line evicts something useful from
L1/L2 to make room for data we're about to discard anyway.

`vmovntps` (or `movntdq`, `movnti`) writes directly to memory via the
write-combining buffer, **bypassing L1 and L2**. The cache lines stay
untouched. For streaming workloads exceeding L2 size, this frees
substantial cache bandwidth.

The trade-off: subsequent reads from the same address are full memory
trips (no L1/L2 hit). Hence the gate on "writer-only arg" — if the
caller re-reads soon, NT loses.

### Where the pass runs in the pipeline

`python/triton/backends/cpu/compiler.py`:
```python
cpu.passes.ttcpuir.add_vector_to_llvmir(pm, options.enable_fast_math)
cpu.passes.ttcpuir.add_non_temporal_store(pm)         # ← my pass
cpu.passes.ttcpuir.add_memref_to_llvmir(pm)
```

Runs **after** `vector.store` has been lowered to `llvm.store`, but
**before** the final LLVM IR emission. At this point the LLVM dialect
ops still have MLIR attributes (`nontemporal: UnitAttr`,
`alignment: I64Attr`) that propagate into the final LLVM IR's `store`
instruction.

### The write-only analysis

Triton-CPU wraps each function pointer arg in a memref descriptor:
```mlir
!llvm.struct<(allocated_ptr, aligned_ptr, offset, sizes, strides)>
```

So the raw arg `%X: !llvm.ptr` is reached by stores via:
```mlir
%desc1 = llvm.insertvalue %X, %desc[1] : !llvm.struct<...>
%ptr   = llvm.extractvalue %desc1[1] : !llvm.struct<...>
%addr  = llvm.getelementptr %ptr[%offs] : !llvm.ptr
llvm.store %val, %addr : vector<1024xf32>
```

The analyzer (`hasReader`) walks forward from each pointer arg,
following:
- pointer-aliasing ops: `gep` / `bitcast` / `addrspacecast`
- struct embedding: `insertvalue` (push position onto a stack) and
  `extractvalue` (peel matching prefix off the stack)

If any reachable use is a `load`, `llvm.intr.masked.load`, `memcpy`,
or call op, the arg is *not* write-only. Otherwise it's classified
as write-only.

For each `llvm.store`, `traceAddrRoot` walks the address operand
back through the same set of ops to find the originating SSA value.
If it's a write-only function arg, the store is eligible.

### Why bump alignment to 32

LLVM's x86 backend has an inflexible rule: it only emits
`vmovntps`/`vmovntdq` if it can prove the store is 32-byte aligned.
Otherwise it scalarizes a `<1024 x float>` NT store into hundreds of
`movntiq` (scalar 64-bit) — bandwidth disaster, **worse** than the
original `vmovups`.

Triton-CPU's lowering annotates stores with `align 4` (one float
width). The actual underlying tensor from `torch.randn(...)` is at
least 64-byte aligned (PyTorch's allocator pool). So bumping the
alignment annotation is safe and unblocks the wide NT emission.

If the underlying buffer is *not* 32-byte aligned (rare with PyTorch
but possible with external allocators), this can cause a GPF. Use the
env-var gate to disable in that case.

## Scope limits

What the PoC deliberately doesn't handle:

1. **`vector.maskedstore`** lowers to `llvm.intr.masked.store` —
   that MLIR op doesn't expose a `nontemporal` attribute and its
   `llvmBuilder` doesn't pass through any nontemporal flag. To support
   masked stores would need either a patch to the MLIR LLVM dialect
   or a custom lowering that emits the masked NT store via direct
   `LLVM::CallOp` to `@llvm.masked.store` with metadata. For now,
   kernels with `mask=` keyword on `tl.store` skip the optimization.

2. **No cross-kernel aliasing analysis.** If the next kernel will
   re-read this kernel's output, NT loses (we just evicted what they
   needed). Default-OFF protects the chained-kernel case; user must
   opt in per workload.

3. **Cache-size aware gating.** Pass marks unconditionally; for very
   small N the NT cost is paid without cache pressure to relieve, so
   we lose ~10%. Acceptable: hot path is large.

4. **AVX-512.** This machine (13700K) is AVX2 only. On AVX-512 hardware
   the same pattern emits `vmovntps zmm` (512-bit), bigger win.

## File map

| File | What changed |
|---|---|
| `third_party/cpu/lib/TritonCPUToLLVM/NonTemporalStore.cpp` | The pass (~155 lines, new) |
| `third_party/cpu/include/TritonCPUToLLVM/Passes.h` | One forward decl |
| `third_party/cpu/include/TritonCPUToLLVM/Passes.td` | TableGen entry |
| `third_party/cpu/lib/TritonCPUToLLVM/CMakeLists.txt` | Source list |
| `third_party/cpu/triton_cpu.cc` | Python binding (`add_non_temporal_store`) |
| `python/triton/backends/cpu/compiler.py` | Wire into pass pipeline |
| `python/test/unit/cpu/_bench_nt_store.py` | Subprocess A/B bench + correctness sweep |

## Build-fix collateral

Building this branch on i7-13700K + GCC 11 + bundled LLVM `20902f0b`
required three small pre-existing-bug fixes that block compilation
on any post-rebase tree:

| File | Fix |
|---|---|
| `include/triton/Dialect/Triton/IR/Dialect.h` | Drop `const` from `getName() final` (base virtual is non-const, so `const` makes it not-an-override; ptxas-style rejection) |
| `include/triton/Dialect/TritonGPU/IR/Dialect.h` | same |
| `include/triton/Dialect/TritonNvidiaGPU/IR/Dialect.h` | same |
| `third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h` | same |
| `lib/Target/LLVMIR/LLVMDIUtils.cpp:69` | Bundled LLVM has older 10-arg `DIDerivedTypeAttr::get`; drop the extra 3 args |
| `third_party/cpu/lib/TritonCPUTransforms/OptimizeMasks.cpp:240`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp:693` | Bundled LLVM lacks `DenseTypedElementsAttr`; rename to `DenseElementsAttr` |

These aren't NT-store work proper, but the build won't succeed without
them on this LLVM-skew configuration.

## Reproducing

```bash
conda run -n triton-cpu python python/test/unit/cpu/_bench_nt_store.py
```

Inspect the asm to verify NT emission:
```bash
rm -rf /tmp/nt_cache && \
TRITON_CACHE_DIR=/tmp/nt_cache TRITON_CPU_NT_STORE=1 \
  conda run -n triton-cpu python -c '...vec_add kernel...'
grep -c vmovntps /tmp/nt_cache/*/*.asm   # expect 128 for 1024-elem BLOCK
```
