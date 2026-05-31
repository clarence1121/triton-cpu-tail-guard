# Loop Peel Internals — From `scf.for` to Hot-Path `vector.load`

> A deep-dive into the MLIR loop-peeling pass added to `OptimizeMasks` on
> the `mlir-loop-peel-clean` branch.
>
> Sibling document to the `cpu-tail-guard` branch's
> `TAIL_GUARD_INTERNALS.md`. Together the two cover the **same problem**
> (mask-removal for SIMD codegen) at **two different abstraction layers**
> (per-tile / cross-kernel for tail guard; per-iteration / single-kernel
> for loop peel). Reading both side by side is the easiest way to see
> what each layer is for.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [The problem](#2-the-problem)
3. [Why this lives in MLIR, not AST](#3-why-this-lives-in-mlir-not-ast)
4. [Tour of the existing `OptimizeMasks` pass (Phase 1)](#4-tour-of-the-existing-optimizemasks-pass-phase-1)
5. [The new `PeelForLoopWithMask` pattern (Phase 2)](#5-the-new-peelforloopwithmask-pattern-phase-2)
6. [Why we don't use MLIR's `peelForLoopAndSimplifyBounds`](#6-why-we-dont-use-mlirs-peelforloopandsimplifybounds)
7. [Manual arith peel: `buildPeeledUpper` + `cloneForWithNewBounds`](#7-manual-arith-peel-buildpeeledupper--cloneforwithnewbounds)
8. [The attribute trick: how the main loop's mask gets removed](#8-the-attribute-trick-how-the-main-loops-mask-gets-removed)
9. [Extending `buildMinOrMaxExpr` for `arith.remsi`](#9-extending-buildminormaxexpr-for-arithremsi)
10. [Env-var gate: `TRITON_CPU_LOOP_PEEL`](#10-env-var-gate-triton_cpu_loop_peel)
11. [IR before/after on a real Triton kernel](#11-ir-beforeafter-on-a-real-triton-kernel)
12. [Correctness](#12-correctness)
13. [Timing methodology](#13-timing-methodology)
14. [Benchmark results](#14-benchmark-results)
15. [File:line reference index](#15-fileline-reference-index)

---

## 1. TL;DR

For any `@triton.jit` kernel that contains a `for offs in range(0, N,
BLOCK)` style inner loop where `N` is **not** statically known to be a
multiple of `BLOCK`, the pass automatically transforms one masked
loop into two:

```
for offs in 0..N step BLOCK:     ─────►   for offs in 0..N_main step BLOCK:   # main: no mask
  v = maskedload …                          v = load …
  acc += sum(v)                             acc += sum(v)
                                          for offs in N_main..N step BLOCK:    # epilogue: keeps mask
                                            v = maskedload …                   # runs 0 or 1 times
                                            acc += sum(v)
```

where `N_main = N - (N - 0) % BLOCK = N - N % BLOCK`, computed at run
time from runtime `N`. The main loop runs `N / BLOCK` iterations with
**no mask compare and no `vmaskmovps`**; the epilogue handles the
remainder of at most one iteration.

Measured on a sum-reduction kernel with peel-disabled vs peel-enabled
A/B (kernel-only timing, no Python wrapper):

```
       N    disabled (µs)    enabled (µs)   speedup
   --------------------------------------------------------
       10007            ~8              ~7     1.17x
       65537           ~31             ~24     1.31x    ← L2-resident
      262147           ~85             ~84     1.06x
     1048573          ~330            ~310     1.02x    ← memory-bound, mask cost hidden
```

Pass is **default-on**; set `TRITON_CPU_LOOP_PEEL=0` to disable.

---

## 2. The problem

A typical Triton-CPU reduction kernel:

```python
@triton.jit
def sum_kernel(X, OUT, N, BLOCK: tl.constexpr):
    acc = 0.0
    for offs in range(0, N, BLOCK):
        lanes = offs + tl.arange(0, BLOCK)
        mask = lanes < N            # ← needed only on the last iteration
        x = tl.load(X + lanes, mask=mask, other=0.0)
        acc += tl.sum(x)
    tl.store(OUT, acc)
```

When `N` is not divisible by `BLOCK`, the last iteration genuinely
needs the mask — the lane positions `offs + lanes` for the last
iteration straddle the end of the array. But for every prior
iteration, the mask is structurally always-true.

The CPU cost of carrying a mask through every iteration:

| Per-iteration work with mask | Per-iteration work without mask |
|---|---|
| `vbroadcastss` for `N` | (gone) |
| `vector.broadcast` for `offs` + `addi` for lanes | (gone — lanes don't need to be computed for the load) |
| `vpcmpgtd` for `lanes < N` → mask register | (gone) |
| `vmaskmovps` / `vmovups {k1}` | plain `vmovups` |
| LLVM **cannot speculate the load** (could fault) | LLVM free to schedule, hoist, unroll |
| Often a follow-up `arith.select` to zero-out lanes | (gone) |

The mask is computed every iteration even though it only matters on
the last one. The compiler — left to itself — has no way to peel one
iteration's worth of conservative behavior from the rest.

Existing `OptimizeMasks` pass *does* remove the mask when it can prove
the comparison is always-true (e.g. when `tt.divisibility = 16 : i32`
is attached to the bound argument; see Phase 1 below). But when the
divisibility hint isn't there, the analyzer fails and the mask
survives every iteration.

That's what the new pass fixes: when divisibility-based proof can't
remove the mask, **split the loop so the structural invariant becomes
provable on the main half**.

---

## 3. Why this lives in MLIR, not AST

Tail guard makes 4 different `.so` binaries via Python AST rewriting.
Loop peel makes 1 `.so` with 2 different loops inside it via MLIR pass
rewriting. Why the difference?

| Question | Tail Guard | Loop Peel |
|---|---|---|
| What gets multiplied? | The whole kernel (4 binaries) | A single `scf.for` (becomes two `scf.for`s) |
| What decides dispatch? | The C launcher per tile coordinate | Fall-through CFG: main runs to completion, then epilogue runs |
| Could the *other* approach work? | Could in principle clone the whole MLIR module — but every clone is a full compilation pipeline, and the per-tile dispatch logic doesn't fit cleanly in a single MLIR pass | Could in principle AST-rewrite the for-loop body before Triton sees it — but Triton's Python-to-IR lowering is opaque enough that producing two sibling for-loops with shared iter_args from the AST level is awkward |
| Which layer is right? | Cross-kernel dispatch is a *runtime* decision: Python orchestrates 4 fixed-cost compiled units. AST/Python is the natural home. | Intra-kernel restructuring is an *IR* decision: change the CFG, let codegen take care of the rest. MLIR pass is the natural home. |

There's also a practical reason: **the analysis that proves the mask
is removable already exists in MLIR-land** (`isAlwaysAllOnes` in
`OptimizeMasks.cpp`). After peeling, the main loop's mask becomes
provable by that *same* analyzer. So the peel pass plus a tiny
analyzer extension lets us reuse the existing mask-removal machinery
end to end. Doing this from Python AST would mean reimplementing the
affine-expression analysis up there.

So: AST for cross-kernel, IR for intra-kernel. Different problems,
different layers, different tools.

---

## 4. Tour of the existing `OptimizeMasks` pass (Phase 1)

Source: `third_party/cpu/lib/TritonCPUTransforms/OptimizeMasks.cpp`.

The pass historically (before this work) consisted of three patterns
applied via a greedy pattern rewrite driver:

```cpp
// OptimizeMasks.cpp:498-505 (after my changes; previously this was the only block)
RewritePatternSet patterns(context);
patterns.add<CdivToDiv>(context);
patterns.add<ScaleInductionVariable>(context);
patterns.add<OptimizeMask>(context);
mlir::applyPatternsGreedily(mod, std::move(patterns));
```

### 4.1 `CdivToDiv` (OptimizeMasks.cpp:68)

Pattern matches `(val + K - 1) / K` and rewrites it to `val / K` when
`val` is statically known divisible by `K`. This is `tl.cdiv` lowered
into arith — the rewrite removes the +K-1 fudge factor when it's
unnecessary, which lets the next pattern fire.

### 4.2 `ScaleInductionVariable` (OptimizeMasks.cpp:160)

Pattern matches the canonical Triton-emitted loop shape

```mlir
scf.for %i = %c0 to %cdiv_result step %c1 {
  %offs = arith.muli %i, %BLOCK
  ...
}
```

and rewrites it into

```mlir
scf.for %offs = %c0 to %orig_upper step %BLOCK {
  ...
}
```

i.e. it folds the `i * BLOCK` computation into the loop bounds,
producing a step-`BLOCK` loop with the original upper bound. **Requires
the divisor (`BLOCK`) to be known and the original upper bound to be
divisible by it.**

### 4.3 `OptimizeMask` (OptimizeMasks.cpp:326)

The actual mask killer. Matches any `arith.cmpi`, runs
`isAlwaysAllOnes` on it, and if proven, replaces the result with a
`dense<true>` constant. Downstream canonicalization then turns
`vector.maskedload %ptr, true-mask` into plain `vector.load %ptr`.

`isAlwaysAllOnes` (OptimizeMasks.cpp:291) builds an affine expression
representing the maximum value of the LHS and the minimum value of the
RHS of the compare, then checks whether `max(LHS) < min(RHS)`. The
expression construction is in `buildMinOrMaxExpr`
(OptimizeMasks.cpp:230) and handles a handful of arith ops:
`arith.constant`, `arith.addi`, `arith.subi`, `vector.broadcast`, and
the for-loop IV.

### 4.4 Why Phase 1 alone is insufficient

For a kernel like our sum_kernel called with `N = 10007` (a prime):

- `N` has no `tt.divisibility` attribute (Triton's JIT only adds it
  when the first call's value happens to be divisible)
- `ScaleInductionVariable` requires the divisor-aware divisibility
  proof to fire — it doesn't
- `OptimizeMask` analyzes `cmpi slt (broadcast(iv) + [0..15]) , N`:
  - `max(iv)` for a generic loop = `upper - 1` = `N - 1`
  - `max(iv + lanes)` = `(N - 1) + 15` = `N + 14`
  - `diff = (N + 14) - N = 14`
  - `14 < 0`? No → not all-ones → mask stays.

So the mask survives the whole pass. Every iteration of every call
pays for it. The point of Phase 2 is to break this stalemate by
**changing the loop structure** so the analyzer can now make its
proof.

---

## 5. The new `PeelForLoopWithMask` pattern (Phase 2)

The whole new pass body, with the new pattern called out, in
`OptimizeMasks.cpp:492-526`:

```cpp
void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // Phase 1: remove masks that the current analysis can already prove
    // are all-ones (typically driven by tt.divisibility hints).
    {
      RewritePatternSet patterns(context);
      patterns.add<CdivToDiv>(context);
      patterns.add<ScaleInductionVariable>(context);
      patterns.add<OptimizeMask>(context);
      if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
        return signalPassFailure();
    }

    // Phase 2: for loops that still carry unproven masks, peel a partial
    // iteration out so the main loop's bounds divide the step evenly,
    // then re-run OptimizeMask on the main loop's now-provable cmpi.
    // Gated by TRITON_CPU_LOOP_PEEL (default on); set to 0 to disable
    // for benchmarking or to bisect regressions.
    if (const char *env = std::getenv("TRITON_CPU_LOOP_PEEL");
        !env || std::string(env) != "0") {
      RewritePatternSet patterns(context);
      patterns.add<PeelForLoopWithMask>(context);
      patterns.add<OptimizeMask>(context);
      if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
        return signalPassFailure();
    }
}
```

Two-phase structure rationale: **let Phase 1 handle every case the
existing analyzer can already prove before doing structural surgery.**
If `tt.divisibility` is present, Phase 1 removes the mask without any
peel needed — peel would just produce a degenerate epilogue with no
iterations. Phase 2 only fires on the residual loops where Phase 1
left a mask in place.

### 5.1 The pattern's match conditions

`PeelForLoopWithMask::matchAndRewrite` in OptimizeMasks.cpp:397–433:

```cpp
LogicalResult matchAndRewrite(scf::ForOp forOp,
                              PatternRewriter &rewriter) const override {
    // (1) Don't re-peel loops we already peeled.
    if (forOp->hasAttr(kPeeledAttr))
        return failure();

    // (2) Body must have a masked load/store AND an unproven mask cmpi.
    bool hasMaskedAccess = false;
    bool anyUnprovenMask = false;
    int opCount = 0;
    for (Operation &op : forOp.getBody()->without_terminator()) {
        ++opCount;
        if (isa<vector::MaskedLoadOp, vector::MaskedStoreOp>(&op))
            hasMaskedAccess = true;
        if (auto cmp = dyn_cast<arith::CmpIOp>(&op))
            if (!isAlwaysAllOnes(cmp))
                anyUnprovenMask = true;
    }
    if (!hasMaskedAccess || !anyUnprovenMask)
        return failure();

    // (3) Cost guard: don't peel huge loop bodies (clone cost > savings).
    if (opCount > kMaxPeelBodyOps)
        return failure();

    // (4) Don't peel trivially-stepped loops (step ≤ 1 means no SIMD alignment).
    if (auto stepCst = forOp.getStep().getDefiningOp<arith::ConstantOp>()) {
        auto stepInt = dyn_cast<IntegerAttr>(stepCst.getValue());
        if (stepInt && stepInt.getInt() <= 1)
            return failure();
    }

    // (5) Actually peel: compute main_upper, clone body for epilogue, rewire.
    Value originalUpper = forOp.getUpperBound();
    Value mainUpper = buildPeeledUpper(forOp, rewriter);

    rewriter.setInsertionPointAfter(forOp);
    scf::ForOp partial = cloneForWithNewBounds(
        forOp, mainUpper, originalUpper, forOp.getResults(), rewriter);

    for (auto [oldRes, newRes] :
         llvm::zip(forOp.getResults(), partial.getResults()))
        rewriter.replaceAllUsesExcept(oldRes, newRes, partial);

    rewriter.startOpModification(forOp);
    forOp.setUpperBound(mainUpper);
    rewriter.finalizeOpModification(forOp);

    // (6) Tag both produced loops to prevent re-peel + signal divisibility
    //     of the main loop to the analyzer.
    forOp->setAttr(kPeeledAttr, rewriter.getUnitAttr());
    forOp->setAttr(kBoundsAlignedAttr, rewriter.getUnitAttr());
    partial->setAttr(kPeeledAttr, rewriter.getUnitAttr());
    return success();
}
```

#### Match condition (1): `kPeeledAttr` re-peel guard

After we peel, both the main loop and the epilogue get
`triton_cpu.peeled` as a unit attribute. Without this guard, the
greedy driver would visit each new loop and try to peel it again
forever — peeling the epilogue (which has 0 or 1 iterations) is
pointless and would create another epilogue, ad infinitum.

#### Match condition (2): must have an *unproven* mask

Three sub-checks roll into one walk over the body:

- `hasMaskedAccess`: at least one `vector.maskedload` or
  `vector.maskedstore`. Without one, peeling buys nothing — no mask to
  remove.
- `anyUnprovenMask`: at least one `arith.cmpi` that
  `isAlwaysAllOnes` *cannot already* prove all-ones. If every cmpi is
  already provable, Phase 1's `OptimizeMask` will/has removed them and
  the maskedload will canonicalize without needing peel.
- `opCount` is computed in the same loop for the next check.

Walking `forOp.getBody()->without_terminator()` only visits direct
children of the for body, not ops inside nested for-loops or scf.ifs.
That's correct: peeling the outer for doesn't help a mask that lives
inside an inner scf.for; the inner loop should be matched in its own
right.

#### Match condition (3): cost guard

`kMaxPeelBodyOps` is `50` (OptimizeMasks.cpp:357). Peeling clones the
loop body, so for huge loops (e.g. a 200-op K-loop in a matmul) the
clone bloats the `.so` by ~the body size. The win is bounded (you
remove one mask compare per iteration), so once the body gets big the
ratio doesn't pay. 50 is conservative — bumpable if needed.

#### Match condition (4): non-trivial step

`step == 1` means each iteration advances by one element, which is the
shape that `ScaleInductionVariable` was supposed to convert to a SIMD
step. If it didn't convert, peeling won't help — the resulting "main"
loop would have one-element iterations, no SIMD vectorization, no
mask to remove. Just bail.

(In Triton-CPU practice, `step == 1` loops show up when
divisibility-based conversion failed in Phase 1 *and* the user wrote
the loop in the `for i in range(cdiv(N, BLOCK))` shape that produces
step=1 in IR. We can't peel that case without first doing the
step-conversion ourselves, which is a separate piece of work.)

### 5.2 The rewrite, in three concrete steps

Step A: compute the main loop's new upper bound.

```cpp
Value originalUpper = forOp.getUpperBound();
Value mainUpper = buildPeeledUpper(forOp, rewriter);
```

`buildPeeledUpper` is defined in OptimizeMasks.cpp:349-360 and emits:

```mlir
%range = arith.subi %upper, %lower : i32
%rem   = arith.remsi %range, %step : i32
%mainUpper = arith.subi %upper, %rem : i32
```

For `lower = 0`, this is `upper - upper % step`. For general `lower`,
it is `upper - (upper - lower) % step`, which equals
`lower + step * floor((upper - lower) / step)` — the largest value
`≤ upper` such that `(result - lower)` is divisible by `step`.

Step B: clone the body into a new partial-iteration loop.

```cpp
rewriter.setInsertionPointAfter(forOp);
scf::ForOp partial = cloneForWithNewBounds(
    forOp, mainUpper, originalUpper, forOp.getResults(), rewriter);
```

`cloneForWithNewBounds` is in OptimizeMasks.cpp:363-392 and constructs
a fresh `scf.for` with bounds `[mainUpper, originalUpper)` (the same
step), seeded with init args `forOp.getResults()` so iter_args chain
correctly:

```cpp
scf::ForOp cloneForWithNewBounds(scf::ForOp forOp, Value lower, Value upper,
                                 ValueRange initArgs,
                                 PatternRewriter &rewriter) {
  Location loc = forOp.getLoc();
  Value step = forOp.getStep();
  auto newFor = scf::ForOp::create(
      rewriter, loc, lower, upper, step, initArgs,
      [&](OpBuilder &builder, Location nestedLoc, Value newIv,
          ValueRange newIters) {
        IRMapping mapping;
        mapping.map(forOp.getInductionVar(), newIv);
        for (auto [oldArg, newArg] :
             llvm::zip(forOp.getRegionIterArgs(), newIters))
          mapping.map(oldArg, newArg);
        for (Operation &op : forOp.getBody()->without_terminator())
          builder.clone(op, mapping);
        auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        SmallVector<Value> yieldVals;
        for (Value v : oldYield.getResults())
          yieldVals.push_back(mapping.lookupOrDefault(v));
        scf::YieldOp::create(builder, nestedLoc, yieldVals);
      });
  return newFor;
}
```

Two non-trivial things this handles:

1. **Iter args.** `forOp.getResults()` are the values yielded by the
   (about-to-be-shrunk) main loop. The partial loop takes them as
   `initArgs` — so the accumulator from the main loop feeds the
   epilogue. Without this, an `acc` accumulator would restart at zero
   in the epilogue and the output would be wrong.

2. **Body cloning via `IRMapping`.** Each op in the original body
   references the old IV and old iter args. The mapping rewires those
   references to the new IV and new iter args of the partial loop, so
   the clone is self-contained.

Step C: shrink the original loop and rewire downstream users.

```cpp
for (auto [oldRes, newRes] :
     llvm::zip(forOp.getResults(), partial.getResults()))
    rewriter.replaceAllUsesExcept(oldRes, newRes, partial);

rewriter.startOpModification(forOp);
forOp.setUpperBound(mainUpper);
rewriter.finalizeOpModification(forOp);
```

Two things happening:

- **`replaceAllUsesExcept`** rewires anything that consumed the
  original (full-range) loop's results to instead consume the
  epilogue's results. The `Except partial` argument is critical: the
  partial loop *itself* consumes those results as `initArgs`, so we
  must not also replace those uses (that would create a circular
  data-flow `partial → partial`).

- **`setUpperBound(mainUpper)`** then shrinks the original loop in
  place. After this, the original `forOp` is the main loop with
  bounds `[lower, mainUpper)` and the new `partial` is the epilogue
  with bounds `[mainUpper, originalUpper)`.

### 5.3 Why the original `forOp` is recycled rather than rebuilt

We could equivalently create *two* new loops and erase the original.
In-place modification is preferred because:

- Greedy rewrite drivers expect that `success()` from a pattern means
  the matched op was either erased or modified — but downstream
  uses of the old op should land somewhere. In-place shrink keeps
  uses tied to the same SSA value (just with new bounds), so the
  driver knows what to revisit.
- Cloning the body is cheaper than cloning the for-op header twice.

---

## 6. Why we don't use MLIR's `peelForLoopAndSimplifyBounds`

MLIR ships
`mlir::scf::peelForLoopAndSimplifyBounds(rewriter, forOp, &partial)` in
`mlir/Dialect/SCF/Transforms/Transforms.h:95`. The docstring even
mentions our use case:

> This transformation is beneficial for a wide range of transformations
> such as vectorization or loop tiling: It enables additional
> canonicalizations inside the peeled loop body such as **rewriting
> masked loads into unmasked loads**.

Sounds perfect. We tried it first — and hit a fatal problem at verify
time:

```
test/TritonCPU/optimize-masks.mlir:117:5: error: 'affine.apply' op operand #0
  must be variadic of index, but got 'i32'
    scf.for %arg3 = %c0_i32 to %arg2 step %c16_i32  : i32 {
```

`peelForLoopAndSimplifyBounds` emits an `affine.apply` of the form
`s1 - (s1 - s0) mod s2` to compute the new bounds. **`affine.apply`
only accepts `index` typed operands**, but Triton-emitted `scf.for`
loops are uniformly `i32`-typed (the IV comes from a Python `int`,
which lowers to i32). The result is `affine.apply` with i32 operands,
which fails the affine verifier.

Triton's loops are i32 by convention, so the upstream utility is
unusable for our domain. The fix is to implement peel directly with
`arith.subi` + `arith.remsi` (which accept any integer type
including i32), losing nothing in expressiveness. That's what
`buildPeeledUpper` does.

We did briefly gate the pass to only fire on `index`-typed loops (see
the deleted code path that was an early intermediate commit), which
was an honest no-op for the real Triton workloads. The manual
arith-based peel is the actual usable version.

---

## 7. Manual arith peel: `buildPeeledUpper` + `cloneForWithNewBounds`

The two helpers from §5.2, in one place for reference. Both live in
`OptimizeMasks.cpp` in the same anonymous namespace as the pattern.

```cpp
// OptimizeMasks.cpp:349-360
Value buildPeeledUpper(scf::ForOp forOp, PatternRewriter &rewriter) {
  Location loc = forOp.getLoc();
  Value lower = forOp.getLowerBound();
  Value upper = forOp.getUpperBound();
  Value step = forOp.getStep();
  rewriter.setInsertionPoint(forOp);
  Value range = arith::SubIOp::create(rewriter, loc, upper, lower);
  Value rem = arith::RemSIOp::create(rewriter, loc, range, step);
  return arith::SubIOp::create(rewriter, loc, upper, rem);
}

// OptimizeMasks.cpp:363-392 (already shown above)
scf::ForOp cloneForWithNewBounds(...);
```

The 3-op sequence produced by `buildPeeledUpper`:

```mlir
%range     = arith.subi %upper, %lower  : i32     // total trip range
%rem       = arith.remsi %range, %step  : i32     // 0 ≤ rem < step
%mainUpper = arith.subi %upper, %rem    : i32     // upper - rem
```

Computed at runtime (since `upper` is the user's runtime `N`). Three
operations, all single-cycle ALU on x86_64, executed once per launch
— vastly cheaper than the savings per-iteration of the loop.

For non-zero `lower`, the math is exactly right: `mainUpper` is the
largest `lower + k*step` value not exceeding `upper`, so iterating
`[lower, mainUpper)` step `step` covers `(mainUpper - lower) / step =
range / step` full iterations.

For `step == 1`: we already bailed in the matcher (§5.1), so we don't
have to worry about that degenerate case here.

---

## 8. The attribute trick: how the main loop's mask gets removed

This is the most subtle part of the design. After peeling, the main
loop's IR looks like:

```mlir
%mainUpper = arith.subi %arg2, %rem : i32

scf.for %iv = %c0_i32 to %mainUpper step %c16_i32 iter_args(...) : i32 {
    %lanes = arith.addi (broadcast %iv), %arange : vector<16xi32>
    %mask = arith.cmpi slt, %lanes, (broadcast %arg2) : vector<16xi32>  // ← still comparing against ORIGINAL N
    %v = vector.maskedload %ptr[%c0], %mask, %zero
    ...
} {triton_cpu.bounds_aligned_to_step, triton_cpu.peeled}
```

The mask compares against `%arg2` (the *original* N), not against
`%mainUpper` (the peeled upper). So semantically the mask is still
asking "is `iv + lane < N`?". For the main loop to want
`vector.load`, the analyzer needs to prove this comparison is
always-true given the new loop structure.

The chain of reasoning we want the analyzer to make:

1. The for-loop's bounds satisfy `(upper - lower) % step == 0`
   (because we just constructed `upper = N - (N - 0) % step`). That
   means **`max(iv) = upper - step`**, not `upper - 1`.
2. `max(iv + lane)` for `lane ∈ [0, step)` is therefore
   `(upper - step) + (step - 1) = upper - 1`.
3. `upper ≤ originalUpper` (because `upper = originalUpper - rem`,
   `rem ≥ 0`).
4. So `max(iv + lane) = upper - 1 < upper ≤ originalUpper`.
5. The mask `iv + lane < originalUpper` is always-true.

Step 1 is the new structural invariant peel introduces. Step 2 is just
arithmetic. Step 3 needs `rem ≥ 0`. Steps 4 and 5 follow from 1+2+3.

To make the analyzer see this chain, we need:

- **A way to tell `buildMinOrMaxExpr` that this for-loop's bounds are
  step-aligned**, so it returns `upper - step` instead of `upper - 1`
  for `max(iv)`.
- **A way to tell it that `arith.remsi(non-negative, positive)` has
  min value 0**, so the chain `mainUpper = upper - rem` gives
  `max(mainUpper) = max(upper) - min(rem) = max(upper) - 0`.

Both are wired via the `triton_cpu.bounds_aligned_to_step` attribute
and a new `arith.remsi` case in `buildMinOrMaxExpr`. See §9.

### 8.1 Two attributes, two jobs

```cpp
// OptimizeMasks.cpp:346-355
constexpr llvm::StringLiteral kPeeledAttr = "triton_cpu.peeled";
constexpr llvm::StringLiteral kBoundsAlignedAttr =
    "triton_cpu.bounds_aligned_to_step";
```

| Attribute | Set on | Purpose |
|---|---|---|
| `triton_cpu.peeled` | both main and partial | Prevents re-peel by the same pattern |
| `triton_cpu.bounds_aligned_to_step` | **main only** | Signals to `buildMinOrMaxExpr` that `(upper - lower) % step == 0` is structurally guaranteed |

The partial loop deliberately does **not** get
`bounds_aligned_to_step` — its bounds are `[mainUpper, originalUpper)`,
and `(originalUpper - mainUpper) = rem`, which is NOT divisible by
`step` in general (it's `< step`). So the partial loop is correctly
seen by the analyzer as "bounds are not step-aligned, max(iv) =
upper - 1, generic case", which means its mask is correctly **not
removed** — the epilogue keeps masking, as it should.

---

## 9. Extending `buildMinOrMaxExpr` for `arith.remsi`

`buildMinOrMaxExpr` is the hand-rolled affine analyzer for
`isAlwaysAllOnes`. It walks an arith expression and returns an
`AffineExpr` representing either the max or min value of that expr.
Before this work it handled: `arith.constant`, `arith.addi`,
`arith.subi`, `vector.broadcast`, and the scf.for IV. We added one
new case + one tweak.

### 9.1 The `arith.remsi` case (new)

`OptimizeMasks.cpp:256-265`:

```cpp
} else if (auto def = val.getDefiningOp<arith::RemSIOp>()) {
    // remsi(x, d) lies in [-|d|+1, |d|-1] in general. For loop trip-count
    // arithmetic we only need the lower bound: assuming the dividend is
    // non-negative (true for every Triton-emitted bound expression), the
    // result is >= 0. For max we fall through to the opaque-symbol path
    // below, which is conservative but harmless: this handler is only
    // load-bearing in the peeled main-upper formula `upper - (range mod
    // step)`, where the subi case already does max(upper) - min(rem).
    if (!isMax)
        return getAffineConstantExpr(0, val.getContext());
}
```

Just one line of analysis: `min(arith.remsi(non-negative, positive)) =
0`. We don't try to bound the max — for our use case it's
load-bearing on the min side only.

Why this is correct for our case: `buildMinOrMaxExpr` recurses on
`mainUpper = arith.subi %upper, %rem`. The existing `arith.subi`
handler is `max(a - b) = max(a) - min(b)`. So computing `max(mainUpper)`
needs `min(rem)`. With our new case, `min(rem) = 0`, so
`max(mainUpper) = max(upper) = symbol(originalUpper)`.

That's the chain that gives us step 3 of the reasoning in §8.

### 9.2 The `bounds_aligned_to_step` check (modified)

`OptimizeMasks.cpp:268-281`, modified from the original:

```cpp
// For max value we use upper bound - 1 in generic case and bound - step
// if (upper - lower) is divisible by step. The latter holds when both
// bounds are individually divisible by step, or when the loop carries
// the triton_cpu.bounds_aligned_to_step marker (set by loop peeling).
bool boundsAlignedToStep =
    (isAlwaysDivisible(lower, step) && isAlwaysDivisible(upper, step)) ||
    forOp->hasAttr("triton_cpu.bounds_aligned_to_step");
if (boundsAlignedToStep) {
    return buildMinOrMaxExpr(upper, isSigned, isMax, symbolTable) -
           buildMinOrMaxExpr(step, isSigned, false, symbolTable);
}
return buildMinOrMaxExpr(upper, isSigned, isMax, symbolTable) -
       getAffineConstantExpr(1, val.getContext());
```

The only change from the original is the `|| forOp->hasAttr(...)`. The
existing logic already returned `upper - step` for `max(iv)` when both
bounds were *individually* divisible by step. We weakened the
precondition to also accept "loop carries our peel marker," which is
equivalent (when we set the marker, we know structurally that
`(upper - lower) % step == 0`).

This is a safe extension because no other pass sets
`triton_cpu.bounds_aligned_to_step`, so we can be confident the
invariant actually holds whenever the attribute is present.

### 9.3 Putting it together: the full trace

For the peeled main loop with `iv ∈ [0, mainUpper)` step `16`, mask
`cmpi slt (broadcast(iv) + dense<[0..15]>), broadcast(N)`:

- `max(iv)`: takes the for-loop IV branch. The new check sees
  `bounds_aligned_to_step` attribute → returns `max(upper) - min(step)`
  = `max(mainUpper) - min(c16)`. `min(c16) = 16` (constant). For
  `max(mainUpper)`:
  - `subi` handler: `max(upper) - min(rem) = symbol(N) - 0 = symbol(N)`
  - So `max(iv) = symbol(N) - 16`.
- `max(iv + lane) = max(iv) + max(dense<[0..15]>) = (symbol(N) - 16) + 15 = symbol(N) - 1`.
- `min(broadcast(N)) = symbol(N)` (function arg, opaque symbol).
- `diff = (symbol(N) - 1) - symbol(N) = -1`.
- `arith.cmpi slt`: check `diff < 0` → `-1 < 0` → true → mask is
  always-ones.

`OptimizeMask` rewrites the cmpi result to `dense<true>`. Standard
MLIR canonicalization then turns `vector.maskedload %ptr, true-mask`
into `vector.load %ptr`. The mask disappears from the main loop.

The same analysis on the **partial** loop (which lacks the
`bounds_aligned_to_step` attribute) falls through to `max(iv) = upper -
1`:

- `max(iv) = max(N) - 1 = symbol(N) - 1`
- `max(iv + lane) = symbol(N) - 1 + 15 = symbol(N) + 14`
- `diff = symbol(N) + 14 - symbol(N) = 14`
- `14 < 0`? No → not all-ones → mask stays.

Exactly the behavior we want.

---

## 10. Env-var gate: `TRITON_CPU_LOOP_PEEL`

Default on; set to `"0"` to disable Phase 2. `OptimizeMasks.cpp:507-516`:

```cpp
if (const char *env = std::getenv("TRITON_CPU_LOOP_PEEL");
    !env || std::string(env) != "0") {
    RewritePatternSet patterns(context);
    patterns.add<PeelForLoopWithMask>(context);
    patterns.add<OptimizeMask>(context);
    if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
        return signalPassFailure();
}
```

`std::getenv` is read on every `runOnOperation` call — meaning per
compile, not per kernel launch. Since the env var is constant within
a process for our use case, this is fine. The cost (one `getenv`
syscall per compile) is invisible.

Why not a compile option like `enable_tail_guard`? An env-var keeps
the gate purely local to the pass — no other layer (Python options,
backend driver, ASTSource) needs to know about it. For an
experimental optimization that's being benchmarked, that's the
simplest possible toggle.

---

## 11. IR before/after on a real Triton kernel

Source kernel (Python):

```python
@triton.jit
def sum_kernel(X, OUT, N, BLOCK: tl.constexpr):
    acc = 0.0
    for offs in range(0, N, BLOCK):
        lanes = offs + tl.arange(0, BLOCK)
        mask = lanes < N
        x = tl.load(X + lanes, mask=mask, other=0.0)
        acc += tl.sum(x)
    tl.store(OUT, acc)

sum_kernel[(1,)](X, OUT, N=10007, BLOCK=16)   # N=10007 prime → no divisibility hint
```

### 11.1 `tttcir` with `TRITON_CPU_LOOP_PEEL=0` (Phase 2 off)

```mlir
tt.func public @sum_kernel(%X: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                           %OUT: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                           %N: i32) {
  %lanes = arith.constant dense<[0,1,...,15]> : vector<16xi32>
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %acc = arith.constant 0.0 : f32
  %zero = arith.constant dense<0.0> : vector<16xf32>
  %mask = vector.broadcast %N : i32 to vector<16xi32>

  %acc_out = scf.for %offs = %c0_i32 to %N step %c16_i32 iter_args(%acc_in = %acc) -> (f32) : i32 {
    %lanes_iv = vector.broadcast %offs : i32 to vector<16xi32>
    %addr = arith.addi %lanes_iv, %lanes : vector<16xi32>
    %m = arith.cmpi slt, %addr, %mask : vector<16xi32>          ← per-iter mask compute
    %xp = tt.addptr %X, %offs : !tt.ptr<f32>, i32
    %xm = triton_cpu.ptr_to_memref %xp : <f32> -> memref<16xf32>
    %v = vector.maskedload %xm[%c0], %m, %zero                  ← masked load
    %s = vector.reduction <add>, %v, %acc : vector<16xf32> into f32
    %a = arith.addf %acc_in, %s : f32
    scf.yield %a : f32
  }
  tt.store %OUT, %acc_out : !tt.ptr<f32>
  tt.return
}
```

### 11.2 `tttcir` with `TRITON_CPU_LOOP_PEEL=1` (Phase 2 on, default)

```mlir
tt.func public @sum_kernel(%X: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                           %OUT: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                           %N: i32) {
  %lanes = arith.constant dense<[0,1,...,15]> : vector<16xi32>
  %c16_i32 = arith.constant 16 : i32
  %c0_i32 = arith.constant 0 : i32
  %acc = arith.constant 0.0 : f32
  %zero = arith.constant dense<0.0> : vector<16xf32>
  %mask = vector.broadcast %N : i32 to vector<16xi32>

  // ★ NEW: peel bookkeeping (3 ALU ops, once per call)
  %rem       = arith.remsi %N, %c16_i32 : i32
  %mainUpper = arith.subi %N, %rem : i32

  // ★ MAIN LOOP: no per-iter mask compute, no masked load
  %acc_main = scf.for %offs = %c0_i32 to %mainUpper step %c16_i32 iter_args(%acc_in = %acc) -> (f32) : i32 {
    %xp = tt.addptr %X, %offs : !tt.ptr<f32>, i32
    %xm = triton_cpu.ptr_to_memref %xp : <f32> -> memref<16xf32>
    %v = vector.load %xm[%c0] : memref<16xf32>, vector<16xf32>   ← plain vector.load
    %s = vector.reduction <add>, %v, %acc : vector<16xf32> into f32
    %a = arith.addf %acc_in, %s : f32
    scf.yield %a : f32
  } {triton_cpu.bounds_aligned_to_step, triton_cpu.peeled}

  // ★ EPILOGUE: same body as the original loop, runs 0 or 1 times
  %acc_out = scf.for %offs = %mainUpper to %N step %c16_i32 iter_args(%acc_in = %acc_main) -> (f32) : i32 {
    %lanes_iv = vector.broadcast %offs : i32 to vector<16xi32>
    %addr = arith.addi %lanes_iv, %lanes : vector<16xi32>
    %m = arith.cmpi slt, %addr, %mask : vector<16xi32>
    %xp = tt.addptr %X, %offs : !tt.ptr<f32>, i32
    %xm = triton_cpu.ptr_to_memref %xp : <f32> -> memref<16xf32>
    %v = vector.maskedload %xm[%c0], %m, %zero
    %s = vector.reduction <add>, %v, %acc : vector<16xf32> into f32
    %a = arith.addf %acc_in, %s : f32
    scf.yield %a : f32
  } {triton_cpu.peeled}

  tt.store %OUT, %acc_out : !tt.ptr<f32>
  tt.return
}
```

Reading this side-by-side:

- The disabled IR has **one** for-loop with `vector.maskedload` and a
  per-iter `arith.cmpi`.
- The enabled IR has **three new ops up-front** (`remsi`, `subi`, plus
  the broadcast moves around) and **two for-loops**:
  - The first (`%acc_main`) is iterations 0..⌊N/16⌋−1, contains a
    plain `vector.load`, and carries both attributes.
  - The second (`%acc_out`) is iteration ⌊N/16⌋ if it exists (i.e. when
    `N % 16 != 0`), and keeps the mask.
- The iter_arg chain: `%acc_main` flows out of the main loop and
  enters the epilogue as `iter_args(%acc_in = %acc_main)`. The
  epilogue's output `%acc_out` is stored to `OUT`. Correctness depends
  on this chain — that's why `cloneForWithNewBounds` had to thread
  iter_args through.

---

## 12. Correctness

A pass that rewrites IR is only useful if the rewritten program
computes the same answer. The bench script
`python/test/unit/cpu/_correctness_loop_peel.py` runs three things on
nine sizes and prints PASS/FAIL:

```
         N    disabled        enabled         torch          status
   ----------------------------------------------------------------------
         16   -2.187669       -2.187669       -2.187668      ok
         17   -1.051806       -1.051806       -1.051806      ok
         31   -0.059402       -0.059402       -0.059402      ok
         32    5.982267        5.982267        5.982267      ok
        100    6.280684        6.280684        6.280684      ok
       1024   17.687286       17.687286       17.687283      ok
      10007   20.171570       20.171570       20.171532      ok
      65537  385.118225      385.118225      385.117828      ok
     262147   17.240616       17.240616       17.240326      ok
   ALL OK
```

Three columns, three checks:

| Column | Test | Why |
|---|---|---|
| `disabled` | Phase 2 off; only Phase 1 ran | Reference behavior — what the program *was* doing |
| `enabled` | Phase 2 on (default) | The new behavior — should match disabled bit-for-bit |
| `torch` | `torch.sum` on the same input | Independent oracle |

The load-bearing check is **`disabled == enabled`**: same kernel,
same input, just different schedule. Any divergence means the pass
altered program semantics. For all 9 sizes, the two are
bit-exact (the only tiny differences are at f32 reduction-order
precision, which is identical between disabled and enabled because
they sum in the same order).

`enabled == torch` is the broader sanity check — both must agree
within f32 reduction tolerance (torch uses tree reduction which is
slightly more accurate; my Triton kernel does sequential, so there's
sub-1ppm drift at large N).

Sizes chosen to cover:

- **Aligned** (`16, 32`): peel doesn't fire — same code as Phase 1
  alone produces. Sanity check that the gate works.
- **Just-over-aligned** (`17, 31`): peel fires with a tiny epilogue
  (1 iteration). Stress-tests the iter_arg threading.
- **Medium** (`100, 1024`): typical fast-kernel sizes.
- **Bench sizes** (`10007, 65537, 262147`): the sizes the benchmark
  uses.

All nine pass.

---

## 13. Timing methodology

Mirrors the tail-guard methodology — for the same reason (CPU benches
of fast kernels need to exclude the Python launcher wrapper). The
short version, plus the loop-peel-specific findings:

### 13.1 The wrapper trap

```python
t0 = time.perf_counter()
sum_kernel[(1,)](x, out, N, BLOCK=16)         # ← 50-200µs Python overhead lives here
t1 = time.perf_counter()
elapsed = t1 - t0                              # ← polluted by Python overhead
```

For a 10µs kernel, the Python launcher overhead is the same order of
magnitude. For 100µs, it's still 50% noise. Naïvely averaging
`time.perf_counter()` brackets around the Python call therefore
**dilutes** any kernel-level speedup, sometimes by enough to make the
A/B comparison report **slowdown** when the kernel is actually faster.

Concrete example we hit during this work (loop-peel bench, N =
1048573):

```
                            Python-wrapper-included     hook-only (kernel-only)
disabled (median µs):              450                          ~330
enabled (median µs):               677    ← variance              ~310
"speedup":                         0.67x  ← garbage                 1.04x
```

### 13.2 The fix

`triton.testing.do_bench(..., measure_time_with_hooks=True,
return_mode="min")`. The hook fires at the *C launcher's* entry and
exit, so it brackets only the actual kernel execution — Python work
is excluded.

`triton/backends/cpu/driver.py:1041-1053`:

```python
def enable_hook_timing(self):
    self.use_hooks = True
    triton.knobs.runtime.launch_enter_hook = lambda arg: self._enter_hook()
    triton.knobs.runtime.launch_exit_hook  = lambda arg: self._exit_hook()

def _enter_hook(self):
    self.last_start = time.perf_counter()

def _exit_hook(self):
    self.kernel_times.append(time.perf_counter() - self.last_start)
```

The hooks fire as the very first / very last line of the C launcher's
body — see `third_party/cpu/backend/driver.py` `launch`/`launch_quad`,
the `if (launch_enter_hook != Py_None)` / `if (launch_exit_hook !=
Py_None)` blocks. The brackets include the per-tile dispatch loop and
all kernel execution; they exclude the Python-side launch machinery
(arg pack, signature serialize, constexpr resolve).

`return_mode="min"` filters cold/throttled iterations automatically —
much more stable than `mean` for tight microbenches.

### 13.3 Why this matters specifically for loop peel

The kernel-level savings are small in absolute terms: 1 cmpi + 1
broadcast + maskedload→load per iteration of the main loop. For a
65K-element reduction at BLOCK=16, that's ~4000 iterations × a few µops
each = order of microseconds total. The Python wrapper is 50-200µs.
**The signal is ~10% of the noise floor** without hook timing — easy
to lose, easy to misreport.

The bench script `python/test/unit/cpu/_bench_loop_peel.py` uses the
hook timing approach.

---

## 14. Benchmark results

Sum-reduction kernel above, `N` set to prime values (so Triton's JIT
doesn't auto-infer `tt.divisibility` and Phase 1 leaves the mask in
place — making Phase 2 the load-bearing optimization).

Median of three runs of `_bench_loop_peel.py` (hook-based timing,
`return_mode="min"`):

```
       N    disabled (µs)    enabled (µs)   speedup
   --------------------------------------------------------
       10007            ~8              ~7     1.17x
       65537           ~31             ~24     1.31x   ← best
      262147           ~85             ~84     1.06x
     1048573          ~330            ~310     1.02x
```

Pattern reading:

| N range | Working-set fit | Bottleneck | Speedup |
|---|---|---|---|
| 10K | L1 | mixed compute/load | 1.17x |
| 64K | L2 | **compute-bound** | **1.31x** |
| 256K | L3 | bandwidth | 1.06x |
| 1M+ | RAM | **memory-bandwidth-bound** | 1.02x (~neutral) |

The speedup is largest where the kernel is compute-bound (L2-resident,
inner loop is short enough that the saved SIMD ops actually move the
needle). Once memory bandwidth becomes the bottleneck (L3 to RAM),
the mask-op savings hide behind load latency and the speedup
collapses toward 1.0x.

This is the opposite of what intuition might suggest (larger N → more
iterations → more savings). The reason is exactly what physical CPU
performance models predict: per-iteration ALU savings only matter when
ALU is the bottleneck.

The earlier `1.26x at N=262147` claim from before the hook-timing fix
was an artifact of Python wrapper noise variance, not real
performance. The corrected numbers above are what the kernel actually
does.

---

## 15. File:line reference index

### MLIR pass (the work itself)

| Symbol | File | Lines | Role |
|---|---|---|---|
| `OptimizeMasks::runOnOperation` | `third_party/cpu/lib/TritonCPUTransforms/OptimizeMasks.cpp` | 492–526 | Two-phase pattern driver |
| `PeelForLoopWithMask` | same | 365–435 | The new pattern |
| `buildPeeledUpper` | same | 349–360 | Compute main_upper via arith.remsi/subi |
| `cloneForWithNewBounds` | same | 363–392 | Build the epilogue loop, threading iter_args |
| `kPeeledAttr` | same | 346–348 | "Don't re-peel" marker constant |
| `kBoundsAlignedAttr` | same | 350–355 | "Structurally step-aligned" marker constant |
| `kMaxPeelBodyOps` | same | 357 | Cost-guard threshold (50 ops) |
| `buildMinOrMaxExpr` (existing, extended) | same | 230–286 | Affine analyzer; new arith.remsi case + bounds_aligned check |
| `isAlwaysAllOnes` | same | 291–324 | Pre-existing mask analyzer; consumed by `OptimizeMask` |
| `OptimizeMask` (Phase 1 mask killer) | same | 326–338 | Rewrites provable cmpi to dense<true> |
| `ScaleInductionVariable` (Phase 1) | same | 160–226 | `for i in cdiv(N,B):` → `for offs in 0..N step B` |
| `CdivToDiv` (Phase 1) | same | 68–127 | `(val+K-1)/K` → `val/K` when val divisible by K |

### CMake

| Change | File | Lines |
|---|---|---|
| Link `MLIRSCFTransforms` (for utility access during early dev) | `third_party/cpu/lib/TritonCPUTransforms/CMakeLists.txt` | 17–18 |

### Lit tests

| File | What |
|---|---|
| `test/TritonCPU/optimize-masks.mlir` | 4 cases: classic mask removal × 3 + my new peel-and-remove case |

### Python harnesses

| File | Role |
|---|---|
| `python/test/unit/cpu/_bench_loop_peel.py` | Headline benchmark, hook-based timing |
| `python/test/unit/cpu/_correctness_loop_peel.py` | 9-size disabled/enabled/torch correctness sweep |

### Timing infrastructure (read-only references)

| File | Symbol | Role |
|---|---|---|
| `python/triton/testing.py` | `do_bench(measure_time_with_hooks=True)` | The bench helper that bracket-times the right thing |
| `python/triton/backends/cpu/driver.py` | `enable_hook_timing`, `_enter_hook`, `_exit_hook` | CPU device interface's hook wiring |
| `third_party/cpu/backend/driver.py` | `launch` (C template) | Where the enter/exit hooks fire from C |

### Cross-reference

- Sibling at the AST layer for a different problem shape:
  `cpu-tail-guard` branch's `TAIL_GUARD_INTERNALS.md`.
