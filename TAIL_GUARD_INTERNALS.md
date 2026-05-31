# Tail Guard Internals — From Python AST to Per-PID Binary Dispatch

> Companion deep-dive to [`CPU_DISPATCH_README.md`](CPU_DISPATCH_README.md).
> Where the README answers *what* and *why*, this document answers *how*,
> all the way from `ast.parse` on the user's `@triton.jit` source down to
> the `cmov` chain in the C launcher that picks one of four compiled
> binaries for each tile.

Read top-to-bottom; every section builds on the previous. File paths are
relative to the repo root; `file.py:NN` citations point at concrete lines
that you can `grep` and read.

---

## Table of contents

1. [The end-to-end picture in 20 lines](#1-the-end-to-end-picture-in-20-lines)
2. [Layer 1 — Python AST analysis](#2-layer-1--python-ast-analysis)
3. [Layer 2 — Variant generation (AST rewriting)](#3-layer-2--variant-generation-ast-rewriting)
4. [Layer 3 — JIT integration: 4 ASTs become 4 binaries](#4-layer-3--jit-integration-4-asts-become-4-binaries)
5. [Layer 4 — C launcher: per-tile dispatch via `launch_quad`](#5-layer-4--c-launcher-per-tile-dispatch-via-launch_quad)
6. [Layer 5 — OpenMP parallelism inside the launcher](#6-layer-5--openmp-parallelism-inside-the-launcher)
7. [Why these layers (architectural rationale)](#7-why-these-layers-architectural-rationale)
8. [Timing methodology: the wrapper-overhead trap](#8-timing-methodology-the-wrapper-overhead-trap)
9. [Where each piece lives (file:line reference index)](#9-where-each-piece-lives-fileline-reference-index)

---

## 1. The end-to-end picture in 20 lines

```
user writes ONE @triton.jit kernel
   │
   ▼
Python AST  ──── inspect.getsource + ast.parse
   │
   ▼
analyze_cpu_tail_2d           ←── recognises the matmul pattern in AST
   │       (find pid_m, pid_n, offs_m, offs_n, M, N, BLOCK_M, BLOCK_N)
   ▼
make_cpu_tail_2d_variant × 4  ←── deep-copies AST, rewrites mask=... per variant
   │   ┌── "main":    drop M-clause AND N-clause
   │   ├── "m_tail":  keep M-clause, drop N-clause
   │   ├── "n_tail":  drop M-clause, keep N-clause
   │   └── "corner":  keep both (original)
   ▼
self.compile × 4              ←── 4 independent Triton compilation pipelines
   │   each AST → TTIR → TTCIR → LLIR → .so binary
   ▼
QuadPathCompiledKernel        ←── holds 4 function pointers + arg indices
   │       for M, N, BLOCK_M, BLOCK_N
   ▼
kernel[grid](args)            ←── Python: ONE call, packs args
   │
   ▼
launch_quad (C function)      ←── parses args, computes full_m, full_n
   │
   ▼
OpenMP parallel for           ←── threads share grid; each tile selects
   │       one of 4 function pointers via cmov chain
   ▼
4 different binaries actually execute concurrently
   on different (pid_m, pid_n) coordinates
```

Five layers, each with a single clearly-scoped job. The rest of this doc
walks each one with code.

---

## 2. Layer 1 — Python AST analysis

### 2.1 Source acquisition

`@triton.jit` decorates a Python function. The `JITFunction` object can
recover its source via `inspect.getsource` and `ast.parse` it. The entry
point we care about is in `python/triton/runtime/jit.py`:

```python
# jit.py:898-908
src = self.ASTSource(self, signature, constexprs, attrs)
dual_tail_analysis = None
quad_tail_analysis = None
if target.backend == "cpu" and getattr(options, "enable_tail_guard", False):
    from triton.compiler.cpu_tail import analyze_cpu_tail, analyze_cpu_tail_2d
    quad_tail_analysis = analyze_cpu_tail_2d(self, src)
    if quad_tail_analysis.diagnostic.matched:
        dual_tail_analysis = None  # 2D takes priority
    else:
        quad_tail_analysis = None
        dual_tail_analysis = analyze_cpu_tail(self, src)
```

Gating:

| Condition | Where |
|---|---|
| Backend must be CPU | `target.backend == "cpu"` |
| Feature must be on | `enable_tail_guard` option (set by `TRITON_CPU_TAIL_GUARD=1`, default on, see jit.py:748) |
| 2-D matmul pattern preferred over 1-D | `quad_tail_analysis.diagnostic.matched` takes priority |

`analyze_cpu_tail_2d` is the function this whole report orbits around.

### 2.2 What `analyze_cpu_tail_2d` is looking for

Source: `python/triton/compiler/cpu_tail.py:435-493`.

The matmul pattern it must find in the AST:

```python
# user code (the only thing we know how to handle)
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
# ... some loads/stores with masks like:
#     mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
# ... a K loop:
#     for k in range(0, K, BLOCK_K):  # BLOCK_K constexpr
```

Six things it records, populated into `TailGuard2DDiagnostic`
(`cpu_tail.py:288-302`):

| Field | What it is | How it is found |
|---|---|---|
| `m_n_name` | The argument name carrying `M` | `_find_bound_for_offsets`: any `Compare` `offs_m[...] < <name>` |
| `n_n_name` | The argument name carrying `N` | same, for `offs_n` |
| `m_block_name` | The constexpr arg `BLOCK_M` | discovered while matching `offs_m` def |
| `n_block_name` | The constexpr arg `BLOCK_N` | same |
| `offs_m_name` | The variable holding the M-offsets | `_find_offsets` (matches `pid_m * BLOCK_M + tl.arange(...)` shape) |
| `offs_n_name` | same for N | same |

Step-by-step (the analyzer just bails the moment any check fails):

```python
# cpu_tail.py:443-446
pid_m = _find_program_id_name_axis(fn_def, 0)   # axis 0 → pid_m
if pid_m is None:
    return TailGuard2DAnalysis(..., reason="missing pid_m = tl.program_id(0)")

# cpu_tail.py:447-449
pid_n = _find_program_id_name_axis(fn_def, 1)   # axis 1 → pid_n

# cpu_tail.py:451-455
offs_m_result = _find_offsets(fn_def, pid_m)    # must match shape
                                                #   tgt = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_m_name, m_block_name = offs_m_result

# cpu_tail.py:456-461  same for offs_n

# cpu_tail.py:463-467
constexprs = _constexpr_names(fn, src)
if m_block_name not in constexprs:
    return TailGuard2DAnalysis(..., reason=f"{m_block_name} is not constexpr")

# cpu_tail.py:469-477
m_n_name = _find_bound_for_offsets(fn_def, offs_m_name)
# scans entire function for any  offs_m[...] < <name>
# (must be a plain Name, not an expression — keeps the analyzer simple)

n_n_name = _find_bound_for_offsets(fn_def, offs_n_name)
```

Failure modes are explicit: every early `return` carries a `reason`
string. If you ever see `diagnostic.matched == False`, the `reason`
field tells you what the analyzer missed.

### 2.3 The "scaled pid" trick: tolerating `tl.program_id() * 2` etc.

Real Triton matmul kernels often introduce a swizzle:

```python
pid = tl.program_id(0)
GROUP_SIZE_M = 8
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
pid_m = group_id * GROUP_SIZE_M + (pid % num_pid_in_group) % GROUP_SIZE_M
pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
```

We don't even try to recognise swizzled forms (they break the launch-time
`gridX == ceil(M/BM)` assumption that `launch_quad` relies on). The
straight-line check `pid_m = tl.program_id(0)` is intentionally strict —
the wrong outcome here is just "no specialisation," which is always
safe.

### 2.4 K loop detection (optional)

`_find_k_loop` (`cpu_tail.py:412-433`) records the K-loop variable name
if present. This is **only stored for diagnostics** in 2-D mode; we do
*not* peel the K loop. See [§3.3 K-mask preservation](#33-k-mask-preservation-do-not-touch-the-k-mask).

---

## 3. Layer 2 — Variant generation (AST rewriting)

If analysis matched, `make_cpu_tail_2d_variant` is called four times,
once per variant. Source: `cpu_tail.py:495-519`.

### 3.1 Top-level shape

```python
# cpu_tail.py:495-519
def make_cpu_tail_2d_variant(variant: str, fn, src):
    analysis = analyze_cpu_tail_2d(fn, src)
    if not analysis.diagnostic.matched:
        return analysis.tree, analysis.diagnostic    # nothing to do

    fn_def = copy.deepcopy(analysis.fn_def)          # ★ fresh AST per variant

    keep_m = variant in ("m_tail", "corner")
    keep_n = variant in ("n_tail", "corner")

    fn_def = _SimplifyMask2D(diag.offs_m_name, diag.offs_n_name,
                             keep_m, keep_n).visit(fn_def)

    ast.fix_missing_locations(fn_def)
    new_tree = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(new_tree)
    return new_tree, diag
```

Two things to notice:

1. **`copy.deepcopy(analysis.fn_def)`**: the analyzer holds the *original*
   AST. Each variant gets a fresh clone before rewriting, so they don't
   stomp on each other.
2. **`keep_m` / `keep_n` matrix**:

   | variant | `keep_m` | `keep_n` | meaning |
   |---|---|---|---|
   | `main`   | False | False | strip both M and N mask clauses |
   | `m_tail` | True  | False | bottom-row tiles: keep M, strip N |
   | `n_tail` | False | True  | right-column tiles: strip M, keep N |
   | `corner` | True  | True  | bottom-right corner: keep everything |

The "K mask" clause (`offs_k + k < K`) doesn't reference `offs_m` or
`offs_n`, so it survives all four rewrites by construction. That's not
an accident — it's the load-bearing invariant; see §3.3.

### 3.2 The mask simplifier itself

The interesting code is `_simplify_mask_expr` (`cpu_tail.py:347-371`) and
its driver `_SimplifyMask2D` (`cpu_tail.py:373-394`):

```python
def _simplify_mask_expr(expr, offs_m, offs_n, keep_m, keep_n):
    """Recursively simplify an AND-tree mask.  Return None → remove entirely."""

    # AND nodes: simplify both sides, then assemble
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitAnd):
        left  = _simplify_mask_expr(expr.left,  offs_m, offs_n, keep_m, keep_n)
        right = _simplify_mask_expr(expr.right, offs_m, offs_n, keep_m, keep_n)
        if left is None and right is None: return None   # both gone → drop whole AND
        if left is None:  return right                   # only right survives
        if right is None: return left
        result = copy.copy(expr)
        result.left = left
        result.right = right
        return result

    # Leaf Compare: decide whether to keep this clause
    if isinstance(expr, ast.Compare):
        refs_m = any(isinstance(n, ast.Name) and n.id == offs_m for n in ast.walk(expr))
        refs_n = any(isinstance(n, ast.Name) and n.id == offs_n for n in ast.walk(expr))
        if refs_m and not keep_m: return None
        if refs_n and not keep_n: return None
    return expr


class _SimplifyMask2D(ast.NodeTransformer):
    def visit_Call(self, node):                          # tl.load / tl.store
        node = self.generic_visit(node)
        if _attr_name(node.func) not in ("load", "store"):
            return node
        mask_kw = next((kw for kw in node.keywords if kw.arg == "mask"), None)
        if mask_kw is None:
            return node
        new_mask = _simplify_mask_expr(mask_kw.value, self.offs_m, self.offs_n,
                                       self.keep_m, self.keep_n)
        if new_mask is None:
            # whole mask collapsed → strip mask= AND other= (other only matters when masked)
            node.keywords = [kw for kw in node.keywords if kw.arg not in ("mask", "other")]
        else:
            mask_kw.value = new_mask
        return node
```

### 3.3 K-mask preservation: do not touch the K mask

Concrete example. A typical matmul load:

```python
a = tl.load(a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0)
```

What `_SimplifyMask2D` produces per variant:

| variant | resulting `mask=` kwarg |
|---|---|
| `main`   | `mask=(offs_k[None, :] + k < K)` (M-clause stripped, K-clause kept) |
| `m_tail` | `mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K)` (unchanged) |
| `n_tail` | `mask=(offs_k[None, :] + k < K)` (same as `main`: this load has no N-clause anyway) |
| `corner` | `mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K)` (unchanged) |

For a `b = tl.load(b_ptrs, mask=(offs_k[:, None]+k<K) & (offs_n[None,:]<N))`,
it's the opposite: `n_tail` strips, `m_tail` keeps everything.

**The K mask is intentionally never stripped, even though it looks
removable for aligned K.** Doing so causes a ~37% regression for fully
aligned K. The LLVM evidence chain is in
[`CPU_DISPATCH_README.md` §2](CPU_DISPATCH_README.md#2-key-insight-k-mask-must-stay)
(with assembly diffs). Short version: LLVM uses `offs_k + k < K` as a
vectorization hint; removing it causes 1024 `vbroadcastss` memory
broadcasts instead of 127 register-to-register broadcasts.

This single design decision is encoded in `_simplify_mask_expr` by *not
checking* whether a clause references `offs_k`. Only M and N clauses get
touched.

### 3.4 What happens if both clauses are stripped

The `_SimplifyMask2D.visit_Call` branch:

```python
if new_mask is None:
    node.keywords = [kw for kw in node.keywords
                     if kw.arg not in ("mask", "other")]
```

removes the `mask=` AND `other=` kwargs entirely, turning

```python
tl.load(ptrs, mask=..., other=0.0)
```

into

```python
tl.load(ptrs)
```

which Triton compiles to a plain `vector.load` — no `vmaskmovps`, no
zero-blend, no compare-and-broadcast.

### 3.5 Why deep copy

`ast.NodeTransformer` mutates the tree in place. Without
`copy.deepcopy(analysis.fn_def)` in `make_cpu_tail_2d_variant`, the
second `_mk2d("m_tail")` call would see the already-stripped AST from
`_mk2d("main")` and produce wrong code. The fresh-copy-per-variant
pattern keeps every transformation idempotent on its input.

---

## 4. Layer 3 — JIT integration: 4 ASTs become 4 binaries

`python/triton/runtime/jit.py:967-983` is the synchronous compile path
(there is an async sibling at 942-956 with the same shape):

```python
if quad_tail_analysis is not None:
    def _mk2d(v):
        return self.compile(
            self.ASTSource(self, signature, constexprs, attrs,
                          cpu_tail_2d_variant=v),
            target=target, options=options.__dict__)
    corner_kernel = self.compile(src, target=target, options=options.__dict__)
    kernel = self.QuadPathCompiledKernel(
        _mk2d("main"), _mk2d("m_tail"), _mk2d("n_tail"),
        corner_kernel,
        quad_tail_analysis.diagnostic)
```

### 4.1 `ASTSource` with the variant tag

Each of the four `_mk2d(v)` calls constructs a fresh `ASTSource` with
`cpu_tail_2d_variant=v` set. This value flows into the AST source's
`parse()` method, which downstream invokes
`make_cpu_tail_2d_variant(v, self, src)` and returns the rewritten AST
in place of the original.

Effect: each of the four `self.compile(...)` calls sees a different AST
even though `signature`, `constexprs`, and `attrs` are identical. The
returned `CompiledKernel` objects all have the same ABI (same
signature, same constants) but distinct `.so` bytes.

The fourth one (`corner_kernel`) deliberately uses the *original*
`src`, not a `_mk2d("corner")` rewrite. Both should produce the same
AST (since `keep_m=keep_n=True` means no simplification), but skipping
the rewrite saves a deep-copy and an AST traversal.

### 4.2 `QuadPathCompiledKernel`

`python/triton/compiler/compiler.py:633-680`:

```python
class QuadPathCompiledKernel:
    """2-D tail guard: main / m_tail / n_tail / corner compiled variants."""

    def __init__(self, main_kernel, m_tail_kernel, n_tail_kernel,
                 corner_kernel, diagnostic):
        self.main_kernel   = main_kernel
        self.m_tail_kernel = m_tail_kernel
        self.n_tail_kernel = n_tail_kernel
        self.corner_kernel = corner_kernel
        ...
        if not isinstance(self.src, ASTSource):
            self._m_n_idx = self._m_block_idx = self._n_n_idx = self._n_block_idx = -1
        else:
            names = self.src.fn.arg_names
            self._m_n_idx     = names.index(diagnostic.m_n_name)     # index of M in arg list
            self._m_block_idx = names.index(diagnostic.m_block_name)
            self._n_n_idx     = names.index(diagnostic.n_n_name)
            self._n_block_idx = names.index(diagnostic.n_block_name)
```

The four argument indices are computed *once at compile time* and
embedded in the kernel object. At launch time the C launcher can pluck
M, N, BLOCK_M, BLOCK_N out of the args tuple without re-parsing.

### 4.3 Dispatch entry: `.run()`

`compiler.py:661-678`:

```python
def run(self, grid_0, grid_1, grid_2, stream, function, packed_metadata,
        launch_metadata, launch_enter_hook, launch_exit_hook, *args):
    if self._m_n_idx < 0:
        # No diagnostic (e.g. non-ASTSource); fall back to corner only.
        return self.corner_kernel.run(...)
    self.main_kernel._init_handles()
    self.m_tail_kernel._init_handles()
    self.n_tail_kernel._init_handles()
    self.corner_kernel._init_handles()
    self.corner_kernel._run.launch_quad(
        grid_0, grid_1, grid_2, stream,
        self.main_kernel.function,   self.main_kernel.packed_metadata,
        self.m_tail_kernel.function, self.m_tail_kernel.packed_metadata,
        self.n_tail_kernel.function, self.n_tail_kernel.packed_metadata,
        self.corner_kernel.function, self.corner_kernel.packed_metadata,
        launch_metadata, launch_enter_hook, launch_exit_hook,
        self._m_n_idx, self._m_block_idx, self._n_n_idx, self._n_block_idx,
        *args)
```

One Python call → one C call → done. Notice there is **no per-tile
Python work** anywhere in the run path. Python's role ends at the
moment of `launch_quad(...)`.

---

## 5. Layer 4 — C launcher: per-tile dispatch via `launch_quad`

The C source for `launch_quad` is generated from a Python template in
`third_party/cpu/backend/driver.py:829-952`. The compiled-once template
gets re-emitted per (constants, signature, ids) tuple — that is, a
unique kernel-signature shape produces its own C launcher.

### 5.1 Arg parsing

```c
// driver.py:846 (template)
PyArg_ParseTuple(args, "iiiOKOKOKOKOOOOiiii{args_format}",
                 &gridX, &gridY, &gridZ,
                 &py_obj_stream,
                 &pMainKrnl,   &main_kernel_metadata,
                 &pMTailKrnl,  &m_tail_kernel_metadata,
                 &pNTailKrnl,  &n_tail_kernel_metadata,
                 &pCornerKrnl, &corner_kernel_metadata,
                 &launch_metadata, &launch_enter_hook, &launch_exit_hook,
                 &m_n_arg_idx, &m_block_arg_idx,
                 &n_n_arg_idx, &n_block_arg_idx,
                 ... kernel-specific args ...);
```

The format string layout is:

| Chars | Meaning |
|---|---|
| `iii`     | gridX, gridY, gridZ |
| `O`       | stream object |
| `KO` × 4  | 4 × (kernel function pointer, metadata object) |
| `OOO`     | launch_metadata, launch_enter_hook, launch_exit_hook |
| `iiii`    | 4 × arg index for M, BM, N, BN |
| `{args_format}` | kernel's own runtime args |

### 5.2 Reading M / N / BLOCK_M / BLOCK_N from args

`driver.py:880-902`:

```c
// Generic int-arg reader; works on plain ints AND on constexpr-from-Python ints.
auto get_int_arg_q = [&](int idx, int64_t &out) -> bool {
    switch (idx) {
    case 0: ...
    case 1: ...
    // ... auto-generated cases per arg
    }
};

bool use_quad = false;
int64_t m_val = 0, m_block_val = 0, n_val = 0, n_block_val = 0;
uint32_t full_m = 0, full_n = 0;
if (get_int_arg_q(m_n_arg_idx, m_val) &&
    get_int_arg_q(m_block_arg_idx, m_block_val) &&
    get_int_arg_q(n_n_arg_idx, n_val) &&
    get_int_arg_q(n_block_arg_idx, n_block_val) &&
    m_block_val > 0 && n_block_val > 0 && m_val >= 0 && n_val >= 0) {

    int64_t exp_gx = (m_val + m_block_val - 1) / m_block_val;   // ceil(M/BM)
    int64_t exp_gy = (n_val + n_block_val - 1) / n_block_val;   // ceil(N/BN)
    if (gridX == exp_gx && gridY == exp_gy && gridZ == 1) {
        full_m = (uint32_t)(m_val / m_block_val);   // floor(M/BM) = first tail row
        full_n = (uint32_t)(n_val / n_block_val);   // floor(N/BN) = first tail col
        use_quad = true;
    }
}
```

Two sanity checks before we can quad-dispatch:

1. **Bounds must be readable as ints.** If for any reason the kernel
   doesn't carry M, N, BLOCK_M, BLOCK_N at the expected indices (or
   their types differ), `use_quad` stays false and we fall through to
   single-kernel mode.
2. **The grid shape must equal `(ceil(M/BM), ceil(N/BN), 1)`.** This is
   how we detect "vanilla matmul launch." If the user hand-tiles the
   grid (e.g. they launch `(ceil(M/BM) - 1,)` to skip the bottom row),
   we don't know which tile is which, so we fall back.

`full_m = floor(M/BM)` is the index of the bottom-row tile (the one
that handles the partial M chunk). Similarly `full_n` for the right
column.

### 5.3 The dispatch loop

`driver.py:904-938`:

```c
if (use_quad) {
    int max_threads_q = ...;
    if (max_threads_q == 1) {
        // serial path — same body, no OMP overhead
        for (uint32_t y = 0; y < gridY; ++y) {
            for (uint32_t x = 0; x < gridX; ++x) {
                bool xt = (x == full_m), yt = (y == full_n);
                kernel_ptr_t kptr = (!xt && !yt) ? main_kernel_ptr
                                  : ( xt && !yt) ? m_tail_kernel_ptr
                                  : (!xt &&  yt) ? n_tail_kernel_ptr
                                  :                corner_kernel_ptr;
                (*kptr)(args..., x, y, 0, gridX, gridY, 1);
            }
        }
    } else {
        uint32_t total = gridX * gridY;
        #pragma omp parallel for schedule(static) num_threads(max_threads_q)
        for (uint32_t i = 0; i < total; ++i) {
            uint32_t x = i % gridX;
            uint32_t y = i / gridX;
            bool xt = (x == full_m), yt = (y == full_n);
            kernel_ptr_t kptr = (!xt && !yt) ? main_kernel_ptr
                              : ( xt && !yt) ? m_tail_kernel_ptr
                              : (!xt &&  yt) ? n_tail_kernel_ptr
                              :                corner_kernel_ptr;
            (*kptr)(args..., x, y, 0, gridX, gridY, 1);
        }
    }
} else {
    // Fall back to running corner_kernel for every tile.
    run_omp_kernels(0, gridX, gridY, gridZ, num_threads, corner_kernel_ptr, ...);
}
```

The 4-way chained ternary

```c
kernel_ptr_t kptr = (!xt && !yt) ? main
                  : ( xt && !yt) ? m_tail
                  : (!xt &&  yt) ? n_tail
                  :                corner;
```

compiles to a `cmov` chain (or one branch with a jump-table on some
targets) — **two comparisons, one conditional move per tile**. For a
matmul that does BLOCK_M × BLOCK_N × BLOCK_K FMAs per tile, that's
truly negligible.

### 5.4 The branchless-cmov shape, visualized

For a 5×5 grid where `full_m == 4, full_n == 4`:

```
y=0  M M M M N        legend
y=1  M M M M N          M  = main_kernel_ptr
y=2  M M M M N          N  = n_tail_kernel_ptr (right column except corner)
y=3  M M M M N          T  = m_tail_kernel_ptr (bottom row except corner)
y=4  T T T T C          C  = corner_kernel_ptr
     x=0 ... x=4
```

16 tiles hit `main` (no masks), 4 hit `m_tail`, 4 hit `n_tail`, 1 hits
`corner`. For larger grids (e.g. 32×32), the `main` fraction approaches
100% and the four-variant trick approaches a free 100% mask removal on
the hot path.

---

## 6. Layer 5 — OpenMP parallelism inside the launcher

### 6.1 Thread count source

```c
// driver.py:867-871
int num_threads = 0;
PyObject *num_threads_attr =
    PyObject_GetAttrString(corner_kernel_metadata, "num_cpu_threads");
if (num_threads_attr && PyLong_Check(num_threads_attr))
    num_threads = PyLong_AsLong(num_threads_attr);
```

Thread count comes from the kernel's compile-time metadata. If
unspecified, the launcher falls back to `omp_get_max_threads()`
(typically the number of logical CPUs). Users can override with
`OMP_NUM_THREADS=...` in the environment.

### 6.2 `schedule(static)` for tail guard

```c
#pragma omp parallel for schedule(static) num_threads(max_threads_q)
```

`schedule(static)` divides the `total` iterations into equal contiguous
chunks per thread. This is the right choice for tail guard because
**every tile's workload is essentially identical** — the four variants
do almost the same work (same FMA count, just differ on a few SIMD
instructions per BLOCK_K iteration). No load imbalance to chase.

Compare with `launch_region_2d` (`driver.py:439`), which uses
`schedule(dynamic, 1)` because tile counts vary per row (e.g.
lower-triangular causal-attention dispatch has 1 tile in row 0 and T
tiles in row T-1).

### 6.3 Single-thread fast path

```c
// driver.py:914-922
if (max_threads_q == 1) {
    // Serial loop, identical body, no OMP fork/join overhead.
    for (...) { ... }
    return;
}
```

OpenMP `parallel for` has a fixed setup cost (~µs) even at 1 thread for
the parallel region itself. For the kernels under test, that's not
free. Hence the explicit non-OMP serial path.

### 6.4 What "different IR runs on different threads" actually means

OpenMP gives each thread a contiguous chunk of `[0, total)` flat tile
indices. Inside the loop body, each thread computes `(x, y)` from `i`
and decides which of the 4 function pointers to invoke. Different
threads may dispatch to different variants on the same iteration
(thread A is in the main region, thread B happens to be on the bottom
edge tile).

End result: at any instant during execution, **the 4 binaries can be
running concurrently on different CPU cores**, each operating on a
disjoint range of `(pid_m, pid_n)`. No locks, no shared mutable state,
no contention.

---

## 7. Why these layers (architectural rationale)

Why split into AST → C launcher → OpenMP, rather than doing everything
in MLIR?

| Question | Answer |
|---|---|
| Why AST rewriting and not an MLIR pass? | The 4 variants need to be **independent compilation units** — different `.so` binaries with potentially different mask predicates. MLIR rewrites a single function in place; producing 4 sibling functions and having the C launcher pick one of them requires infrastructure that doesn't fit neatly in a single pass. AST cloning before compile is dramatically simpler. |
| Why per-tile dispatch in C, not Python? | A 1000×1000 matmul with 64×64 tiles is `15 × 15 = 225` tiles. Python dispatch overhead per call (~50–200µs) would dominate the per-tile work (~µs). C dispatch is `cmov`-chain cost — invisible. |
| Why C and not Cython/Pybind? | The launcher is one template (`driver.py:make_launcher`) that gets specialised per kernel signature anyway. Code-generating C, compiling it once, and calling it via `mod.launch_quad` is the simplest reliable path that gives us OpenMP integration. |
| Why OpenMP and not pthreads or std::thread? | OMP gives us schedule strategies (static / dynamic) and thread-pool reuse for free. The cost is one `#pragma omp parallel for` directive vs ~30 lines of pthread setup. |

This is the same engineering question loop peeling resolves at a
*different* layer; see the loop-peel deep dive for the parallel
argument going the other way.

---

## 8. Timing methodology: the wrapper-overhead trap

Before any tail-guard speedup claim should be believed, the timing
methodology has to be right. There are two pitfalls — both bit us
during development.

### 8.1 The trap: timing the Python wrapper, not the kernel

The naive bench:

```python
import time
for _ in range(20):
    matmul[grid](A, B, C, M, N, K)
elapsed = time.perf_counter() - t0
```

`matmul[grid](...)` is a Python operation. It:

1. Builds a `runner` via `__getitem__`
2. Resolves constexprs
3. Serializes the signature (looks up which `.so` matches)
4. Validates arg types
5. Packs args into the C launcher's expected format
6. **Finally** calls into `launch_quad`

Steps 1–5 cost roughly **50–200µs per call** on this machine. For a
matmul kernel that takes ~100µs total, that means roughly half of every
wall-clock measurement is Python overhead that the tail-guard change
does *not* affect.

Consequence: if tail guard makes the kernel itself 25% faster, the
naive bench shows ~10% speedup. Worse — at large N where the kernel
work is small relative to wrapper noise (e.g. 5µs kernel, 5–200µs
wrapper jitter), the comparison can flip sign and report **slowdown**
when the kernel is actually faster.

Concrete example we observed (loop-peel bench, same effect applies to
tail guard):

```
                       without hooks (Python wrapper included)
N=1048573    disabled 450µs    enabled 677µs    "0.67x" ← garbage
                       with hooks (kernel-only)
N=1048573    disabled ~330µs   enabled ~310µs    1.04x  ← real
```

### 8.2 The fix: hook-based timing

`triton.testing.do_bench` has a `measure_time_with_hooks=True` mode
that uses the CPU driver's `launch_enter_hook` / `launch_exit_hook` to
bracket *only the kernel execution*:

```python
ms = triton.testing.do_bench(
    lambda: matmul[grid](A, B, C, M, N, K),
    warmup=50, rep=200,
    measure_time_with_hooks=True,
    return_mode="min",   # filters out cold/throttled iterations
)
```

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

The hooks fire at the C launcher's first and last instruction —
specifically, in `launch_quad`:

```c
// driver.py:870-875
if (launch_enter_hook != Py_None) {                   // ← _enter_hook
    PyObject_CallObject(launch_enter_hook, ...);
}
// ... grid loop runs all the tiles ...
if (launch_exit_hook != Py_None) {                    // ← _exit_hook
    PyObject_CallObject(launch_exit_hook, ...);
}
```

So the measured interval includes: per-tile dispatch decisions,
OpenMP `parallel for` setup/teardown, and all kernel execution. It
**excludes**: Python `__getitem__`, signature serialize, constexpr
resolve, arg pack.

The hook itself costs a tiny Python-callback round-trip (~0.5–1µs),
but it's paid **identically** by every variant being compared, so it
cancels in the ratio.

### 8.3 The other trap: averaging across iterations

Even with correct timing brackets, the naive

```python
t0 = time.perf_counter()
for _ in range(N): fn()       # ★ single measurement covering N iters
mean_us = (time.perf_counter() - t0) / N * 1e6
```

is fragile: one OS preemption / page fault during the N iterations
sticks in the mean and stays there.

Better: time each iteration separately, then take median or min:

```python
samples = []
for _ in range(N):
    t0 = time.perf_counter()
    fn()
    samples.append(time.perf_counter() - t0)
return min(samples)   # or statistics.median(samples)
```

Even better: interleave variants in the same loop, so all variants
experience the same thermal/cache/scheduler state per round.
`_bench_tail_guard_3way.py:83-94` does exactly this:

```python
def bench3(fn_base, fn_man, fn_quad):
    for _ in range(N_WARMUP):
        fn_base(); fn_man(); fn_quad()
    bt, mt, qt = [], [], []
    for _ in range(N_ROUNDS):
        t0 = time.perf_counter(); fn_base(); t1 = time.perf_counter()
        t2 = time.perf_counter(); fn_man();  t3 = time.perf_counter()
        t4 = time.perf_counter(); fn_quad(); t5 = time.perf_counter()
        bt.append((t1 - t0) * 1e6)
        mt.append((t3 - t2) * 1e6)
        qt.append((t5 - t4) * 1e6)
    return bt, mt, qt
```

Used together with `do_bench(measure_time_with_hooks=True)`, this
removes both Python-wrapper dilution and outlier contamination.

### 8.4 Why does this matter so much for tail guard

Tail guard's per-tile saving is small in absolute terms (a few SIMD
instructions per BLOCK_K iteration of the K loop, multiplied across
~16 main-region tiles for a small matmul). The total speedup is in
the 2–10% range for matmul sizes around 300–512. That's exactly the
range where wrapper noise can flood the signal — so accurate timing
isn't optional. The headline numbers in `CPU_DISPATCH_README.md` §5.5
were measured with `bench3` + per-iteration `time.perf_counter`
brackets.

---

## 9. Where each piece lives (file:line reference index)

### Python layer (AST + JIT)

| Function / class | File | Lines | Role |
|---|---|---|---|
| `analyze_cpu_tail_2d` | `python/triton/compiler/cpu_tail.py` | 435–493 | Walk AST, find pattern, populate diagnostic |
| `_find_program_id_name_axis` | same | 314–326 | Detect `pid_m = tl.program_id(0)` |
| `_find_offsets` | same | 120–131 | Detect `offs_m = pid_m * BLOCK_M + tl.arange(...)` |
| `_find_bound_for_offsets` | same | 329–344 | Detect `offs_m[...] < M` |
| `_find_k_loop` | same | 412–433 | Detect `for k in range(0, K, BLOCK_K)` |
| `_constexpr_names` | same | 215–223 | Pull constexpr arg names from signature |
| `_simplify_mask_expr` | same | 347–371 | Recursively prune AND-tree mask clauses |
| `_SimplifyMask2D` | same | 373–394 | NodeTransformer that drives the simplifier |
| `make_cpu_tail_2d_variant` | same | 495–519 | Deep-copy + apply simplifier → one variant AST |
| `TailGuard2DDiagnostic` | same | 288–302 | Dataclass of names + arg indices |
| `QuadPathCompiledKernel` | `python/triton/compiler/compiler.py` | 633–680 | Stores 4 binaries + arg indices; dispatch entry |
| JIT integration (sync) | `python/triton/runtime/jit.py` | 967–983 | `_mk2d(v)` for each variant; build `QuadPathCompiledKernel` |
| JIT integration (async) | same | 942–956 | Same logic for async compile path |
| Env var → option | same | 748 | `TRITON_CPU_TAIL_GUARD` → `options.enable_tail_guard` |

### C launcher layer

| Function | File | Lines | Role |
|---|---|---|---|
| `make_launcher` (template) | `third_party/cpu/backend/driver.py` | 120–502 | Generates launcher C from signature |
| `launch_quad` (C template) | same | 829–952 | Per-tile dispatch entry point |
| `run_omp_kernels` (fallback) | same | 272–305 | Used when `use_quad == false` |
| `enable_hook_timing` | same | 1041–1053 | Sets up enter/exit hooks for timing |

### Tests + benches

| File | Role |
|---|---|
| `python/test/unit/cpu/test_tail_guard.py` | Correctness for 4-variant matmul |
| `python/test/unit/cpu/_bench_matmul_tail.py` | Baseline / quad comparison |
| `python/test/unit/cpu/_bench_tail_guard_3way.py` | Interleaved 3-way bench, the timing-correct one |
| `python/test/unit/cpu/_bench_tail_guard_kmask.py` | K-mask isolation evidence |

### Cross-references

- High-level "what / why" + benchmarks: [`CPU_DISPATCH_README.md`](CPU_DISPATCH_README.md)
- Repo overview / portfolio README: [`README.md`](README.md)
- Sibling optimization at the IR layer: see the
  `mlir-loop-peel-clean` branch's `LOOP_PEEL_INTERNALS.md`
