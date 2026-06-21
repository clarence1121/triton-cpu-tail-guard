"""Minimal ergonomic API for the CPU launcher graph PoC.

Wraps the proven trampoline pattern (see python/test/unit/cpu/_bench_cpu_graph_*.py)
into a small class so users don't have to write C source by hand.

Status: PoC. Single-kernel chain only. Tail-guard not supported (use
TRITON_CPU_TAIL_GUARD=0). See CPU_GRAPH_INTERNALS.md for the rationale.

Example
-------
>>> import torch, triton, triton.language as tl
>>> from triton.cpu_graph import CPUGraph
>>>
>>> @triton.jit
... def axpy(X, Y, OUT, N, BLOCK: tl.constexpr):
...     pid = tl.program_id(0)
...     offs = pid * BLOCK + tl.arange(0, BLOCK)
...     tl.store(OUT + offs, tl.load(X + offs) + tl.load(Y + offs))
>>>
>>> N, BLOCK = 1024, 1024
>>> x = torch.randn(N); y = torch.randn(N); out = torch.zeros(N)
>>> axpy[(1,)](x, y, out, N, BLOCK=BLOCK)        # compile
>>> graph = CPUGraph.from_kernel(axpy, args=(x, y, out, N), grid=(1,),
...                              constexprs={'BLOCK': BLOCK})
>>> graph.replay(n=100)                          # 100 chained calls in C
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Sequence


_PTR_TYPE_MAP = {
    "*fp32": "float*",
    "*fp64": "double*",
    "*i32":  "int32_t*",
    "*i64":  "int64_t*",
}

_SCALAR_TYPE_MAP = {
    "i32":   "int32_t",
    "i64":   "int64_t",
    "fp32":  "float",
    "fp64":  "double",
}


def _c_type_for(arg_ty: str) -> str:
    if arg_ty in _PTR_TYPE_MAP:
        return _PTR_TYPE_MAP[arg_ty]
    if arg_ty in _SCALAR_TYPE_MAP:
        return _SCALAR_TYPE_MAP[arg_ty]
    raise ValueError(f"unsupported arg type for graph: {arg_ty}")


@dataclass
class _ChainConfig:
    fn_ptr: int                # kernel function pointer
    arg_types: list[str]       # ordered Triton arg types (no constexpr)
    grid_x: int                # grid[0]


class CPUGraph:
    """A 'graph' of repeated calls to the same kernel.

    Generates and compiles a C trampoline that calls the kernel's raw
    function pointer N times in a loop, bypassing Triton's Python wrapper.

    Single-thread sequential by default; pass `omp=True` to wrap the grid
    loop in an OMP parallel for (single-precision, schedule(static)).
    """

    def __init__(self, cfg: _ChainConfig, args: Sequence, omp: bool = False,
                 num_threads: int = 1):
        self._cfg = cfg
        self._args = list(args)
        self._omp = omp
        self._num_threads = num_threads
        self._lib = self._compile_trampoline()

    @classmethod
    def from_kernel(cls, kernel, args, grid, constexprs=None, *,
                    omp: bool = False, num_threads: int = 1):
        """Build a graph from a Triton @triton.jit kernel.

        `args` must be the non-constexpr positional args (tensors and scalars).
        `constexprs` is a dict {name: value} of constexpr-marked params.

        Calls kernel.warmup() to compile and extract the function pointer.
        """
        kwargs = dict(constexprs or {})
        kwargs["grid"] = grid
        compiled = kernel.warmup(*args, **kwargs)
        fn_ptr = compiled.function
        if fn_ptr is None:
            # tail-guard wrapper
            for attr in ("main_kernel", "corner_kernel"):
                sub = getattr(compiled, attr, None)
                if sub is not None:
                    sub._init_handles()
                    if sub.function is not None:
                        fn_ptr = sub.function
                        break
        if fn_ptr is None:
            raise RuntimeError(
                "compiled.function is None — disable TRITON_CPU_TAIL_GUARD or "
                "extract the appropriate variant manually")

        # Collect non-constexpr arg types in declaration order
        sig = compiled.src.signature  # {name: type-str}
        arg_types = [ty for name, ty in sig.items()
                     if ty != "constexpr" and name not in (constexprs or {})]
        cfg = _ChainConfig(fn_ptr=fn_ptr, arg_types=arg_types, grid_x=grid[0])
        return cls(cfg, args, omp=omp, num_threads=num_threads)

    def _compile_trampoline(self):
        # Build C signature matching the JIT'd kernel function
        kernel_arg_decls = ", ".join(
            f"{_c_type_for(t)} a{i}" for i, t in enumerate(self._cfg.arg_types)
        )
        kernel_arg_names = ", ".join(f"a{i}" for i in range(len(self._cfg.arg_types)))

        # Trampoline takes pointer-or-scalar args matching the kernel signature
        tramp_arg_decls = ", ".join(
            f"void* a{i}" if t.startswith("*")
            else f"{_c_type_for(t)} a{i}"
            for i, t in enumerate(self._cfg.arg_types)
        )
        tramp_call_args = ", ".join(
            f"({_c_type_for(t)})a{i}" if t.startswith("*") else f"a{i}"
            for i, t in enumerate(self._cfg.arg_types)
        )

        omp_pragma = (
            f"#pragma omp parallel for schedule(static) num_threads({self._num_threads})"
            if self._omp else ""
        )

        c_src = f"""
#include <stdint.h>
{'#include <omp.h>' if self._omp else ''}

typedef void (*kfn)({kernel_arg_decls},
                    int32_t, int32_t, int32_t,
                    int32_t, int32_t, int32_t);

void replay_chain(void* fp, int n_chain, {tramp_arg_decls}) {{
    kfn fn = (kfn)fp;
    int32_t G = {self._cfg.grid_x};
    for (int i = 0; i < n_chain; i++) {{
        {omp_pragma}
        for (int32_t pid = 0; pid < G; pid++) {{
            fn({tramp_call_args}, pid, 0, 0, G, 1, 1);
        }}
    }}
}}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write(c_src)
            cpath = f.name
        sopath = cpath.replace(".c", ".so")
        cmd = ["gcc", "-O2", "-shared", "-fPIC"]
        if self._omp:
            cmd += ["-fopenmp"]
        cmd += [cpath, "-o", sopath]
        subprocess.run(cmd, check=True)
        lib = ctypes.CDLL(sopath)

        argtypes = [ctypes.c_void_p, ctypes.c_int]  # fp, n_chain
        for t in self._cfg.arg_types:
            if t.startswith("*"):
                argtypes.append(ctypes.c_void_p)
            elif t in ("i32", "i64", "fp32", "fp64"):
                argtypes.append({"i32": ctypes.c_int32, "i64": ctypes.c_int64,
                                 "fp32": ctypes.c_float, "fp64": ctypes.c_double}[t])
        lib.replay_chain.argtypes = argtypes
        lib.replay_chain.restype = None
        return lib

    def _packed_args(self):
        """Pack self._args into the ctypes form replay_chain expects."""
        packed = []
        for arg, ty in zip(self._args, self._cfg.arg_types):
            if ty.startswith("*"):
                # tensor: use data_ptr()
                packed.append(arg.data_ptr())
            else:
                packed.append(arg)
        return packed

    def replay(self, n: int = 1):
        """Call the kernel `n` times in C, no Python wrapper per call."""
        packed = self._packed_args()
        self._lib.replay_chain(self._cfg.fn_ptr, n, *packed)

    def update_args(self, *new_args):
        """Replace args used at replay time (e.g. point at new tensors)."""
        if len(new_args) != len(self._args):
            raise ValueError(f"expected {len(self._args)} args, got {len(new_args)}")
        self._args = list(new_args)
