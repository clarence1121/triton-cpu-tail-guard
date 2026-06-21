"""Break down Triton-CPU per-launch overhead into Python wrapper vs C launcher.

Probes two call paths for the SAME kernel:
  1. eager:  kernel[grid](args)   — standard Python path through __getitem__,
             runner closure, constexpr resolution, signature serialization,
             then into the PyBind launch function.
  2. direct: compiled.run(...)    — skip __getitem__/runner, go straight to
             the C launcher's PyBind entry. Still pays the PyBind marshalling
             and the C launcher's OMP setup, but no wrapper Python work.

The gap (eager - direct) is the cost the CPU "graph" PoC is trying to amortize
across a chain of launches.

Result on this machine (Triton 3.7.0, tiny vec_add 1024 elements):
  eager  kernel[grid](args)  : ~7.4 us/call
  direct compiled.run(...)   : ~0.7 us/call
  -> Python wrapper overhead : ~6.7 us (91% of total)
"""
import os
os.environ["TRITON_CPU_TAIL_GUARD"] = "0"
import time
import torch, triton, triton.language as tl

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def tiny(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs); y = tl.load(Y + offs)
    tl.store(OUT + offs, x + y)


def main():
    N = 1024; BLOCK = 1024
    x = torch.randn(N); y = torch.randn(N); out = torch.zeros(N)
    tiny[(1,)](x, y, out, N, BLOCK=BLOCK)
    compiled = tiny.warmup(x, y, out, N, BLOCK=BLOCK, grid=(1,))
    fn = compiled.function
    meta = compiled.packed_metadata

    iters = 5000
    # Eager
    t0 = time.perf_counter()
    for _ in range(iters):
        tiny[(1,)](x, y, out, N, BLOCK=BLOCK)
    us_eager = (time.perf_counter() - t0) / iters * 1e6

    # Direct (note: constexpr BLOCK still occupies an arg slot in the C launcher)
    t0 = time.perf_counter()
    for _ in range(iters):
        compiled.run(1, 1, 1, 0, fn, meta, None, None, None, x, y, out, N, BLOCK)
    us_direct = (time.perf_counter() - t0) / iters * 1e6

    print(f"eager  kernel[grid](args)  : {us_eager:.2f} us/call")
    print(f"direct compiled.run(...)   : {us_direct:.2f} us/call")
    print(f"Python wrapper overhead    : {us_eager - us_direct:.2f} us "
          f"({100*(us_eager - us_direct)/us_eager:.0f}% of total)")


if __name__ == "__main__":
    main()
