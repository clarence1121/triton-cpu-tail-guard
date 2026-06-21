"""Smoke test for triton.cpu_graph.CPUGraph API.

Verifies the ergonomic wrapper produces the same results as eager Triton
across single-thread, OMP-parallel, and update-args use cases.
"""
import os
os.environ["TRITON_CPU_TAIL_GUARD"] = "0"

import time
import torch
import triton
import triton.language as tl

from triton.cpu_graph import CPUGraph

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def axpy(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(OUT + offs, x + y)


def test_single_thread():
    N, BLOCK = 1024, 1024
    x = torch.randn(N)
    y = torch.randn(N)
    out_g = torch.zeros(N)
    out_e = torch.zeros(N)
    axpy[(1,)](x, y, out_e, N, BLOCK=BLOCK)

    g = CPUGraph.from_kernel(
        axpy, args=(x, y, out_g, N), grid=(1,),
        constexprs={"BLOCK": BLOCK},
    )
    g.replay(n=1)
    err = (out_g - out_e).abs().max().item()
    assert err < 1e-5, f"single-thread mismatch: {err}"
    print(f"  single_thread: err={err:.2e} OK")


def test_omp():
    N, BLOCK = 65536, 1024
    grid_x = N // BLOCK
    x = torch.randn(N)
    y = torch.randn(N)
    out_g = torch.zeros(N)
    out_e = torch.zeros(N)
    axpy[(grid_x,)](x, y, out_e, N, BLOCK=BLOCK)

    g = CPUGraph.from_kernel(
        axpy, args=(x, y, out_g, N), grid=(grid_x,),
        constexprs={"BLOCK": BLOCK},
        omp=True, num_threads=4,
    )
    g.replay(n=1)
    err = (out_g - out_e).abs().max().item()
    assert err < 1e-5, f"omp mismatch: {err}"
    print(f"  omp:           err={err:.2e} OK")


def test_update_args():
    N, BLOCK = 1024, 1024
    x1 = torch.randn(N); y1 = torch.randn(N); out1 = torch.zeros(N)
    x2 = torch.randn(N); y2 = torch.randn(N); out2 = torch.zeros(N)
    axpy[(1,)](x1, y1, out1, N, BLOCK=BLOCK)  # compile

    g = CPUGraph.from_kernel(
        axpy, args=(x1, y1, out1, N), grid=(1,),
        constexprs={"BLOCK": BLOCK},
    )
    g.replay(n=1)
    err1 = (out1 - (x1 + y1)).abs().max().item()
    assert err1 < 1e-5

    # Update to new buffers, replay again
    g.update_args(x2, y2, out2, N)
    g.replay(n=1)
    err2 = (out2 - (x2 + y2)).abs().max().item()
    assert err2 < 1e-5, f"update_args mismatch: {err2}"
    print(f"  update_args:   err1={err1:.2e}, err2={err2:.2e} OK")


def perf_demo():
    """Show speedup with the ergonomic API matches the hand-coded PoC."""
    N, BLOCK = 1024, 1024
    x = torch.randn(N); y = torch.randn(N); out = torch.zeros(N)
    axpy[(1,)](x, y, out, N, BLOCK=BLOCK)
    g = CPUGraph.from_kernel(
        axpy, args=(x, y, out, N), grid=(1,),
        constexprs={"BLOCK": BLOCK},
    )

    iters = 200
    print(f"\n  {'chain':>6}  {'eager (us)':>10}  {'graph (us)':>10}  {'speedup':>8}")
    for n in [1, 64, 1024]:
        # warmup
        for _ in range(20):
            for _ in range(n): axpy[(1,)](x, y, out, N, BLOCK=BLOCK)
            g.replay(n)
        t0 = time.perf_counter()
        for _ in range(iters):
            for _ in range(n): axpy[(1,)](x, y, out, N, BLOCK=BLOCK)
        us_e = (time.perf_counter() - t0) / iters * 1e6
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay(n)
        us_g = (time.perf_counter() - t0) / iters * 1e6
        print(f"  {n:>6}  {us_e:>10.2f}  {us_g:>10.2f}  {us_e/us_g:>7.2f}x")


if __name__ == "__main__":
    print("Correctness tests:")
    test_single_thread()
    test_omp()
    test_update_args()
    print("\nPerformance demo:")
    perf_demo()
    print("\nAll tests passed.")
