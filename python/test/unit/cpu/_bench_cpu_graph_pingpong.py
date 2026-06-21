"""CPU graph PoC with DIFFERENT args per call — realistic ping-pong pattern.

Simulates the typical chain shape: each call writes to a different output
buffer, which becomes the input for the next call. Tests whether the
trampoline approach generalizes beyond "same args repeated".
"""
import os
os.environ["TRITON_CPU_TAIL_GUARD"] = "0"
import time, ctypes, tempfile, subprocess
import torch, triton, triton.language as tl

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def tiny_axpy(X, Y, OUT, N, BLOCK: tl.constexpr):
    """OUT = X + Y, tiny."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs); y = tl.load(Y + offs)
    tl.store(OUT + offs, x + y)


def build_pingpong_so():
    """Trampoline takes arrays of (x_ptr, y_ptr, out_ptr) per call position."""
    c_src = r"""
#include <stdint.h>
typedef void (*kfn)(float*, float*, float*, int32_t,
                    int32_t, int32_t, int32_t,
                    int32_t, int32_t, int32_t);
void replay_chain(void* fp,
                  float** xs, float** ys, float** outs,
                  int32_t N, int n_chain) {
    kfn fn = (kfn)fp;
    for (int i = 0; i < n_chain; i++) {
        fn(xs[i], ys[i], outs[i], N, 0, 0, 0, 1, 1, 1);
    }
}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_src); cp = f.name
    sp = cp.replace('.c', '.so')
    subprocess.run(['gcc', '-O2', '-shared', '-fPIC', cp, '-o', sp], check=True)
    lib = ctypes.CDLL(sp)
    lib.replay_chain.argtypes = [
        ctypes.c_void_p,                          # fn_ptr
        ctypes.POINTER(ctypes.c_void_p),          # xs[]
        ctypes.POINTER(ctypes.c_void_p),          # ys[]
        ctypes.POINTER(ctypes.c_void_p),          # outs[]
        ctypes.c_int32,                            # N
        ctypes.c_int,                              # n_chain
    ]
    lib.replay_chain.restype = None
    return lib


def make_pingpong_buffers(N, n_chain):
    """Allocate n_chain+1 buffers; first is input, rest are intermediate/final."""
    bufs = [torch.randn(N) if i == 0 else torch.zeros(N) for i in range(n_chain + 1)]
    # x[i] = bufs[i], out[i] = bufs[i+1], y[i] = some constant buffer
    y_const = torch.ones(N)
    xs = [bufs[i].data_ptr() for i in range(n_chain)]
    ys = [y_const.data_ptr()] * n_chain
    outs = [bufs[i+1].data_ptr() for i in range(n_chain)]
    return bufs, y_const, xs, ys, outs


def main():
    N = 1024; BLOCK = 1024
    tiny_axpy[(1,)](torch.zeros(N), torch.zeros(N), torch.zeros(N), N, BLOCK=BLOCK)
    compiled = tiny_axpy.warmup(torch.zeros(N), torch.zeros(N), torch.zeros(N), N,
                                 BLOCK=BLOCK, grid=(1,))
    fn = compiled.function
    lib = build_pingpong_so()
    PtrArr = ctypes.c_void_p * 256  # max chain we'll test

    # Correctness: chain=4, each step adds y_const to the previous result
    n_chain = 4
    bufs, y_const, xs, ys, outs = make_pingpong_buffers(N, n_chain)
    xs_arr = PtrArr(*[xs[i] for i in range(n_chain)] + [0]*(256-n_chain))
    ys_arr = PtrArr(*[ys[i] for i in range(n_chain)] + [0]*(256-n_chain))
    outs_arr = PtrArr(*[outs[i] for i in range(n_chain)] + [0]*(256-n_chain))
    lib.replay_chain(fn, xs_arr, ys_arr, outs_arr, N, n_chain)
    # Expected: bufs[4] = bufs[0] + 4 * 1 = bufs[0] + 4
    expected = bufs[0] + 4
    err = (bufs[4] - expected).abs().max().item()
    print(f"Correctness (n_chain=4): max abs err = {err}")
    assert err < 1e-5

    # Bench
    print(f"\n{'chain':>6}  {'eager (us)':>11}  {'graph (us)':>11}  {'speedup':>8}")
    iters = 200
    for n_chain in [1, 4, 16, 64, 256]:
        bufs, y_const, xs, ys, outs = make_pingpong_buffers(N, n_chain)
        # Reset bufs[0] each iter for fair comparison
        x0 = bufs[0].clone()
        xs_arr = PtrArr(*[xs[i] for i in range(n_chain)] + [0]*(256-n_chain))
        ys_arr = PtrArr(*[ys[i] for i in range(n_chain)] + [0]*(256-n_chain))
        outs_arr = PtrArr(*[outs[i] for i in range(n_chain)] + [0]*(256-n_chain))
        # warmup
        for _ in range(20):
            bufs[0].copy_(x0)
            for i in range(n_chain):
                tiny_axpy[(1,)](bufs[i], y_const, bufs[i+1], N, BLOCK=BLOCK)
            lib.replay_chain(fn, xs_arr, ys_arr, outs_arr, N, n_chain)
        # Eager
        t0 = time.perf_counter()
        for _ in range(iters):
            for i in range(n_chain):
                tiny_axpy[(1,)](bufs[i], y_const, bufs[i+1], N, BLOCK=BLOCK)
        us_e = (time.perf_counter() - t0) / iters * 1e6
        # Graph
        t0 = time.perf_counter()
        for _ in range(iters):
            lib.replay_chain(fn, xs_arr, ys_arr, outs_arr, N, n_chain)
        us_g = (time.perf_counter() - t0) / iters * 1e6
        print(f"{n_chain:>6}  {us_e:>11.2f}  {us_g:>11.2f}  {us_e/us_g:>7.2f}x")


if __name__ == "__main__":
    main()
