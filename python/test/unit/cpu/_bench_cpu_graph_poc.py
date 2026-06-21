"""CPU Triton 'graph' proof-of-concept: bypass Python wrapper via C replay loop.

Inspired by NVIDIA CUDA Graphs (which on GPU gave 2x-12x for chained kernels,
see GPU_EXPERIMENTS.md). The CPU analogue: amortize Triton's per-launch Python
wrapper cost across a chained kernel sequence by running the launches in C.

Findings on this machine (Triton 3.7.0 + CPU backend):

  Per-launch overhead breakdown (probed in _bench_cpu_graph_overhead.py):
    eager  kernel[grid](args) Python path:  ~7.4 us
    direct compiled.run(...) C entry path:  ~0.7 us
    -> Python wrapper is ~91% of per-launch cost for tiny kernels

  Graph PoC (this script): emit a C function that calls the JIT'd kernel
  function pointer in a tight loop. Single-thread, grid=(1,), 1024-element
  tiny kernel:

      chain  eager (us)  graph (us)   speedup
          1        8.45        0.53    15.87x
          4       31.10        0.62    50.47x
         16      119.59        1.07   111.42x
         64      474.19        2.93   161.85x
        256     1908.59       10.07   189.46x
       1024     7653.72       38.56   198.47x

  Correctness: bit-exact vs eager output.

Important caveats (deliberate scope limits of this PoC):

  1. Single-thread, grid=(1,) only. Multi-program OMP path segfaults on
     first attempt -- likely thread-init / TLS interaction. Future work.
  2. Same kernel called N times with same args. Real "graph" use case
     (different kernels, ping-pong buffers) needs per-call arg recording.
  3. Tiny kernel; for kernels with significant compute, Python wrapper
     becomes smaller share of total -> smaller speedup.

This script demonstrates the *ceiling*, not the production technique.
"""
import os
os.environ["TRITON_CPU_TAIL_GUARD"] = "0"
import time, ctypes, tempfile, subprocess
import torch, triton, triton.language as tl

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def tiny_vec_add(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(OUT + offs, x + y)


def build_chain_so():
    """Compile a tiny C trampoline that calls the kernel fn N times.

    The kernel function pointer's signature for a (X, Y, OUT, N) Triton CPU
    kernel after JIT codegen is:
      void(float*, float*, float*, int32_t,
           int32_t pid_x, int32_t pid_y, int32_t pid_z,
           int32_t gridX, int32_t gridY, int32_t gridZ)
    """
    c_src = r"""
#include <stdint.h>
typedef void (*kernel_fn_t)(float*, float*, float*, int32_t,
                            int32_t, int32_t, int32_t,
                            int32_t, int32_t, int32_t);
void replay_chain(void* fn_ptr, float* x, float* y, float* out, int32_t N, int n_chain) {
    kernel_fn_t fn = (kernel_fn_t)fn_ptr;
    for (int i = 0; i < n_chain; i++) {
        fn(x, y, out, N, 0, 0, 0, 1, 1, 1);
    }
}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_src); cpath = f.name
    sopath = cpath.replace('.c', '.so')
    subprocess.run(['gcc', '-O2', '-shared', '-fPIC', cpath, '-o', sopath], check=True)
    lib = ctypes.CDLL(sopath)
    lib.replay_chain.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int,
    ]
    lib.replay_chain.restype = None
    return lib


def main():
    N = 1024
    BLOCK = 1024
    x = torch.randn(N)
    y = torch.randn(N)
    out_eager = torch.zeros(N)
    out_graph = torch.zeros(N)

    # Compile + get raw kernel function pointer
    tiny_vec_add[(1,)](x, y, out_eager, N, BLOCK=BLOCK)
    compiled = tiny_vec_add.warmup(x, y, out_eager, N, BLOCK=BLOCK, grid=(1,))
    fn_ptr = compiled.function
    assert fn_ptr is not None, "compiled.function is None — kernel handles not initialized"
    print(f"kernel fn_ptr: 0x{fn_ptr:x}")

    lib = build_chain_so()

    # Correctness: 1 graph call == 1 eager call
    out_graph.zero_(); out_eager.zero_()
    lib.replay_chain(fn_ptr, x.data_ptr(), y.data_ptr(), out_graph.data_ptr(), N, 1)
    tiny_vec_add[(1,)](x, y, out_eager, N, BLOCK=BLOCK)
    err = (out_graph - out_eager).abs().max().item()
    print(f"Correctness max abs err: {err}")
    assert err < 1e-5

    # Benchmark across chain lengths
    iters = 200
    print(f"\n{'chain':>6}  {'eager (us)':>10}  {'graph (us)':>10}  {'speedup':>8}")
    print("-" * 50)
    for n_chain in [1, 4, 16, 64, 256, 1024]:
        # warmup
        for _ in range(20):
            for _ in range(n_chain):
                tiny_vec_add[(1,)](x, y, out_eager, N, BLOCK=BLOCK)
            lib.replay_chain(fn_ptr, x.data_ptr(), y.data_ptr(),
                             out_graph.data_ptr(), N, n_chain)
        # eager
        t0 = time.perf_counter()
        for _ in range(iters):
            for _ in range(n_chain):
                tiny_vec_add[(1,)](x, y, out_eager, N, BLOCK=BLOCK)
        us_e = (time.perf_counter() - t0) / iters * 1e6
        # graph
        t0 = time.perf_counter()
        for _ in range(iters):
            lib.replay_chain(fn_ptr, x.data_ptr(), y.data_ptr(),
                             out_graph.data_ptr(), N, n_chain)
        us_g = (time.perf_counter() - t0) / iters * 1e6
        print(f"{n_chain:>6}  {us_e:>10.2f}  {us_g:>10.2f}  {us_e/us_g:>7.2f}x")


if __name__ == "__main__":
    main()
