"""CPU graph PoC with OMP-parallel kernel (grid > 1).

Handles tail-guard wrapped compiled objects by extracting main_kernel.
Trampoline does OMP-parallel call across program ids, looped n_chain.
"""
import os, time, ctypes, tempfile, subprocess
import torch, triton, triton.language as tl

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def vec_add(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(OUT + offs, x + y, mask=mask)


def get_fn_ptr(compiled):
    """Extract kernel function pointer, unwrapping tail-guard variants."""
    fn = compiled.function
    if fn is not None:
        return fn
    for attr in ('main_kernel', 'corner_kernel'):
        if hasattr(compiled, attr):
            sub = getattr(compiled, attr)
            sub._init_handles()
            if sub.function is not None:
                return sub.function
    raise RuntimeError("can't find function pointer")


def build_omp_so(num_threads):
    c_src = f"""
#include <stdint.h>
#include <omp.h>
typedef void (*kfn)(float*, float*, float*, int32_t,
                    int32_t, int32_t, int32_t,
                    int32_t, int32_t, int32_t);
void replay_chain(void* fp, float* x, float* y, float* o,
                  int32_t N, int32_t G, int n_chain) {{
    kfn fn = (kfn)fp;
    for (int i = 0; i < n_chain; i++) {{
        #pragma omp parallel for schedule(static) num_threads({num_threads})
        for (int32_t pid = 0; pid < G; pid++) {{
            fn(x, y, o, N, pid, 0, 0, G, 1, 1);
        }}
    }}
}}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(c_src); cp = f.name
    sp = cp.replace('.c', '.so')
    subprocess.run(['gcc', '-O2', '-fopenmp', '-shared', '-fPIC', cp, '-o', sp], check=True)
    lib = ctypes.CDLL(sp)
    lib.replay_chain.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int32, ctypes.c_int32, ctypes.c_int]
    return lib


def main():
    NUM_THREADS = 8
    print(f"OMP_NUM_THREADS={NUM_THREADS}\n")
    for N, BLOCK in [(4096, 1024), (65536, 1024), (1048576, 1024)]:
        grid_x = (N + BLOCK - 1) // BLOCK
        x = torch.randn(N); y = torch.randn(N)
        out_e = torch.zeros(N); out_g = torch.zeros(N)
        vec_add[(grid_x,)](x, y, out_e, N, BLOCK=BLOCK)
        compiled = vec_add.warmup(x, y, out_e, N, BLOCK=BLOCK, grid=(grid_x,))
        fn = get_fn_ptr(compiled)
        lib = build_omp_so(NUM_THREADS)

        # Correctness
        out_g.zero_(); out_e.zero_()
        lib.replay_chain(fn, x.data_ptr(), y.data_ptr(), out_g.data_ptr(), N, grid_x, 1)
        vec_add[(grid_x,)](x, y, out_e, N, BLOCK=BLOCK)
        err = (out_g - out_e).abs().max().item()

        print(f"N={N:,} grid={grid_x:>5} (correctness err={err:.6f})")
        print(f"  {'chain':>6}  {'eager (us)':>11}  {'graph (us)':>11}  {'speedup':>8}")
        iters = 50
        for n_chain in [1, 4, 16, 64]:
            for _ in range(20):
                for _ in range(n_chain): vec_add[(grid_x,)](x, y, out_e, N, BLOCK=BLOCK)
                lib.replay_chain(fn, x.data_ptr(), y.data_ptr(), out_g.data_ptr(), N, grid_x, n_chain)
            t0 = time.perf_counter()
            for _ in range(iters):
                for _ in range(n_chain): vec_add[(grid_x,)](x, y, out_e, N, BLOCK=BLOCK)
            us_e = (time.perf_counter() - t0) / iters * 1e6
            t0 = time.perf_counter()
            for _ in range(iters):
                lib.replay_chain(fn, x.data_ptr(), y.data_ptr(), out_g.data_ptr(), N, grid_x, n_chain)
            us_g = (time.perf_counter() - t0) / iters * 1e6
            print(f"  {n_chain:>6}  {us_e:>11.2f}  {us_g:>11.2f}  {us_e/us_g:>7.2f}x")
        print()


if __name__ == "__main__":
    main()
