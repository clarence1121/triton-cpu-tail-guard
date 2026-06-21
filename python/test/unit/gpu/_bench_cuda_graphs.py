"""CUDA Graphs vs eager Triton kernel chain — orchestration-layer win.

Workload model: a chain of N tiny Triton kernels (e.g. RNN cell, LLM
decode step, ML pipeline). Each kernel does a small amount of work, so
per-launch driver overhead dominates wall-clock.

CUDA Graphs captures the whole chain into a single dispatchable graph
that the driver replays in hardware, amortizing launch overhead.

Result on RTX 4090 + Triton 3.2.0: 2× ... 12× speedup, growing with
chain length. This is the ONE clean positive result from the GPU
side-project (see GPU_EXPERIMENTS.md for the negative results).
"""
import torch, triton, triton.language as tl

device = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")


@triton.jit
def tiny_kernel(X, OUT, N, BLOCK: tl.constexpr):
    """Add 1 to N elements. Tiny — workload exists so the kernel is non-trivial."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(OUT + offs, x + 1.0, mask=mask)


def time_eager(x, scratch, n_ops, N, BLOCK):
    grid = ((N + BLOCK - 1) // BLOCK,)
    for _ in range(20):
        a, b = x, scratch
        for i in range(n_ops):
            tiny_kernel[grid](a, b, N, BLOCK=BLOCK); a, b = b, a
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(50):
        s.record()
        a, b = x, scratch
        for i in range(n_ops):
            tiny_kernel[grid](a, b, N, BLOCK=BLOCK); a, b = b, a
        e.record(); e.synchronize()
        samples.append(s.elapsed_time(e) * 1000)
    return min(samples)


def time_graph(x, scratch, n_ops, N, BLOCK):
    grid = ((N + BLOCK - 1) // BLOCK,)
    for _ in range(20):
        a, b = x, scratch
        for i in range(n_ops):
            tiny_kernel[grid](a, b, N, BLOCK=BLOCK); a, b = b, a
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(g):
            a, b = x, scratch
            for i in range(n_ops):
                tiny_kernel[grid](a, b, N, BLOCK=BLOCK); a, b = b, a
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(50):
        s.record(); g.replay(); e.record(); e.synchronize()
        samples.append(s.elapsed_time(e) * 1000)
    return min(samples)


def main():
    BLOCK = 1024
    Nmax = 1 << 20
    x = torch.randn(Nmax, device=device)
    scratch = torch.zeros(Nmax, device=device)

    print(f"BLOCK = {BLOCK}, tiny kernel = vec += 1")
    print(f"{'chain len':>10}  {'N':>8}  {'eager us':>10}  {'graph us':>10}  {'speedup':>8}")
    print("-" * 60)
    for n_ops in [1, 4, 16, 64, 256]:
        for N in [256, 4096, 65536]:
            us_e = time_eager(x[:N], scratch[:N], n_ops, N, BLOCK)
            us_g = time_graph(x[:N], scratch[:N], n_ops, N, BLOCK)
            print(f"{n_ops:>10}  {N:>8}  {us_e:>10.2f}  {us_g:>10.2f}  {us_e/us_g:>7.2f}x")


if __name__ == "__main__":
    main()
