"""GPU vec_add bandwidth across BLOCK sizes — finds where Triton's load
codegen saturates and where it doesn't.

Run on an SM89 (Ada/RTX 4090) or newer GPU with vanilla Triton.

Finding (4090 + Triton 3.2.0):
  - BLOCK >= 1024 → Triton emits ld.global.v4.b32 (128-bit), ~900 GB/s
  - BLOCK <  1024 → Triton emits ld.global.b32 (32-bit)
  - But at small BLOCK, launch overhead dominates, not load width.
    Going from BLOCK=32 (one warp per program, narrow load) to
    BLOCK=1024 (32 warps per program, wide load) buys ~3-4x bandwidth,
    BUT the gain is from FEWER programs to launch (1/32x), not from
    wider loads. Manual inline PTX to force 128-bit at BLOCK=32 would
    still leave you with 32x more programs than BLOCK=1024.
"""
import torch, triton, triton.language as tl

device = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(0)}, Triton {triton.__version__}\n")


@triton.jit
def vec_add(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(OUT + offs, x + y)


def main():
    sizes = [1 << 20, 1 << 24]
    blocks = [32, 64, 128, 256, 512, 1024, 2048]

    for N in sizes:
        x = torch.randn(N, device=device)
        y = torch.randn(N, device=device)
        out = torch.zeros(N, device=device)
        bytes_moved = 3 * N * 4

        print(f"\nN = {N:,} ({bytes_moved/1e6:.0f} MB total)")
        print(f"  {'BLOCK':>6}  {'time (us)':>10}  {'bandwidth':>12}")
        for BLOCK in blocks:
            assert N % BLOCK == 0
            def call():
                vec_add[(N // BLOCK,)](x, y, out, N, BLOCK=BLOCK)
            us = triton.testing.do_bench(call, warmup=50, rep=500, return_mode="min") * 1000
            bw_gbs = bytes_moved / (us * 1e-6) / 1e9
            print(f"  {BLOCK:>6}  {us:>10.2f}  {bw_gbs:>10.1f} GB/s")


if __name__ == "__main__":
    main()
