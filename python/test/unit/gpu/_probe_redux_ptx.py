"""Inspect Triton's warp-reduction PTX vs the (unavailable on SM89) redux.sync.

What we see (Triton 3.2.0):
  - tl.sum on warp-sized input → 5 shfl.sync.bfly.b32 instructions.
  - redux.sync.add.f32 would do this in 1 instruction, BUT only on
    sm_100+ (Blackwell). On sm_89 (Ada/RTX 4090) and sm_90 (Hopper),
    only INTEGER redux.sync (u32/s32) is supported -- no FP variant.
  - So for FP reductions on Ada/Hopper, the shfl tree IS what we get;
    inline PTX gains nothing.
"""
import os, shutil, glob
import torch, triton, triton.language as tl


@triton.jit
def warp_sum(X, OUT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    s = tl.sum(x)
    tl.store(OUT, s)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}, Triton {triton.__version__}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: sm_{cap[0]}{cap[1]}")
    print(f"redux.sync FP support: {'YES' if cap[0] >= 10 else 'NO (FP redux is sm_100+ Blackwell)'}")
    print()
    N = 32
    x = torch.randn(N, device='cuda')
    out = torch.zeros(1, device='cuda')
    cache = "/tmp/reduxprobe"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = cache
    warp_sum[(1,)](x, out, BLOCK=N)
    for f in sorted(glob.glob(f"{cache}/**/*.ptx", recursive=True)):
        text = open(f).read()
        print(f"--- {f} ---")
        for line in text.split("\n"):
            for kw in ["shfl.sync", "redux.sync", "bar.sync"]:
                if kw in line:
                    print(f"  {line.strip()}")
        break


if __name__ == "__main__":
    main()
