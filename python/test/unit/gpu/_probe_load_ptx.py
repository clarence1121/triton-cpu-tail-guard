"""Inspect what load opcodes Triton emits across BLOCK sizes.

Used to verify whether Triton already uses vector loads for a given
kernel shape. If it does, hand-written inline-PTX 128-bit loads buy
nothing; if it doesn't, there might be a real attack surface.
"""
import os, shutil, glob
import torch, triton, triton.language as tl


@triton.jit
def vec_add(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(OUT + offs, x + y)


def get_loads(call, name):
    cache = f"/tmp/loadprobe_{name}"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = cache
    call()
    for f in sorted(glob.glob(f"{cache}/**/*.ptx", recursive=True)):
        ops = []
        for line in open(f).read().split("\n"):
            if "ld.global" in line:
                parts = line.strip().split()
                op = next((p for p in parts if "ld.global" in p), None)
                if op:
                    ops.append(op)
        return ops
    return []


def main():
    N = 1 << 20
    x = torch.randn(N, device='cuda')
    y = torch.randn(N, device='cuda')
    out = torch.zeros(N, device='cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}, Triton {triton.__version__}\n")
    print(f"{'BLOCK':>6}  {'# loads':>8}  {'opcodes (unique)':>40}")
    for BLOCK in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if N % BLOCK != 0:
            continue
        loads = get_loads(lambda: vec_add[(N//BLOCK,)](x, y, out, N, BLOCK=BLOCK),
                          f"b{BLOCK}")
        uniq = sorted(set(loads))
        print(f"{BLOCK:>6}  {len(loads):>8}  {', '.join(uniq):>40}")


if __name__ == "__main__":
    main()
