"""GPU grid-launch vs persistent vs dynamic-persistent comparison.

Two workload kinds:
  1. Uniform vec_add: tile costs equal.  Hardware scheduler wins.
  2. Variable-length: random tile costs.  Tests work-stealing potential.

Run on any CUDA GPU with vanilla Triton.

Finding (4090 + Triton 3.2.0):
  - For uniform workloads, persistent does NOT beat grid-launch — the
    GPU hardware scheduler distributes tiles across SMs very efficiently.
  - For variable-length, atomic-counter-based dynamic persistent LOSES
    to grid (the 128 SMs contend for the global counter every tile,
    and that contention costs more than the load imbalance it solves).
  - Persistent kernels do win in narrower scenarios:
    * Chaining many tiny kernels back-to-back (skipping the host-side
      launch latency between them)
    * Hopper TMA + clusters (DSM coordination)
    * Producer/consumer warp specialization (no API on 3.2.0)
"""
import torch, triton, triton.language as tl

device = torch.device("cuda:0")
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
print(f"GPU: {torch.cuda.get_device_name(0)}, SMs={NUM_SMS}\n")


# Uniform vec_add: grid vs static persistent
@triton.jit
def vec_add_grid(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(OUT + offs, x + y, mask=mask)


@triton.jit
def vec_add_persistent(X, Y, OUT, N, NUM_TILES,
                       BLOCK: tl.constexpr, NUM_PROGRAMS: tl.constexpr):
    pid = tl.program_id(0)
    for tile in range(pid, NUM_TILES, NUM_PROGRAMS):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = tl.load(Y + offs, mask=mask)
        tl.store(OUT + offs, x + y, mask=mask)


# Variable-length row reduction: grid / static / dynamic persistent
@triton.jit
def varlen_grid(X, OUT, OFFSETS, NUM_TILES, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = tl.load(OFFSETS + pid)
    end   = tl.load(OFFSETS + pid + 1)
    acc = 0.0
    for off in range(start, end, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        x = tl.load(X + idx, mask=mask, other=0.0)
        acc += tl.sum(x)
    tl.store(OUT + pid, acc)


@triton.jit
def varlen_static_pers(X, OUT, OFFSETS, NUM_TILES,
                       BLOCK: tl.constexpr, NUM_PROGRAMS: tl.constexpr):
    pid = tl.program_id(0)
    for tile in range(pid, NUM_TILES, NUM_PROGRAMS):
        start = tl.load(OFFSETS + tile)
        end   = tl.load(OFFSETS + tile + 1)
        acc = 0.0
        for off in range(start, end, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < end
            x = tl.load(X + idx, mask=mask, other=0.0)
            acc += tl.sum(x)
        tl.store(OUT + tile, acc)


@triton.jit
def varlen_dynamic_pers(X, OUT, OFFSETS, NUM_TILES, COUNTER,
                        BLOCK: tl.constexpr):
    done = False
    while not done:
        tile = tl.atomic_add(COUNTER, 1)
        if tile >= NUM_TILES:
            done = True
        else:
            start = tl.load(OFFSETS + tile)
            end   = tl.load(OFFSETS + tile + 1)
            acc = 0.0
            for off in range(start, end, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                mask = idx < end
                x = tl.load(X + idx, mask=mask, other=0.0)
                acc += tl.sum(x)
            tl.store(OUT + tile, acc)


def make_varlen(num_tiles, max_len, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    lens = torch.randint(1, max_len + 1, (num_tiles,), generator=g, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32),
                         torch.cumsum(lens, 0).to(torch.int32)])
    total = int(offsets[-1])
    x = torch.randn(total, device=device, dtype=torch.float32)
    out = torch.zeros(num_tiles, device=device, dtype=torch.float32)
    return x, out, offsets.to(device)


def t(fn):
    fn()
    return triton.testing.do_bench(fn, warmup=20, rep=200, return_mode="min") * 1000


def main():
    print("─── 1. Uniform vec_add: grid vs static persistent ───")
    BLOCK = 1024
    print(f"\nBLOCK={BLOCK}")
    print(f"{'N':>10}  {'tiles':>6}  {'grid us':>9}  {'pers us':>9}  {'speedup':>8}")
    for N in [1024, 16384, 262144, 1048576, 16777216, 67108864]:
        n_tiles = (N + BLOCK - 1) // BLOCK
        x = torch.randn(N, device=device); y = torch.randn(N, device=device)
        out = torch.zeros(N, device=device)
        us_g = t(lambda: vec_add_grid[(n_tiles,)](x, y, out, N, BLOCK=BLOCK))
        us_p = t(lambda: vec_add_persistent[(NUM_SMS,)](x, y, out, N, n_tiles,
                                                         BLOCK=BLOCK, NUM_PROGRAMS=NUM_SMS))
        print(f"{N:>10,}  {n_tiles:>6}  {us_g:>9.2f}  {us_p:>9.2f}  {us_g/us_p:>7.2f}x")

    print("\n─── 2. Variable-length workload ───")
    print(f"{'tiles':>6}  {'max_len':>8}  {'grid us':>9}  {'static us':>10}"
          f"  {'dyn us':>9}  {'st/grid':>8}  {'dyn/grid':>9}")
    for num_tiles, max_len in [(128, 1024), (1024, 1024), (4096, 1024)]:
        x, out, off = make_varlen(num_tiles, max_len)
        counter = torch.zeros(1, dtype=torch.int32, device=device)
        us_g = t(lambda: varlen_grid[(num_tiles,)](x, out, off, num_tiles, BLOCK=128))
        us_s = t(lambda: varlen_static_pers[(NUM_SMS,)](x, out, off, num_tiles,
                                                          BLOCK=128, NUM_PROGRAMS=NUM_SMS))
        def call_dyn():
            counter.zero_()
            varlen_dynamic_pers[(NUM_SMS,)](x, out, off, num_tiles, counter, BLOCK=128)
        us_d = t(call_dyn)
        print(f"{num_tiles:>6}  {max_len:>8}  {us_g:>9.2f}  {us_s:>10.2f}"
              f"  {us_d:>9.2f}  {us_g/us_s:>7.2f}x  {us_g/us_d:>8.2f}x")


if __name__ == "__main__":
    main()
