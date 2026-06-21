"""Benchmark + correctness for the non-temporal store pass.

Each row runs the same kernel two ways via subprocess (so each gets a
fresh JIT cache) with TRITON_CPU_NT_STORE flipped.

Sweet spot on a 13700K (32MB L3, 2MB L2/P-core): N in 64K..1M, 1.5×-1.8×.
Below ~32K: NT pays write-combine flush overhead with no cache pressure
to relieve, so it ties baseline. Above L3 size: DRAM-bound, NT wins shrink.
"""
import os
import shutil
import statistics
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def vec_add_nomask(X, Y, OUT, N, BLOCK: tl.constexpr):
    # NT-store pass only kicks in on plain `vector.store` (no mask) -- the
    # MLIR `LLVM_MaskedStoreOp` does not expose a `nontemporal` attribute.
    # Real workloads paired with tail-guard or aligned N hit this fast path.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    tl.store(OUT + offs, x + y)


def time_min(call, warmup=50, iters=500):
    for _ in range(warmup):
        call()
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        call()
        s.append(time.perf_counter() - t)
    return min(s)


def child(nt_flag, sizes):
    cache = f"/tmp/nt_bench_cache_{nt_flag}"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = cache
    BLOCK = 1024
    for N in sizes:
        if N % BLOCK:
            continue
        x = torch.randn(N)
        y = torch.randn(N)
        out = torch.zeros(N)

        def call():
            vec_add_nomask[(triton.cdiv(N, BLOCK),)](x, y, out, N, BLOCK=BLOCK)

        us = time_min(call) * 1e6
        # Correctness against torch reference
        err = (out - (x + y)).abs().max().item()
        print(f"{nt_flag},{N},{us:.6f},{err:.6e}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], [int(s) for s in sys.argv[3].split(",")])
        return

    sizes = [1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24]
    results = {}
    for setting in ("0", "1"):
        env = dict(os.environ)
        env["TRITON_CPU_TAIL_GUARD"] = "0"
        env["TRITON_CPU_NT_STORE"] = setting
        # Pin to a single thread for a stable bandwidth measurement.
        # Multi-threaded interleaving of NT and regular stores via OMP
        # can defeat the write-combining buffer.
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        proc = subprocess.run(
            [sys.executable, __file__, "child", setting,
             ",".join(str(n) for n in sizes)],
            env=env, capture_output=True, text=True, check=True,
        )
        for line in proc.stdout.strip().splitlines():
            s, N, us, err = line.split(",")
            results[(s, int(N))] = (float(us), float(err))

    print()
    print(f"{'N':>10}  {'OFF (us)':>10}  {'ON (us)':>10}  {'speedup':>8}"
          f"  {'OFF GB/s':>10}  {'ON GB/s':>10}  {'corr err':>10}")
    print("-" * 80)
    for N in sizes:
        us_off, err_off = results[("0", N)]
        us_on, err_on = results[("1", N)]
        bw_off = 3 * N * 4 / (us_off * 1e-6) / 1e9
        bw_on = 3 * N * 4 / (us_on * 1e-6) / 1e9
        spd = us_off / us_on
        worst_err = max(err_off, err_on)
        print(f"{N:>10,}  {us_off:>10.2f}  {us_on:>10.2f}  {spd:>7.2f}x"
              f"  {bw_off:>9.1f}   {bw_on:>9.1f}   {worst_err:>10.2e}")


if __name__ == "__main__":
    main()
