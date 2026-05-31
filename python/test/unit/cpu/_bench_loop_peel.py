"""
Loop-peeling benchmark for reduction-style kernels.

Run:
    conda run -n triton-cpu python python/test/unit/cpu/_bench_loop_peel.py

Compares two compilations of the same kernel:
  - disabled : TRITON_CPU_LOOP_PEEL=0  (single masked loop)
  - enabled  : TRITON_CPU_LOOP_PEEL=1  (peeled: main without mask + epilogue with mask)

The kernel is a single-program reduction over N elements with no
divisibility hint on N -- the existing OptimizeMask analysis can't prove
the per-iteration mask all-ones, so without peel every iteration pays
the cmpi + vmaskmovps cost. With peel, ~N/BLOCK iterations run as plain
vector.load.

Timing uses triton.testing.do_bench(measure_time_with_hooks=True). The
default `kernel[grid](...)` call path includes a Python-side launcher
(signature serialize / constexpr resolve / arg pack) whose overhead is
in the same order of magnitude as a fast kernel and dilutes the
kernel-level speedup -- and at large N, the launcher overhead variance
can flip the sign of the comparison. The CPU-only entry/exit hooks
bracket the actual kernel execution, isolating the IR change.

We use N values that are NOT multiples of BLOCK so that Triton's JIT
does not auto-infer tt.divisibility on N.
"""
import os
import shutil
import subprocess
import sys

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active_to_cpu()


@triton.jit
def sum_kernel(X, OUT, N, BLOCK: tl.constexpr):
    acc = 0.0
    for offs in range(0, N, BLOCK):
        lanes = offs + tl.arange(0, BLOCK)
        mask = lanes < N
        x = tl.load(X + lanes, mask=mask, other=0.0)
        acc += tl.sum(x)
    tl.store(OUT, acc)


def time_kernel(N: int, block: int) -> float:
    """Returns median seconds-per-call, kernel-only (no Python wrapper)."""
    x = torch.randn(N, dtype=torch.float32)
    out = torch.zeros(1, dtype=torch.float32)

    def call():
        sum_kernel[(1,)](x, out, N, BLOCK=block)

    call()  # warmup compile
    # return_mode="min" is more stable than median for hot-path measurements:
    # the cold/throttled iterations get filtered out automatically.
    ms = triton.testing.do_bench(
        call, warmup=100, rep=500,
        measure_time_with_hooks=True,
        return_mode="min",
    )
    return ms * 1e-3  # ms → s


def child_run(setting: str, sizes, block: int):
    """Subprocess entry: forces fresh JIT cache so the env-var-gated pass
    actually re-runs. Prints one CSV line per size."""
    cache_dir = f"/tmp/triton_peel_bench_{setting}"
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    for N in sizes:
        secs = time_kernel(N, block)
        # CSV: setting,N,block,seconds
        print(f"{setting},{N},{block},{secs:.9f}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        setting = sys.argv[2]
        sizes = [int(s) for s in sys.argv[3].split(",")]
        block = int(sys.argv[4])
        child_run(setting, sizes, block)
        return

    sizes = [10007, 65537, 262147, 1048573]  # all prime-ish, not mult of 16
    block = 16

    results = {}
    for setting in ("disabled", "enabled"):
        env = dict(os.environ)
        env["TRITON_CPU_LOOP_PEEL"] = "0" if setting == "disabled" else "1"
        proc = subprocess.run(
            [sys.executable, __file__, "child", setting,
             ",".join(str(s) for s in sizes), str(block)],
            env=env, capture_output=True, text=True, check=True,
        )
        for line in proc.stdout.strip().splitlines():
            s, N, _b, secs = line.split(",")
            results[(s, int(N))] = float(secs)

    print()
    print(f"{'N':>10}  {'disabled (us)':>15}  {'enabled (us)':>15}  {'speedup':>8}")
    print("-" * 56)
    for N in sizes:
        d = results[("disabled", N)] * 1e6
        e = results[("enabled", N)] * 1e6
        speedup = d / e if e > 0 else float("nan")
        print(f"{N:>10}  {d:>15.2f}  {e:>15.2f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
