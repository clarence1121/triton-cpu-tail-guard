"""Correctness check for loop-peel pass.

For multiple Ns (some aligned, some not), compare:
  - peel disabled
  - peel enabled
  - torch reference
All three must agree to within float tolerance.
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


def child(setting: str, n_list):
    cache = f"/tmp/triton_peel_correctness_{setting}"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = cache
    # Use the same input for every N so child results are comparable
    g = torch.Generator().manual_seed(42)
    out_lines = []
    for N in n_list:
        x = torch.randn(N, generator=g, dtype=torch.float32)
        out = torch.zeros(1, dtype=torch.float32)
        sum_kernel[(1,)](x, out, N, BLOCK=16)
        out_lines.append(f"{N},{out.item():.6f}")
    print("\n".join(out_lines))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        setting = sys.argv[2]
        ns = [int(s) for s in sys.argv[3].split(",")]
        child(setting, ns)
        return

    # Mix: 16-multiples (peel won't fire), non-multiples (peel fires),
    # edge cases (very small, just over a multiple)
    sizes = [16, 17, 31, 32, 100, 1024, 10007, 65537, 262147]
    settings = ("disabled", "enabled")
    results = {}
    for s in settings:
        env = dict(os.environ)
        env["TRITON_CPU_LOOP_PEEL"] = "0" if s == "disabled" else "1"
        proc = subprocess.run(
            [sys.executable, __file__, "child", s, ",".join(str(n) for n in sizes)],
            env=env, capture_output=True, text=True, check=True,
        )
        for line in proc.stdout.strip().splitlines():
            n, v = line.split(",")
            results[(s, int(n))] = float(v)

    # Torch reference: use the same generator/seed as child()
    g = torch.Generator().manual_seed(42)
    refs = {}
    for N in sizes:
        x = torch.randn(N, generator=g, dtype=torch.float32)
        refs[N] = x.sum().item()

    print(f"{'N':>10}  {'disabled':>14}  {'enabled':>14}  {'torch':>14}  status")
    print("-" * 70)
    ok = True
    for N in sizes:
        d = results[("disabled", N)]
        e = results[("enabled", N)]
        r = refs[N]
        # f32 reduction accumulates error in O(N) elements; allow a bit of slack
        atol = max(1e-4 * abs(r), 1e-3)
        d_ok = abs(d - r) <= atol
        e_ok = abs(e - r) <= atol
        de_ok = abs(d - e) <= 1e-6 * max(abs(d), 1.0)  # same kernel, should be near-bit-exact
        status = []
        if not d_ok: status.append("DISABLED!=TORCH")
        if not e_ok: status.append("ENABLED!=TORCH")
        if not de_ok: status.append("DISABLED!=ENABLED")
        if not status:
            status = ["ok"]
        else:
            ok = False
        print(f"{N:>10}  {d:>14.6f}  {e:>14.6f}  {r:>14.6f}  {','.join(status)}")
    print()
    print("ALL OK" if ok else "FAILURES DETECTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
