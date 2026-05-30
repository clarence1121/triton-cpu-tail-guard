import torch

import triton
import triton.language as tl


@triton.jit
def add_tail_guard_kernel(x, y, z, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(x + offsets, mask=mask, other=0.0)
    b = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(z + offsets, a + b, mask=mask)


@triton.jit
def unsupported_tail_guard_kernel(x, y, z, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets + 1 < n
    a = tl.load(x + offsets, mask=mask, other=0.0)
    b = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(z + offsets, a + b, mask=mask)


@triton.jit
def block_start_tail_guard_kernel(x, y, z, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(x + offsets, mask=mask, other=0.0)
    b = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(z + offsets, a + b, mask=mask)


@triton.jit
def disabled_tail_guard_kernel(x, y, z, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    a = tl.load(x + offsets, mask=mask, other=0.0)
    b = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(z + offsets, a + b, mask=mask)


def _run_add(kernel, n, block=16):
    x = torch.arange(n, dtype=torch.float32, device="cpu")
    y = torch.arange(n, dtype=torch.float32, device="cpu") * 2
    z = torch.empty_like(x)
    grid = (triton.cdiv(n, block), )
    compiled = kernel[grid](x, y, z, n, BLOCK=block)
    return compiled, x, y, z


def test_tail_guard_divisible():
    compiled, x, y, z = _run_add(add_tail_guard_kernel, 64)
    assert torch.equal(z, x + y)
    assert compiled.metadata.cpu_tail_guard["matched"]
    assert hasattr(compiled, "main_kernel")
    assert "vector.load" in compiled.main_asm["tttcir"]
    assert "vector.maskedload" not in compiled.main_asm["tttcir"]
    assert "vector.maskedstore" not in compiled.main_asm["tttcir"]
    assert "vector.maskedload" in compiled.asm["tttcir"]
    assert "vector.maskedstore" in compiled.asm["tttcir"]


def test_tail_guard_non_divisible():
    compiled, x, y, z = _run_add(add_tail_guard_kernel, 70)
    assert torch.equal(z, x + y)
    assert compiled.metadata.cpu_tail_guard["matched"]


def test_tail_guard_unsupported_mask_falls_back():
    n = 33
    compiled, x, y, z = _run_add(unsupported_tail_guard_kernel, n)
    expected = x + y
    assert torch.equal(z[:n - 1], expected[:n - 1])
    assert not compiled.metadata.cpu_tail_guard["matched"]
    assert "missing mask = offsets < n" in compiled.metadata.cpu_tail_guard["reason"]


def test_tail_guard_block_start_pattern():
    compiled, x, y, z = _run_add(block_start_tail_guard_kernel, 70)
    assert torch.equal(z, x + y)
    assert compiled.metadata.cpu_tail_guard["matched"]


def test_tail_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TRITON_CPU_TAIL_GUARD", "0")
    compiled, x, y, z = _run_add(disabled_tail_guard_kernel, 64)
    assert torch.equal(z, x + y)
    assert not hasattr(compiled, "main_kernel")
    assert not hasattr(compiled.metadata, "cpu_tail_guard")


def test_tail_guard_env_change_invalidates_jit_cache(monkeypatch):
    compiled, x, y, z = _run_add(add_tail_guard_kernel, 64)
    assert torch.equal(z, x + y)
    assert compiled.metadata.cpu_tail_guard["matched"]

    monkeypatch.setenv("TRITON_CPU_TAIL_GUARD", "0")
    compiled, x, y, z = _run_add(add_tail_guard_kernel, 64)
    assert torch.equal(z, x + y)
    assert not hasattr(compiled.metadata, "cpu_tail_guard")
    assert not hasattr(compiled, "main_kernel")


def test_tail_guard_kwarg_can_disable():
    n = 64
    block = 16
    x = torch.arange(n, dtype=torch.float32, device="cpu")
    y = torch.arange(n, dtype=torch.float32, device="cpu") * 2
    z = torch.empty_like(x)
    grid = (triton.cdiv(n, block), )
    compiled = add_tail_guard_kernel[grid](x, y, z, n, BLOCK=block, enable_tail_guard=False)
    assert torch.equal(z, x + y)
    assert not hasattr(compiled, "main_kernel")
    assert not hasattr(compiled.metadata, "cpu_tail_guard")
