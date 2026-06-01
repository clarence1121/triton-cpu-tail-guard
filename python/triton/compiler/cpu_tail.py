from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class TailGuardDiagnostic:
    requested: bool
    matched: bool
    reason: str
    mask_name: str | None = None
    offsets_name: str | None = None
    n_name: str | None = None
    block_name: str | None = None

    def asdict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "matched": self.matched,
            "reason": self.reason,
            "mask_name": self.mask_name,
            "offsets_name": self.offsets_name,
            "n_name": self.n_name,
            "block_name": self.block_name,
        }


@dataclass
class TailGuardAnalysis:
    tree: ast.Module
    fn_def: ast.FunctionDef | None
    diagnostic: TailGuardDiagnostic
    mask_idx: int | None = None


def _attr_name(node: ast.AST) -> str | None:
    return node.attr if isinstance(node, ast.Attribute) else None


def _is_call_to(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and _attr_name(node.func) == name


def _name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _constant_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0


def _find_program_id_name(fn: ast.FunctionDef) -> str | None:
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = _name(stmt.targets[0])
        if target is None or not _is_call_to(stmt.value, "program_id"):
            continue
        call = stmt.value
        if len(call.args) == 1 and _constant_zero(call.args[0]):
            return target
    return None


def _match_pid_times_block(node: ast.AST, pid_name: str) -> str | None:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return None
    lhs = _name(node.left)
    rhs = _name(node.right)
    if lhs == pid_name and rhs is not None:
        return rhs
    if rhs == pid_name and lhs is not None:
        return lhs
    return None


def _match_arange_block(node: ast.AST) -> str | None:
    if not _is_call_to(node, "arange"):
        return None
    call = node
    if len(call.args) != 2 or not _constant_zero(call.args[0]):
        return None
    return _name(call.args[1])


def _find_scaled_pid_vars(fn: ast.FunctionDef, pid_name: str) -> dict[str, str]:
    scaled: dict[str, str] = {}
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = _name(stmt.targets[0])
        if target is None:
            continue
        block = _match_pid_times_block(stmt.value, pid_name)
        if block is not None:
            scaled[target] = block
    return scaled


def _match_offsets_expr(node: ast.AST, pid_name: str, scaled_vars: dict[str, str]) -> str | None:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return None
    pairs = ((node.left, node.right), (node.right, node.left))
    for base, arange in pairs:
        arange_block = _match_arange_block(arange)
        if arange_block is None:
            continue
        base_block = _match_pid_times_block(base, pid_name)
        if base_block is None:
            base_name = _name(base)
            base_block = scaled_vars.get(base_name) if base_name else None
        if base_block == arange_block:
            return arange_block
    return None


def _find_offsets(fn: ast.FunctionDef, pid_name: str) -> tuple[str, str] | None:
    scaled_vars = _find_scaled_pid_vars(fn, pid_name)
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = _name(stmt.targets[0])
        if target is None:
            continue
        block = _match_offsets_expr(stmt.value, pid_name, scaled_vars)
        if block is not None:
            return target, block
    return None


def _find_mask(fn: ast.FunctionDef, offsets_name: str) -> tuple[int, str, str] | None:
    for idx, stmt in enumerate(fn.body):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = _name(stmt.targets[0])
        value = stmt.value
        if target is None or not isinstance(value, ast.Compare):
            continue
        if len(value.ops) != 1 or not isinstance(value.ops[0], ast.Lt):
            continue
        if _name(value.left) != offsets_name or len(value.comparators) != 1:
            continue
        n_name = _name(value.comparators[0])
        if n_name is not None:
            return idx, target, n_name
    return None


def _is_offsets_pointer(node: ast.AST, offsets_name: str) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    return _name(node.left) == offsets_name or _name(node.right) == offsets_name


class _MemoryAndMaskChecker(ast.NodeVisitor):

    def __init__(self, mask_name: str, offsets_name: str):
        self.mask_name = mask_name
        self.offsets_name = offsets_name
        self.memory_ops = 0
        self.ok = True
        self.reason = ""

    def _reject(self, reason: str):
        if self.ok:
            self.ok = False
            self.reason = reason

    def visit_Call(self, node: ast.Call):
        callee = _attr_name(node.func)
        if callee in ("load", "store"):
            self.memory_ops += 1
            if not node.args:
                self._reject("memory operation has no pointer argument")
                return
            if not _is_offsets_pointer(node.args[0], self.offsets_name):
                self._reject("memory pointer is not base + offsets")
                return
            mask_kw = next((kw for kw in node.keywords if kw.arg == "mask"), None)
            if mask_kw is None or _name(mask_kw.value) != self.mask_name:
                self._reject("memory operation does not use the detected mask")
                return
        else:
            self._reject("kernel body contains non-memory function calls")
            return
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg == "mask" and _name(node.value) == self.mask_name:
            return
        self.visit(node.value)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id == self.mask_name:
            self._reject("detected mask is used outside tl.load/tl.store mask keywords")


class _RemoveMaskKeywords(ast.NodeTransformer):

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if _attr_name(node.func) in ("load", "store"):
            node.keywords = [kw for kw in node.keywords if kw.arg not in ("mask", "other")]
        return node


def _has_unsupported_stmt(stmts: list[ast.stmt]) -> bool:
    unsupported = (ast.For, ast.While, ast.If, ast.Try, ast.With, ast.Match, ast.Return, ast.Yield, ast.YieldFrom)
    return any(isinstance(stmt, unsupported) for stmt in stmts)


def _constexpr_names(fn, src) -> set[str]:
    names: set[str] = set()
    for key in getattr(src, "constants", {}):
        if len(key) == 1:
            names.add(fn.arg_names[key[0]])
    for name, ty in getattr(src, "signature", {}).items():
        if ty == "constexpr":
            names.add(name)
    return names


def analyze_cpu_tail(fn, src) -> TailGuardAnalysis:
    tree = fn.parse()
    if not isinstance(tree, ast.Module) or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return TailGuardAnalysis(tree, None, TailGuardDiagnostic(True, False, "expected a single function definition"))

    fn_def = copy.deepcopy(tree.body[0])
    pid_name = _find_program_id_name(fn_def)
    if pid_name is None:
        return TailGuardAnalysis(tree, None, TailGuardDiagnostic(True, False, "missing pid = tl.program_id(0)"))

    offsets = _find_offsets(fn_def, pid_name)
    if offsets is None:
        return TailGuardAnalysis(
            tree, None, TailGuardDiagnostic(True, False, "missing offsets = pid * BLOCK + tl.arange(0, BLOCK)"))
    offsets_name, block_name = offsets

    if block_name not in _constexpr_names(fn, src):
        return TailGuardAnalysis(
            tree, None,
            TailGuardDiagnostic(True, False, "BLOCK is not constexpr", offsets_name=offsets_name,
                                block_name=block_name))

    mask = _find_mask(fn_def, offsets_name)
    if mask is None:
        return TailGuardAnalysis(
            tree, None,
            TailGuardDiagnostic(True, False, "missing mask = offsets < n", offsets_name=offsets_name,
                                block_name=block_name))
    mask_idx, mask_name, n_name = mask

    tail_body = fn_def.body[mask_idx + 1:]
    if _has_unsupported_stmt(tail_body):
        return TailGuardAnalysis(
            tree, None,
            TailGuardDiagnostic(True, False, "kernel body after mask contains unsupported control flow",
                                mask_name=mask_name, offsets_name=offsets_name, n_name=n_name, block_name=block_name))

    checker = _MemoryAndMaskChecker(mask_name, offsets_name)
    for stmt in tail_body:
        checker.visit(stmt)
    if not checker.ok:
        return TailGuardAnalysis(
            tree, None,
            TailGuardDiagnostic(True, False, checker.reason, mask_name=mask_name, offsets_name=offsets_name,
                                n_name=n_name, block_name=block_name))
    if checker.memory_ops == 0:
        return TailGuardAnalysis(
            tree, None,
            TailGuardDiagnostic(True, False, "no masked memory operations found", mask_name=mask_name,
                                offsets_name=offsets_name, n_name=n_name, block_name=block_name))

    diagnostic = TailGuardDiagnostic(True, True, "matched", mask_name=mask_name, offsets_name=offsets_name,
                                     n_name=n_name, block_name=block_name)
    return TailGuardAnalysis(tree, fn_def, diagnostic, mask_idx)


# ---------------------------------------------------------------------------
# 2-D tail guard (matmul-style: two pid axes, two independent masks)
# ---------------------------------------------------------------------------


@dataclass
class TailGuard2DDiagnostic:
    matched: bool
    reason: str
    m_n_name: str | None = None
    m_block_name: str | None = None
    n_n_name: str | None = None
    n_block_name: str | None = None
    offs_m_name: str | None = None
    offs_n_name: str | None = None
    # K loop peeling (optional — 2D tail guard works without it)
    k_loop_found: bool = False
    k_name: str | None = None
    k_n_name: str | None = None
    k_block_name: str | None = None

    def asdict(self):
        return self.__dict__.copy()


@dataclass
class TailGuard2DAnalysis:
    tree: ast.Module
    fn_def: ast.FunctionDef | None
    diagnostic: TailGuard2DDiagnostic


def _find_program_id_name_axis(fn: ast.FunctionDef, axis: int) -> str | None:
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = _name(stmt.targets[0])
        if target is None or not _is_call_to(stmt.value, "program_id"):
            continue
        call = stmt.value
        if len(call.args) == 1:
            arg = call.args[0]
            if isinstance(arg, ast.Constant) and arg.value == axis:
                return target
    return None


def _find_bound_for_offsets(fn_def: ast.FunctionDef, offsets_name: str) -> str | None:
    """Scan entire function (including loops) for: offsets[...] < N or offsets < N."""
    for node in ast.walk(fn_def):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Lt):
            continue
        left = node.left
        base = _name(left)
        if base is None and isinstance(left, ast.Subscript):
            base = _name(left.value)
        if base == offsets_name:
            bound = _name(node.comparators[0])
            if bound is not None:
                return bound
    return None


def _simplify_mask_expr(expr: ast.AST, offs_m: str, offs_n: str,
                        keep_m: bool, keep_n: bool) -> ast.AST | None:
    """Recursively simplify an AND-tree mask.  Return None → remove entirely."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitAnd):
        left = _simplify_mask_expr(expr.left, offs_m, offs_n, keep_m, keep_n)
        right = _simplify_mask_expr(expr.right, offs_m, offs_n, keep_m, keep_n)
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        result = copy.copy(expr)
        result.left = left
        result.right = right
        return result
    if isinstance(expr, ast.Compare):
        refs_m = any(isinstance(n, ast.Name) and n.id == offs_m for n in ast.walk(expr))
        refs_n = any(isinstance(n, ast.Name) and n.id == offs_n for n in ast.walk(expr))
        if refs_m and not keep_m:
            return None
        if refs_n and not keep_n:
            return None
    return expr


class _SimplifyMask2D(ast.NodeTransformer):

    def __init__(self, offs_m: str, offs_n: str, keep_m: bool, keep_n: bool):
        self.offs_m = offs_m
        self.offs_n = offs_n
        self.keep_m = keep_m
        self.keep_n = keep_n

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if _attr_name(node.func) not in ("load", "store"):
            return node
        mask_kw = next((kw for kw in node.keywords if kw.arg == "mask"), None)
        if mask_kw is None:
            return node
        new_mask = _simplify_mask_expr(mask_kw.value, self.offs_m, self.offs_n,
                                       self.keep_m, self.keep_n)
        if new_mask is None:
            node.keywords = [kw for kw in node.keywords if kw.arg not in ("mask", "other")]
        else:
            mask_kw.value = new_mask
        return node


def _body_has_k_mask(stmts: list[ast.stmt], k_name: str) -> bool:
    """True if any tl.load/store mask in stmts references the loop variable k_name."""
    for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if not _is_call_to(node, 'load') and not _is_call_to(node, 'store'):
            continue
        mask_kw = next((kw for kw in node.keywords if kw.arg == 'mask'), None)
        if mask_kw is None:
            continue
        if any(isinstance(n, ast.Name) and n.id == k_name for n in ast.walk(mask_kw.value)):
            return True
    return False


def _find_k_loop(fn_def: ast.FunctionDef, constexprs: set[str]) -> tuple[str, str, str] | None:
    """Find: for k in range(0, K, BLOCK_K) where BLOCK_K is constexpr and
    the loop body has at least one K-dependent mask."""
    for node in ast.walk(fn_def):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        k_name = node.target.id
        it = node.iter
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and
                it.func.id == 'range' and len(it.args) == 3):
            continue
        start, stop, step = it.args
        if not (isinstance(start, ast.Constant) and start.value == 0):
            continue
        k_n_name = _name(stop)
        k_block_name = _name(step)
        if k_n_name is None or k_block_name is None:
            continue
        if k_block_name not in constexprs:
            continue
        if _body_has_k_mask(node.body, k_name):
            return k_name, k_n_name, k_block_name
    return None


def analyze_cpu_tail_2d(fn, src) -> TailGuard2DAnalysis:
    tree = fn.parse()
    if not isinstance(tree, ast.Module) or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return TailGuard2DAnalysis(tree, None, TailGuard2DDiagnostic(False, "expected single function def"))

    fn_def = copy.deepcopy(tree.body[0])

    pid_m = _find_program_id_name_axis(fn_def, 0)
    if pid_m is None:
        return TailGuard2DAnalysis(tree, None, TailGuard2DDiagnostic(False, "missing pid_m = tl.program_id(0)"))

    pid_n = _find_program_id_name_axis(fn_def, 1)
    if pid_n is None:
        return TailGuard2DAnalysis(tree, None, TailGuard2DDiagnostic(False, "missing pid_n = tl.program_id(1)"))

    offs_m_result = _find_offsets(fn_def, pid_m)
    if offs_m_result is None:
        return TailGuard2DAnalysis(tree, None,
                                   TailGuard2DDiagnostic(False, "missing offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)"))
    offs_m_name, m_block_name = offs_m_result

    offs_n_result = _find_offsets(fn_def, pid_n)
    if offs_n_result is None:
        return TailGuard2DAnalysis(tree, None,
                                   TailGuard2DDiagnostic(False, "missing offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)"))
    offs_n_name, n_block_name = offs_n_result

    constexprs = _constexpr_names(fn, src)
    if m_block_name not in constexprs:
        return TailGuard2DAnalysis(tree, None, TailGuard2DDiagnostic(False, f"{m_block_name} is not constexpr"))
    if n_block_name not in constexprs:
        return TailGuard2DAnalysis(tree, None, TailGuard2DDiagnostic(False, f"{n_block_name} is not constexpr"))

    m_n_name = _find_bound_for_offsets(fn_def, offs_m_name)
    if m_n_name is None:
        return TailGuard2DAnalysis(tree, None,
                                   TailGuard2DDiagnostic(False, f"cannot find M bound for {offs_m_name}"))

    n_n_name = _find_bound_for_offsets(fn_def, offs_n_name)
    if n_n_name is None:
        return TailGuard2DAnalysis(tree, None,
                                   TailGuard2DDiagnostic(False, f"cannot find N bound for {offs_n_name}"))

    k_loop = _find_k_loop(fn_def, constexprs)
    k_loop_found = k_loop is not None
    k_name = k_n_name_k = k_block_name_k = None
    if k_loop:
        k_name, k_n_name_k, k_block_name_k = k_loop

    diag = TailGuard2DDiagnostic(
        matched=True, reason="matched",
        m_n_name=m_n_name, m_block_name=m_block_name,
        n_n_name=n_n_name, n_block_name=n_block_name,
        offs_m_name=offs_m_name, offs_n_name=offs_n_name,
        k_loop_found=k_loop_found,
        k_name=k_name, k_n_name=k_n_name_k, k_block_name=k_block_name_k,
    )
    return TailGuard2DAnalysis(tree, fn_def, diag)


def make_cpu_tail_2d_variant(variant: str, fn, src) -> tuple[ast.Module, TailGuard2DDiagnostic]:
    """variant ∈ {"main", "m_tail", "n_tail", "corner"}.  "corner" = original.

    Removes M/N boundary masks for the appropriate variant while leaving the K
    mask intact.  Keeping the K mask is important for performance: LLVM uses the
    `offs_k + k < K` predicate as a vectorization hint, and removing it causes a
    significant regression even for aligned K.
    """
    analysis = analyze_cpu_tail_2d(fn, src)
    if not analysis.diagnostic.matched:
        return analysis.tree, analysis.diagnostic

    fn_def = copy.deepcopy(analysis.fn_def)
    diag = analysis.diagnostic

    keep_m = variant in ("m_tail", "corner")
    keep_n = variant in ("n_tail", "corner")

    fn_def = _SimplifyMask2D(diag.offs_m_name, diag.offs_n_name, keep_m, keep_n).visit(fn_def)

    ast.fix_missing_locations(fn_def)
    new_tree = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(new_tree)
    return new_tree, diag


def make_cpu_tail_main_variant(fn, src) -> tuple[ast.Module, TailGuardDiagnostic]:
    analysis = analyze_cpu_tail(fn, src)
    if not analysis.diagnostic.matched:
        return analysis.tree, analysis.diagnostic

    assert analysis.fn_def is not None
    assert analysis.mask_idx is not None
    fn_def = analysis.fn_def
    mask_idx = analysis.mask_idx
    tail_body = fn_def.body[mask_idx + 1:]

    prefix = copy.deepcopy(fn_def.body[:mask_idx])
    main_body = [_RemoveMaskKeywords().visit(copy.deepcopy(stmt)) for stmt in tail_body]
    fn_def.body = prefix + main_body

    new_tree = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(new_tree)
    return new_tree, analysis.diagnostic
