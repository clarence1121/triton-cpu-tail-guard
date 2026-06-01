"""
General-purpose 2D pid-region dispatch framework.

Abstractions
------------
RegionSpec   -- one region: (name, condition_expr, c_launcher_name)
RegionPlan   -- the full dispatch plan for a kernel: pid_vars + list[RegionSpec]

Supported condition expressions
--------------------------------
Simple pairwise comparisons between two pid variables:
    pid_a OP pid_b   where OP ∈ {<, <=, >, >=, ==, !=}

The constraint solver decides, given that the *region* condition R holds,
whether a *statement* condition S in the kernel body is:
  - always True  → inline the if-branch  (dead else-branch / guard)
  - always False → inline the else-branch (dead if-branch)
  - unknown      → leave the statement unchanged

Usage
-----
Annotate a JITFunction at definition time::

    from triton.compiler.cpu_pid_region import RegionPlan, RegionSpec, region_dispatch

    causal_plan = RegionPlan(
        pid_vars=["pid_q", "pid_k"],
        regions=[
            RegionSpec("full", "pid_k < pid_q",  "region_2d_lt"),
            RegionSpec("diag", "pid_k == pid_q", "region_2d_eq"),
        ],
    )

    @region_dispatch(causal_plan)
    @triton.jit
    def _kernel(Q, K, S, ...):
        pid_q = tl.program_id(0)
        pid_k = tl.program_id(1)
        if pid_k > pid_q:          # dead for full, dead for diag → removed
            return
        ...
        if pid_k == pid_q:         # dead for full → removed; tautology for diag → inlined
            row_idx = ...
            acc = tl.where(...)
        tl.store(...)

When the kernel is compiled the first time, the compiler:
1. Parses the kernel body
2. For each active region (non-None launcher), generates a branch-free variant
3. Compiles each variant to a separate binary
4. Wraps all binaries in a RegionCompiledKernel that dispatches via C-level launchers

Backward compatibility
----------------------
The old auto-detection path (enable_pid_region_dispatch=True) still works.
analyze_pid_region() now returns a PidRegionAnalysis whose plan() method builds
a RegionPlan in the new format, so _do_compile can use the same generic code path.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Any

# Op codes for run_omp_region_2d (pidY OP pidX, i.e. pid_axis1 OP pid_axis0)
REGION_2D_LT = 0   # y < x  (lower triangle, pid_axis1 < pid_axis0)
REGION_2D_LE = 1   # y <= x
REGION_2D_EQ = 2   # y == x (diagonal)
REGION_2D_GE = 3   # y >= x
REGION_2D_GT = 4   # y > x  (upper triangle)
REGION_2D_NE = 5   # y != x

# Canonical launcher name for each op (RegionSpec.launcher values)
_OP_TO_LAUNCHER = {
    '<':  "region_2d_lt",
    '<=': "region_2d_le",
    '==': "region_2d_eq",
    '>=': "region_2d_ge",
    '>':  "region_2d_gt",
    '!=': "region_2d_ne",
}


# ---------------------------------------------------------------------------
# Public API: RegionSpec and RegionPlan
# ---------------------------------------------------------------------------

@dataclass
class RegionSpec:
    """One region in a 2D pid-space dispatch plan.

    Parameters
    ----------
    name:     identifier, used for debug messages and cache keys
    condition: Python expression string that holds for every pid in this region,
               e.g. ``"pid_k < pid_q"`` or ``"pid_k == pid_q"``.
               Currently only simple pairwise comparisons are supported for
               constraint-based simplification; other conditions are kept verbatim.
    launcher: name of the C-level launch method on CompiledKernel, e.g.
              ``"lower_tri_2d"`` → ``CompiledKernel.run_lower_tri_2d``.
              Pass ``None`` to mark this region as *invalid* (never dispatched).
    """
    name: str
    condition: str
    launcher: str | None   # None = skip / invalid region

    def cache_key(self) -> str:
        return f"{self.name}:{self.condition}:{self.launcher}"


@dataclass
class RegionPlan:
    """Complete dispatch plan for a Triton kernel with pid-based specialization.

    Parameters
    ----------
    pid_vars: names of the pid variables in axis order,
              e.g. ``["pid_q", "pid_k"]`` for a 2-D grid.
    regions:  list of RegionSpec, ordered by dispatch priority.
              Regions with ``launcher=None`` are never launched (invalid tiles).
    """
    pid_vars: list[str]
    regions: list[RegionSpec]

    @property
    def active_regions(self) -> list[RegionSpec]:
        """Regions that actually get dispatched (launcher is not None)."""
        return [r for r in self.regions if r.launcher is not None]

    @property
    def skip_regions(self) -> list[RegionSpec]:
        """Invalid / skip regions (launcher is None)."""
        return [r for r in self.regions if r.launcher is None]

    def cache_key(self) -> str:
        parts = [",".join(self.pid_vars)] + [r.cache_key() for r in self.regions]
        return "|".join(parts)


def region_dispatch(plan: RegionPlan):
    """Decorator that attaches a RegionPlan to a JITFunction.

    Usage::

        @region_dispatch(my_plan)
        @triton.jit
        def _kernel(...): ...
    """
    def decorator(fn):
        fn._cpu_region_plan = plan
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Constraint solver for simple pairwise pid comparisons
# ---------------------------------------------------------------------------
#
# Supported condition grammar:
#   expr  ::= NAME OP NAME
#   OP    ::= '<' | '<=' | '>' | '>=' | '==' | '!='
#
# The solver handles conditions of the form "pid_a OP pid_b" where both
# identifiers appear in pid_vars.  For other condition shapes it returns None
# (unknown).

# Implication table: given region_op holds between (a, b),
# does stmt_op also hold between (a, b)?
#   True  → stmt is tautology under region condition
#   False → stmt is contradiction under region condition
#   None  → unknown
_IMPLIES: dict[str, dict[str, bool | None]] = {
    '<': {'<': True,  '<=': True,  '==': False, '>=': False, '>': False, '!=': True},
    '>': {'<': False, '<=': False, '==': False, '>=': True,  '>': True,  '!=': True},
    '==': {'<': False, '<=': True, '==': True,  '>=': True,  '>': False, '!=': False},
    '<=': {'<': None, '<=': True,  '==': None,  '>=': None,  '>': False, '!=': None},
    '>=': {'<': False, '<=': None, '==': None,  '>=': True,  '>': None,  '!=': None},
    '!=': {'<': None, '<=': None,  '==': False, '>=': None,  '>': None,  '!=': None},
}

# Complement operators: OP → ¬OP
_COMPLEMENT = {'<': '>=', '<=': '>', '==': '!=', '!=': '==', '>': '<=', '>=': '<'}

# Mirror operators: swapping lhs and rhs: a OP b ↔ b MIRROR(OP) a
_MIRROR = {'<': '>', '<=': '>=', '==': '==', '!=': '!=', '>': '<', '>=': '<='}


def _parse_simple_comparison(expr_str: str, pid_vars: list[str]) -> tuple[str, str, str] | None:
    """Parse "pid_a OP pid_b" and return (lhs_name, op_str, rhs_name), or None."""
    try:
        tree = ast.parse(expr_str.strip(), mode='eval')
    except SyntaxError:
        return None
    node = tree.body
    if not isinstance(node, ast.Compare):
        return None
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return None
    lhs = node.left.id if isinstance(node.left, ast.Name) else None
    rhs = node.comparators[0].id if isinstance(node.comparators[0], ast.Name) else None
    if lhs is None or rhs is None:
        return None
    if lhs not in pid_vars or rhs not in pid_vars:
        return None
    op = node.ops[0]
    op_map = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
              ast.Eq: '==', ast.NotEq: '!='}
    op_str = op_map.get(type(op))
    if op_str is None:
        return None
    return lhs, op_str, rhs


def _normalize(lhs: str, op: str, rhs: str, canonical_lhs: str, canonical_rhs: str
               ) -> tuple[str, str, str] | None:
    """Re-express (lhs OP rhs) with canonical order (canonical_lhs OP' canonical_rhs).
    Returns None if the variables don't match the canonical pair."""
    if lhs == canonical_lhs and rhs == canonical_rhs:
        return lhs, op, rhs
    if lhs == canonical_rhs and rhs == canonical_lhs:
        return canonical_lhs, _MIRROR[op], canonical_rhs
    return None


def eval_condition(stmt_cond_str: str, region_cond_str: str, pid_vars: list[str]) -> bool | None:
    """Determine if *stmt_cond* is always True or always False given *region_cond*.

    Returns
    -------
    True  if region_cond implies stmt_cond
    False if region_cond implies ¬stmt_cond
    None  if unknown (cannot determine, leave statement unchanged)
    """
    region_parsed = _parse_simple_comparison(region_cond_str, pid_vars)
    stmt_parsed = _parse_simple_comparison(stmt_cond_str, pid_vars)
    if region_parsed is None or stmt_parsed is None:
        return None

    r_lhs, r_op, r_rhs = region_parsed
    s_lhs, s_op, s_rhs = stmt_parsed

    # Bring both to the same canonical order
    canonical = (r_lhs, r_rhs)
    r_norm = _normalize(r_lhs, r_op, r_rhs, *canonical)
    s_norm = _normalize(s_lhs, s_op, s_rhs, *canonical)
    if r_norm is None or s_norm is None:
        return None

    _, r_op_n, _ = r_norm
    _, s_op_n, _ = s_norm
    return _IMPLIES.get(r_op_n, {}).get(s_op_n, None)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _attr_name(node: ast.AST) -> str | None:
    return node.attr if isinstance(node, ast.Attribute) else None


def _is_call_to(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and _attr_name(node.func) == name


def _name(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _constant_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _uses_any_name(node: ast.AST, names: set[str]) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in names:
            return True
    return False


def _ast_unparse_condition(test: ast.expr) -> str | None:
    """Try to unparse an AST test node to a condition string."""
    try:
        return ast.unparse(test)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generic variant generator
# ---------------------------------------------------------------------------

def make_region_variant(fn, src, region_spec: RegionSpec, all_specs: list[RegionSpec],
                        pid_vars: list[str]) -> tuple[ast.Module, bool]:
    """Generate a branch-free variant of *fn* specialized for *region_spec*.

    For every ``if <cond>: <body> [else: <else_body>]`` statement in the
    kernel body that involves pid variables:

    - If *region_spec.condition* implies *cond* is **always True**:
      replace the if-statement with ``<body>`` (inline true branch).
    - If *region_spec.condition* implies *cond* is **always False**:
      replace with ``<else_body>`` (inline else branch, possibly empty).
    - Otherwise: leave unchanged.

    Returns (new_module_ast, ok) where ok=False if the kernel body could
    not be parsed (pass-through with no transformation applied).
    """
    tree = fn.parse()
    if not isinstance(tree, ast.Module) or len(tree.body) != 1 \
            or not isinstance(tree.body[0], ast.FunctionDef):
        return tree, False

    fn_def = copy.deepcopy(tree.body[0])
    pid_name_set = set(pid_vars)

    def _simplify_stmts(stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ast.If) and _uses_any_name(stmt.test, pid_name_set):
                cond_str = _ast_unparse_condition(stmt.test)
                result = eval_condition(cond_str, region_spec.condition, pid_vars) if cond_str else None
                if result is True:
                    # condition always holds → inline the if-body
                    out.extend(copy.deepcopy(stmt.body))
                elif result is False:
                    # condition never holds → inline the else-body (may be empty)
                    out.extend(copy.deepcopy(stmt.orelse))
                else:
                    out.append(copy.deepcopy(stmt))
            else:
                out.append(copy.deepcopy(stmt))
        return out

    fn_def.body = _simplify_stmts(fn_def.body)
    new_tree = ast.Module(body=[fn_def], type_ignores=[])
    ast.fix_missing_locations(new_tree)
    return new_tree, True


# ---------------------------------------------------------------------------
# Auto-detection path (backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class PidRegionDiagnostic:
    requested: bool
    matched: bool
    reason: str
    pid0_name: str | None = None
    pid1_name: str | None = None
    invalid_guard_idx: int | None = None
    diag_cond_idx: int | None = None

    def asdict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "matched": self.matched,
            "reason": self.reason,
            "pid0_name": self.pid0_name,
            "pid1_name": self.pid1_name,
            "invalid_guard_idx": self.invalid_guard_idx,
            "diag_cond_idx": self.diag_cond_idx,
        }


@dataclass
class PidRegionAnalysis:
    tree: ast.Module
    fn_def: ast.FunctionDef | None
    diagnostic: PidRegionDiagnostic

    def to_region_plan(self) -> RegionPlan | None:
        """Convert auto-detection result to a RegionPlan (if matched)."""
        if not self.diagnostic.matched:
            return None
        p0 = self.diagnostic.pid0_name
        p1 = self.diagnostic.pid1_name
        return RegionPlan(
            pid_vars=[p0, p1],
            regions=[
                RegionSpec("full", f"{p1} < {p0}", "region_2d_lt"),
                RegionSpec("diag", f"{p1} == {p0}", "region_2d_eq"),
            ],
        )


def _find_program_id_assigns(fn: ast.FunctionDef) -> dict[int, str]:
    result: dict[int, str] = {}
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = _name(stmt.targets[0])
        if target is None or not _is_call_to(stmt.value, "program_id"):
            continue
        call = stmt.value
        if len(call.args) == 1:
            axis = _constant_int(call.args[0])
            if axis is not None:
                result[axis] = target
    return result


def _is_invalid_guard(stmt: ast.stmt, pid0: str, pid1: str) -> bool:
    if not isinstance(stmt, ast.If) or stmt.orelse:
        return False
    if len(stmt.body) != 1 or not isinstance(stmt.body[0], ast.Return):
        return False
    if stmt.body[0].value is not None:
        return False
    test = stmt.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = _name(test.left)
    right = _name(test.comparators[0])
    op = test.ops[0]
    return (left == pid1 and right == pid0 and isinstance(op, ast.Gt)) or \
           (left == pid0 and right == pid1 and isinstance(op, ast.Lt))


def _is_diag_condition(stmt: ast.stmt, pid0: str, pid1: str) -> bool:
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = _name(test.left)
    right = _name(test.comparators[0])
    op = test.ops[0]
    if not isinstance(op, ast.Eq):
        return False
    return (left == pid0 and right == pid1) or (left == pid1 and right == pid0)


def analyze_pid_region(fn, src) -> PidRegionAnalysis:
    """Auto-detect the causal-attention pid-region pattern.

    Looks for:
    - ``pid_q = tl.program_id(0)`` and ``pid_k = tl.program_id(1)``
    - An upper-triangle guard ``if pid_k > pid_q: return``
    - A diagonal condition ``if pid_k == pid_q: <body>``
    """
    tree = fn.parse()
    if not isinstance(tree, ast.Module) or len(tree.body) != 1 \
            or not isinstance(tree.body[0], ast.FunctionDef):
        return PidRegionAnalysis(tree, None,
                                 PidRegionDiagnostic(True, False, "expected a single function definition"))

    fn_def = copy.deepcopy(tree.body[0])
    pid_assigns = _find_program_id_assigns(fn_def)

    if 0 not in pid_assigns:
        return PidRegionAnalysis(tree, None,
                                 PidRegionDiagnostic(True, False, "missing pid0 = tl.program_id(0)"))
    if 1 not in pid_assigns:
        return PidRegionAnalysis(tree, None,
                                 PidRegionDiagnostic(True, False, "missing pid1 = tl.program_id(1)"))

    pid0 = pid_assigns[0]
    pid1 = pid_assigns[1]
    pid_names = {pid0, pid1}

    invalid_guard_idx: int | None = None
    diag_cond_idx: int | None = None

    for idx, stmt in enumerate(fn_def.body):
        if invalid_guard_idx is None and _is_invalid_guard(stmt, pid0, pid1):
            invalid_guard_idx = idx
            continue
        if diag_cond_idx is None and _is_diag_condition(stmt, pid0, pid1):
            diag_cond_idx = idx
            continue
        if isinstance(stmt, ast.If) and _uses_any_name(stmt.test, pid_names):
            return PidRegionAnalysis(
                tree, None,
                PidRegionDiagnostic(True, False,
                                    f"unrecognized pid-conditional at statement {idx}",
                                    pid0_name=pid0, pid1_name=pid1))

    if invalid_guard_idx is None:
        return PidRegionAnalysis(
            tree, None,
            PidRegionDiagnostic(True, False,
                                f"missing invalid guard (if {pid1} > {pid0}: return)",
                                pid0_name=pid0, pid1_name=pid1))

    if diag_cond_idx is None:
        return PidRegionAnalysis(
            tree, None,
            PidRegionDiagnostic(True, False,
                                f"missing diagonal condition (if {pid1} == {pid0}: ...)",
                                pid0_name=pid0, pid1_name=pid1))

    diag = PidRegionDiagnostic(True, True, "matched",
                               pid0_name=pid0, pid1_name=pid1,
                               invalid_guard_idx=invalid_guard_idx,
                               diag_cond_idx=diag_cond_idx)
    return PidRegionAnalysis(tree, fn_def, diag)


# ---------------------------------------------------------------------------
# Legacy variant helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def make_pid_region_full_variant(fn, src) -> tuple[ast.Module, PidRegionDiagnostic]:
    """Full-tile variant for lower-triangle dispatch (pid1 < pid0). Legacy entry point."""
    analysis = analyze_pid_region(fn, src)
    if not analysis.diagnostic.matched:
        return analysis.tree, analysis.diagnostic
    plan = analysis.to_region_plan()
    full_spec = next(r for r in plan.active_regions if r.name == "full")
    tree, _ = make_region_variant(fn, src, full_spec, plan.regions, plan.pid_vars)
    return tree, analysis.diagnostic


def make_pid_region_diag_variant(fn, src) -> tuple[ast.Module, PidRegionDiagnostic]:
    """Diagonal-tile variant for diagonal dispatch (pid1 == pid0). Legacy entry point."""
    analysis = analyze_pid_region(fn, src)
    if not analysis.diagnostic.matched:
        return analysis.tree, analysis.diagnostic
    plan = analysis.to_region_plan()
    diag_spec = next(r for r in plan.active_regions if r.name == "diag")
    tree, _ = make_region_variant(fn, src, diag_spec, plan.regions, plan.pid_vars)
    return tree, analysis.diagnostic
