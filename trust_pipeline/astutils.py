"""AST helpers: function under test, probe inputs from the visible test, realistic bug mutations."""
from __future__ import annotations

import ast
import builtins
import copy
import random
import warnings


def parse_quietly(src):
    """ast.parse without SyntaxWarnings (e.g. invalid escapes like '\\d' common in MBPP code)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(src)


def expected_call(test_src):
    """First call in the assert whose target is a non-builtin bare name: the function under test."""
    try:
        tree = parse_quietly(test_src)
    except SyntaxError:
        return None, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and not hasattr(builtins, node.func.id):
            return node.func.id, node
    return None, None


class _InputPerturber(ast.NodeTransformer):
    def __init__(self, rng):
        self.rng = rng

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, bool) or v is None:
            return node
        if isinstance(v, int):
            new = v + self.rng.choice([-2, -1, 1, 2, 3])
        elif isinstance(v, float):
            new = round(v + self.rng.choice([-1.5, 0.5, 2.0]), 3)
        elif isinstance(v, str) and len(v) > 1:
            chars = list(v)
            self.rng.shuffle(chars)
            new = "".join(chars)
        else:
            return node
        return ast.copy_location(ast.Constant(new), node)

    def visit_List(self, node):
        self.generic_visit(node)
        if len(node.elts) > 1:
            r = self.rng.random()
            if r < 0.33:
                node.elts = node.elts[::-1]
            elif r < 0.66:
                node.elts = node.elts[:-1]
        return node


def make_probes(visible_test, n=6, seed=42):
    """Probe expressions: the visible call plus perturbed copies of its arguments."""
    _, call = expected_call(visible_test)
    if call is None:
        return []
    probes = [ast.unparse(call)]
    rng = random.Random(f"{seed}-{visible_test}")
    for _ in range(n * 5):
        if len(probes) >= n:
            break
        new_call = copy.deepcopy(call)
        perturber = _InputPerturber(rng)
        new_call.args = [perturber.visit(a) for a in new_call.args]
        new_call.keywords = [perturber.visit(k) for k in new_call.keywords]
        expr = ast.unparse(ast.fix_missing_locations(new_call))
        if expr not in probes:
            probes.append(expr)
    return probes


_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
             ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv,
             ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}


def _mutation_sites(tree):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(type(o) in _CMP_SWAP for o in node.ops):
            sites.append(("flip_comparison", node))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
            sites.append(("swap_operator", node))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites.append(("off_by_one", node))
        elif isinstance(node, ast.BoolOp):
            sites.append(("swap_and_or", node))
        elif isinstance(node, ast.Return) and node.value is not None:
            sites.append(("drop_return_value", node))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and not hasattr(builtins, node.id):
            sites.append(("misspell_name", node))
    return sites


def mutate_code(code_str, rng):
    try:
        tree = parse_quietly(code_str)
    except SyntaxError:
        return None, None
    sites = _mutation_sites(tree)
    if not sites:
        return None, None
    kind, node = rng.choice(sites)
    if kind == "flip_comparison":
        i = next(i for i, o in enumerate(node.ops) if type(o) in _CMP_SWAP)
        node.ops[i] = _CMP_SWAP[type(node.ops[i])]()
    elif kind == "swap_operator":
        node.op = _BIN_SWAP[type(node.op)]()
    elif kind == "off_by_one":
        node.value = node.value + rng.choice([-1, 1])
    elif kind == "swap_and_or":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "drop_return_value":
        node.value = ast.Constant(None)
    elif kind == "misspell_name":
        node.id = node.id + "_"
    return ast.unparse(tree), kind


def make_mutants(code_str, n, seed):
    """Up to n distinct single-bug mutants of code_str, as (code, mutation_type) pairs."""
    rng = random.Random(seed)
    seen, out = {ast.unparse(parse_quietly(code_str))}, []
    for _ in range(n * 5):
        if len(out) >= n:
            break
        mutant, kind = mutate_code(code_str, rng)
        if mutant and mutant not in seen:
            seen.add(mutant)
            out.append((mutant, kind))
    return out
