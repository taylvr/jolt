from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

JoinTree = Union[str, tuple]

def load_plan_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    return data

def plan_to_tree(plan: dict) -> Optional[JoinTree]:
    if "Alias" in plan:
        return plan["Alias"]
    subs = []
    for child in plan.get("Plans", []) or []:
        t = plan_to_tree(child)
        if t is not None:
            subs.append(t)
    if not subs:
        return None
    if len(subs) == 1:
        return subs[0]
    if len(subs) > 2:
        logger.warning("node %s has %d children; left-folding", plan.get("Node Type"), len(subs))
    t = subs[0]
    for s in subs[1:]:
        t = (t, s)
    return t

def tree_to_str(t: JoinTree) -> str:
    if isinstance(t, str):
        return t
    return "(" + " ".join(tree_to_str(c) for c in t) + ")"

def str_to_tree(s: str) -> JoinTree:
    tokens = s.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            items = []
            while tokens[pos] != ")":
                items.append(parse())
            pos += 1
            if len(items) == 1:
                return items[0]
            t = items[0]
            for it in items[1:]:
                t = (t, it)
            return t
        pos += 1
        return tok

    out = parse()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens in {s!r}")
    return out

def leaves(t: JoinTree) -> List[str]:
    if isinstance(t, str):
        return [t]
    out: List[str] = []
    for c in t:
        out.extend(leaves(c))
    return out

def subtrees(t: JoinTree) -> List[JoinTree]:
    if isinstance(t, str):
        return [t]
    out = [t]
    for c in t:
        out.extend(subtrees(c))
    return out

def canon(t: JoinTree) -> JoinTree:
    if isinstance(t, str):
        return t
    a, b = canon(t[0]), canon(t[1])
    return (a, b) if tree_to_str(a) <= tree_to_str(b) else (b, a)

def is_linear(t: JoinTree) -> bool:
    if isinstance(t, str):
        return True
    a, b = t
    if not isinstance(a, str) and not isinstance(b, str):
        return False
    return is_linear(a) and is_linear(b)

def replace_subtree_by_leafset(t: JoinTree, leafset: set, name: str) -> Tuple[JoinTree, bool]:
    if set(leaves(t)) == leafset:
        return name, True
    if isinstance(t, str):
        return t, False
    a, ok_a = replace_subtree_by_leafset(t[0], leafset, name)
    if ok_a:
        return (a, t[1]), True
    b, ok_b = replace_subtree_by_leafset(t[1], leafset, name)
    return (t[0], b), ok_b

def find_subtree_by_leafset(t: JoinTree, leafset: set) -> Optional[JoinTree]:
    for s in subtrees(t):
        if set(leaves(s)) == leafset:
            return s
    return None

def expand_leaf(t: JoinTree, name: str, sub: JoinTree) -> JoinTree:
    if isinstance(t, str):
        return sub if t == name else t
    return (expand_leaf(t[0], name, sub), expand_leaf(t[1], name, sub))

def tree_to_seq(t: JoinTree) -> Tuple[List[str], List[int], bool]:
    if isinstance(t, str):
        return [t], [0], True
    a, b = t
    if isinstance(a, str) and isinstance(b, str):
        return [a, b], [0, 0], True
    if isinstance(b, str):
        seq, sides, ex = tree_to_seq(a)
        return seq + [b], sides + [0], ex
    if isinstance(a, str):
        seq, sides, ex = tree_to_seq(b)
        return seq + [a], sides + [1], ex
    la, lb = leaves(a), leaves(b)
    big, small = (a, b) if len(la) >= len(lb) else (b, a)
    seq, sides, _ = tree_to_seq(big)
    s2, _, _ = tree_to_seq(small)
    return seq + s2, sides + [0] * len(s2), False

def seq_to_tree(seq: Sequence[str], sides: Optional[Sequence[int]] = None) -> JoinTree:
    if not seq:
        raise ValueError("empty seq")
    if sides is None:
        sides = [0] * len(seq)
    t: JoinTree = seq[0]
    for k in range(1, len(seq)):
        if k >= 2 and sides[k] == 1:
            t = (seq[k], t)
        else:
            t = (t, seq[k])
    return t

def hint_leading(t: JoinTree) -> str:
    return "/*+  Leading(" + tree_to_str(t) + ") */"

def iter_nodes(plan: dict):
    yield plan
    for c in plan.get("Plans", []) or []:
        yield from iter_nodes(c)

def node_aliases(plan: dict) -> set:
    if "Alias" in plan:
        return {plan["Alias"]}
    s = set()
    for c in plan.get("Plans", []) or []:
        s |= node_aliases(c)
    return s

def annotate_plan(plan: dict) -> Dict[int, dict]:
    info: Dict[int, dict] = {}

    def rec(n, parent):
        nid = len(info)
        info[nid] = {"node": n, "parent": parent, "children": [], "aliases": set()}
        if parent is not None:
            info[parent]["children"].append(nid)
        if "Alias" in n:
            info[nid]["aliases"] = {n["Alias"]}
            for c in n.get("Plans", []) or []:
                rec(c, nid)
            return info[nid]["aliases"]
        s = set()
        for c in n.get("Plans", []) or []:
            s |= rec(c, nid)
        info[nid]["aliases"] = s
        return s

    rec(plan, None)
    return info

def topmost_node_with_aliases(plan: dict, aliases: set) -> Optional[dict]:
    for n in iter_nodes(plan):
        if "Alias" in n:
            if {n["Alias"]} == aliases:
                return n
            continue
        if node_aliases(n) == aliases:
            return n
    return None

def scan_node(plan: dict, alias: str) -> Optional[dict]:
    for n in iter_nodes(plan):
        if n.get("Alias") == alias:
            return n
    return None

def join_node_adding(plan: dict, alias) -> Optional[Tuple[dict, dict, dict, int]]:
    target = {alias} if isinstance(alias, str) else set(alias)
    for n in iter_nodes(plan):
        if "Alias" in n:
            continue
        kids = [c for c in (n.get("Plans", []) or []) if node_aliases(c)]
        if len(kids) != 2:
            continue
        for idx, c in enumerate(kids):
            if node_aliases(c) == target:
                return n, c, kids[1 - idx], (0 if idx == 1 else 1)
    return None

def legacy_extract_join_order_with_parentheses(plan: dict) -> str:
    def traverse_plan(node):
        if 'Alias' in node:
            return node['Alias']
        if 'Plans' in node:
            sub_expressions = []
            for sub_plan in node['Plans']:
                result = traverse_plan(sub_plan)
                if result:
                    sub_expressions.append(result)
            if len(sub_expressions) == 1:
                return sub_expressions[0]
            if sub_expressions:
                return "(" + " ".join(sub_expressions) + ")"
        return ""
    return traverse_plan(plan)

def legacy_extract_substructures(structure: str) -> List[str]:
    stack, substructures = [], []
    for char in structure:
        if char == '(':
            stack.append('')
        elif char == ')':
            sub_expr = stack.pop()
            if stack:
                stack[-1] += '(' + sub_expr + ')'
            substructures.append(sub_expr)
        elif char is not None:
            if stack:
                stack[-1] += char
    return ["(" + sub + ")" for sub in substructures]
