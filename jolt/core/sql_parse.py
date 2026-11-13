from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

_FROM_RE = re.compile(r"\bFROM\b(.*?)(?:\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|;|\Z)", re.S | re.I)
_EQ_JOIN_RE = re.compile(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*=\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\b")

def read_sql(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql

def from_aliases(sql: str) -> Dict[str, str]:
    sql = strip_comments(sql)
    m = _FROM_RE.search(sql)
    if not m:
        raise ValueError("cannot find FROM clause")
    out: Dict[str, str] = {}
    for item in m.group(1).split(","):
        item = item.strip()
        if not item:
            continue
        parts = [p for p in re.split(r"\s+AS\s+|\s+", item, flags=re.I) if p]
        out[parts[-1]] = parts[0]
    return out

def equi_join_predicates(sql: str, aliases: Optional[Set[str]] = None) -> List[Tuple[str, str, str, str]]:
    sql = strip_comments(sql)
    idx = re.search(r"\bWHERE\b", sql, re.I)
    body = sql[idx.end():] if idx else ""
    preds = []
    for a, x, b, y in _EQ_JOIN_RE.findall(body):
        if a == b:
            continue
        if aliases is not None and (a not in aliases or b not in aliases):
            continue
        preds.append((a, x, b, y))
    return preds

def build_join_graph(preds: List[Tuple[str, str, str, str]], nodes: Set[str],
                     collapse: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    collapse = collapse or {}
    g: Dict[str, Dict[str, List[Tuple[str, str]]]] = {n: {} for n in nodes}
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(u):
        parent.setdefault(u, u)
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for a, x, b, y in preds:
        ra, rb = find((a, x)), find((b, y))
        if ra != rb:
            parent[ra] = rb
    classes: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for u in list(parent):
        classes.setdefault(find(u), []).append(u)
    for members in classes.values():
        for i, (a, x) in enumerate(members):
            for (b, y) in members[i + 1:]:
                a2, b2 = collapse.get(a, a), collapse.get(b, b)
                if a2 == b2 or a2 not in g or b2 not in g:
                    continue
                if (x, y) not in g[a2].setdefault(b2, []):
                    g[a2][b2].append((x, y))
                if (y, x) not in g[b2].setdefault(a2, []):
                    g[b2][a2].append((y, x))
    return g

def is_pk_column(col: str) -> bool:
    return col == "id"

def is_connected(g: Dict[str, Dict], nodes: Set[str]) -> bool:
    nodes = set(nodes)
    if not nodes:
        return True
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in g.get(u, {}):
            if v in nodes and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen == nodes
