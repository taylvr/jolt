
from .cost_model import WRAPPERS

def _aliases(n, memo):
    k = id(n)
    if k not in memo:
        if "Alias" in n:
            memo[k] = frozenset([n["Alias"]])
        else:
            s = set()
            for c in n.get("Plans", []) or []:
                s |= _aliases(c, memo)
            memo[k] = frozenset(s)
    return memo[k]

def _walk(n, parent=None, idx=None):
    yield n, parent, idx
    for i, c in enumerate(n.get("Plans", []) or []):
        yield from _walk(c, n, i)

def find_view_part(plan, v_aliases, mv_name=None):
    target = frozenset(v_aliases)
    memo = {}
    for n, parent, idx in _walk(plan):
        if _aliases(n, memo) == target:
            core = n
            while core.get("Node Type") in WRAPPERS and len(core.get("Plans", []) or []) == 1:
                core = core["Plans"][0]
            inner = int(parent is not None and parent.get("Node Type") == "Nested Loop" and idx == 1)
            return n, core, inner, "subtree"
    if mv_name:
        for n, parent, idx in _walk(plan):
            if n.get("Relation Name") == mv_name and "Scan" in n.get("Node Type", ""):
                inner = int(parent is not None and parent.get("Node Type") == "Nested Loop" and idx == 1)
                return n, n, inner, "mv_scan"
    return None, None, 0, None

def scan_cost(rows, width):
    rows = max(float(rows or 0), 0.0)
    width = max(float(width or 0), 0.0)
    pages = math.ceil(rows * (width + TUPLE_OVERHEAD) / BLOCK) if rows > 0 else 0
    return SEQ_PAGE_COST * pages + CPU_TUPLE_COST * rows
