import math

SEQ_PAGE_COST = 1.0
CPU_TUPLE_COST = 0.01
BLOCK = 8192
TUPLE_OVERHEAD = 28
EPS = 1e-3
WRAPPERS = {"Hash", "Materialize", "Memoize", "Sort", "Incremental Sort", "Gather",
            "Gather Merge", "Unique", "Result"}

def scan_cost(rows, width):
    rows = max(float(rows or 0), 0.0)
    width = max(float(width or 0), 0.0)
    pages = math.ceil(rows * (width + TUPLE_OVERHEAD) / BLOCK) if rows > 0 else 0
    return SEQ_PAGE_COST * pages + CPU_TUPLE_COST * rows

def _lg(x):
    return math.log(max(float(x), EPS))
