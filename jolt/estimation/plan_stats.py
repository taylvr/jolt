import math

EPS = 1e-6

def lg(x):
    return math.log(max(float(x or 0), EPS))

def walk(n):
    yield n
    for c_ in n.get("Plans", []):
        yield from walk(c_)

def plan_stats(plan, prefix):
    nodes = list(walk(plan))
    joins = [n for n in nodes if "Join" in n.get("Node Type", "")]
    join_rows = [n.get("Plan Rows", 0) for n in joins]
    scans = [n for n in nodes if "Scan" in n.get("Node Type", "") and "Relation Name" in n]
    return {
        f"{prefix}log_cost": lg(plan.get("Total Cost", 0)),
        f"{prefix}log_root_rows": lg(plan.get("Plan Rows", 0)),
        f"{prefix}log_startup_cost": lg(plan.get("Startup Cost", 0)),
        f"{prefix}n_nodes": float(len(nodes)),
        f"{prefix}n_nl": float(sum(1 for n in nodes if n["Node Type"] == "Nested Loop")),
        f"{prefix}n_hash": float(sum(1 for n in nodes if n["Node Type"] == "Hash Join")),
        f"{prefix}n_merge": float(sum(1 for n in nodes if n["Node Type"] == "Merge Join")),
        f"{prefix}n_sort": float(sum(1 for n in nodes if n["Node Type"] == "Sort")),
        f"{prefix}n_agg": float(sum(1 for n in nodes if "Aggregate" in n["Node Type"]
                                    or n["Node Type"] == "GroupAggregate")),
        f"{prefix}n_memoize": float(sum(1 for n in nodes if n["Node Type"] == "Memoize")),
        f"{prefix}n_materialize": float(sum(1 for n in nodes if n["Node Type"] == "Materialize")),
        f"{prefix}n_scans": float(len(scans)),
        f"{prefix}n_idx_scans": float(sum(1 for n in scans if "Index" in n["Node Type"])),
        f"{prefix}n_bitmap": float(sum(1 for n in scans if "Bitmap" in n["Node Type"])),
        f"{prefix}log_max_join_rows": lg(max(join_rows)) if join_rows else 0.0,
        f"{prefix}log_sum_join_rows": lg(sum(join_rows)) if join_rows else 0.0,
        f"{prefix}log_max_scan_rows": lg(max((n.get("Plan Rows", 0) for n in scans), default=0)),
        f"{prefix}log_sum_scan_cost": lg(sum(n.get("Total Cost", 0) for n in scans)),
        f"{prefix}log_max_width": lg(max((n.get("Plan Width", 0) for n in scans), default=0)),
    }
