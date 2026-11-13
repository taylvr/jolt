import math

EPS = 1e-6

def lg(x):
    return math.log(max(float(x or 0), EPS))

def walk(n):
    yield n
    for c in n.get("Plans", []):
        yield from walk(c)

def plan_feats(plan):
    nodes = list(walk(plan))
    joins = [n for n in nodes if n["Node Type"] in ("Nested Loop", "Hash Join", "Merge Join")]
    join_rows = [n.get("Plan Rows", 0) for n in joins]
    scans = [n for n in nodes if "Scan" in n["Node Type"] and "Relation Name" in n]
    return {
        "log_cost": lg(plan.get("Total Cost", 0)),
        "log_root_rows": lg(plan.get("Plan Rows", 0)),
        "n_nodes": float(len(nodes)),
        "n_nl": float(sum(1 for n in nodes if n["Node Type"] == "Nested Loop")),
        "n_hash": float(sum(1 for n in nodes if n["Node Type"] == "Hash Join")),
        "n_merge": float(sum(1 for n in nodes if n["Node Type"] == "Merge Join")),
        "n_memoize": float(sum(1 for n in nodes if n["Node Type"] == "Memoize")),
        "n_materialize": float(sum(1 for n in nodes if n["Node Type"] == "Materialize")),
        "n_sort": float(sum(1 for n in nodes if n["Node Type"] == "Sort")),
        "log_max_join_rows": lg(max(join_rows)) if join_rows else 0.0,
        "log_sum_join_rows": lg(sum(join_rows)) if join_rows else 0.0,
        "log_max_scan_rows": lg(max((n.get("Plan Rows", 0) for n in scans), default=0)),
        "log_sum_scan_cost": lg(sum(n.get("Total Cost", 0) for n in scans)),
    }

def with_extra(rec):
    f = plan_feats(rec["plan"])
    f["adhere"] = float(rec["adhere_canon"])
    f["mv_scan"] = float(rec.get("mv_scan") or 0)
    f["vsub_mismatch"] = (lg(rec.get("vsub_est_rows")) - lg(rec.get("rows_mv"))
                          if rec.get("vsub_est_rows") else 0.0)
    if rec.get("vsub_est_rows"):
        f["vsub_corr_factor"] = min(max(rec["rows_mv"] / rec["vsub_est_rows"], 0.01), 100.0)
    else:
        f["vsub_corr_factor"] = 1.0
    f["log_cost_fix"] = f["log_cost"] + math.log(f["vsub_corr_factor"])
    f["logp"] = rec.get("logp") if rec.get("logp") is not None else 0.0
    return f
