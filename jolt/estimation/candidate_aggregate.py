import math

import numpy as np

from .hint_features import plan_feats

def agg_n(ks, prefix, r):
    if not ks:
        return
    pf = [plan_feats(c["plan"]) for c in ks]
    costs = np.array([p["log_cost"] for p in pf])
    imin = int(np.argmin([c["plan"].get("Total Cost", 0) for c in ks]))
    for k in pf[imin]:
        r[f"{prefix}min_{k}"] = pf[imin][k]
    r[f"{prefix}cost_min"] = float(costs.min())
    r[f"{prefix}cost_mean"] = float(costs.mean())
    r[f"{prefix}cost_std"] = float(costs.std())
    r[f"{prefix}argmin_is_top1"] = float(imin == 0)
    r[f"{prefix}adhere_mean"] = float(np.mean([c["adhere_canon"] for c in ks]))
    r[f"{prefix}max_join_rows_std"] = float(np.std([p["log_max_join_rows"] for p in pf]))
    r[f"{prefix}n_cands"] = float(len(ks))
    fix = []
    for c in ks:
        vs, rm = c.get("vsub_est_rows"), c.get("rows_mv")
        fac = min(max(rm / vs, 0.01), 100.0) if (vs and rm) else 1.0
        fix.append(math.log(max(c["plan"].get("Total Cost", 0), 1)) + math.log(fac))
    fix = np.array(fix)
    r[f"{prefix}cost_fix_min"] = float(fix.min())
    r[f"{prefix}cost_fix_std"] = float(fix.std())
    r[f"{prefix}argmin_fix_is_top1"] = float(int(np.argmin(fix)) == 0)
    r[f"{prefix}agree_raw_fix_argmin"] = float(int(np.argmin(fix)) == imin)
