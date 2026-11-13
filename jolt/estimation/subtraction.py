import math

import numpy as np

from .cost_model import scan_cost, _lg
from .plan_walk import find_view_part

def subtraction(plan, v_aliases, c_scan, mv_name=None, prefix="SUB_x_"):
    root = float(plan.get("Total Cost", 0) or 0)
    top, core, inner, kind = find_view_part(plan, v_aliases, mv_name)
    if top is None:
        nan = float("nan")
        return {f"{prefix}found": 0.0, f"{prefix}log_comp": nan, f"{prefix}log_sigma": nan,
                f"{prefix}view_share": nan, f"{prefix}inner_nl": 0.0}
    view_cost = float(top.get("Total Cost", 0) or 0)
    comp = max(root - view_cost, 0.0)
    return {f"{prefix}found": 1.0 if kind == "subtree" else 0.5,
            f"{prefix}log_comp": _lg(comp),
            f"{prefix}log_sigma": _lg(comp + c_scan),
            f"{prefix}view_share": view_cost / root if root > 0 else float("nan"),
            f"{prefix}inner_nl": float(inner)}

def sub_row(q_plan, v_plan, sur_plan, v_aliases, mv_name=None):
    c_scan = scan_cost(v_plan.get("Plan Rows", 0), v_plan.get("Plan Width", 0))
    r = {"SUB_log_scan": _lg(c_scan)}
    inh = subtraction(q_plan, v_aliases, c_scan, None, "SUB_inh_")
    r.update({k: inh[k] for k in ("SUB_inh_found", "SUB_inh_log_comp", "SUB_inh_log_sigma",
                                  "SUB_inh_view_share")})
    if sur_plan is not None:
        hat = subtraction(sur_plan, v_aliases, c_scan, mv_name, "SUB_hat_")
    else:
        hat = {k.replace("SUB_x_", "SUB_hat_"): float("nan") for k in
               ("SUB_x_found", "SUB_x_log_comp", "SUB_x_log_sigma", "SUB_x_view_share", "SUB_x_inner_nl")}
    r.update(hat)
    r["SUB_delta"] = r["SUB_hat_log_sigma"] - r["SUB_inh_log_sigma"]
    return r

SUB_COLS = ["SUB_log_scan", "SUB_inh_found", "SUB_inh_log_comp", "SUB_inh_log_sigma",
            "SUB_inh_view_share", "SUB_hat_found", "SUB_hat_log_comp", "SUB_hat_log_sigma",
            "SUB_hat_view_share", "SUB_hat_inner_nl", "SUB_delta"]

