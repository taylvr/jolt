import json
import os

import pandas as pd

from .hint_features import with_extra, plan_feats
from .candidate_aggregate import agg_n

N_MAX = 5
HERE = os.path.dirname(os.path.abspath(__file__))
EXPLAIN_DIR = os.environ.get("EXPLAIN_DIR", os.path.join(HERE, "data", "rbq_explain"))
BEAMS = os.environ.get("BEAMS", os.path.join(HERE, "data", "beams.json"))
OUT = os.environ.get("HB_OUT", os.path.join(HERE, "data", "hb_features.csv"))

def cand(rid, tag):
    p = os.path.join(EXPLAIN_DIR, f"{rid}__{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None

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

def build():
    beams = json.load(open(BEAMS))
    rows = []
    for rid in sorted(beams):
        cands = {tag: cand(rid, tag) for tag in
                 [f"k{i}" for i in range(N_MAX)] + ["qbr", "true"]}
        if cands.get("k0") is None:
            continue
        r = dict(rid=rid)
        for k, v in with_extra(cands["k0"]).items():
            r[f"HB_{k}"] = v
        if cands.get("true"):
            for k, v in with_extra(cands["true"]).items():
                r[f"HBT_{k}"] = v
        if cands.get("qbr"):
            f1, fq = plan_feats(cands["k0"]["plan"]), plan_feats(cands["qbr"]["plan"])
            for k in f1:
                r[f"HBD_{k}"] = f1[k] - fq[k]
            r["HBD_adhere"] = float(cands["qbr"]["adhere_canon"])
        for n in range(1, N_MAX + 1):
            ks = [cands[f"k{i}"] for i in range(n) if cands.get(f"k{i}")]
            agg_n(ks, f"HBK{n}_", r)
        rows.append(r)
    df = pd.DataFrame(rows).fillna(0.0)
    df.to_csv(OUT, index=False)
    print(f"hb features: {len(df)} rows x {len(df.columns)} cols -> {OUT}")
    for g in ["HB_", "HBT_", "HBD_", "HBK1_", "HBK5_"]:
        cols = [c for c in df.columns if c.startswith(g)]
        print(f"  {g.rstrip('_')}: {len(cols)} cols")
    return df

if __name__ == "__main__":
    build()
