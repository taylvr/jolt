import json
import os
import sys

import numpy as np

from ..core import plan_tree as pt
from ..core import sql_parse as sp
from ..core.sample import derive_v_aliases
from .sql_rewrite import rewrite
from .plan_substitution import replace_leaf
from .explain_backend import explain_json
from .compare import cmp_trees
from .view_rows import vsub_rows

PKG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MVJO_ROOT = os.environ.get("MVJO_ROOT", os.path.join(PKG, "data"))
D = os.path.join(MVJO_ROOT, "data", "exec_json")
CVP = os.path.join(MVJO_ROOT, "outputs", "gbdt", "cv_predictions.json")

def main(n_samples=15):
    cvp = json.load(open(CVP))
    rids = sorted(cvp)[::max(1, len(cvp) // n_samples)][:n_samples]
    results = []
    for rid in rids:
        qid, vid = rid.split("_")[0], rid.split("_")[-1]
        try:
            q_sql = open(os.path.join(D, "q", qid + ".sql")).read()
            qj = json.load(open(os.path.join(D, "q_json", qid + ".json")))
            vj = json.load(open(os.path.join(D, "v_json", vid + ".json")))
            v_sql = open(os.path.join(D, "v", vid + ".sql")).read()
            r_sql = open(os.path.join(D, "r", rid + ".sql")).read()
        except FileNotFoundError:
            continue
        a2t = sp.from_aliases(q_sql)
        q_tree = pt.plan_to_tree(qj["Plan"])
        mv = f"mv{vid}"
        v_aliases, method = derive_v_aliases(a2t, q_tree, vj["Plan"], v_sql, r_sql, mv, qj["Plan"])
        v_sub = pt.find_subtree_by_leafset(q_tree, set(v_aliases))

        rows_mv = vj["Plan"].get("Actual Rows", 0) or 0
        rec = dict(rid=rid, n=len(a2t), v_method=method, rows_mv=rows_mv)

        qbr_tree = replace_leaf(pt.str_to_tree(cvp[rid]["qbr"]), mv, v_sub)
        true_tree = replace_leaf(pt.str_to_tree(cvp[rid]["true"]), mv, v_sub)
        seq_true, sides_true, _ = pt.tree_to_seq(true_tree)
        rev_tree = pt.seq_to_tree(list(reversed(seq_true)), [0] * len(seq_true))

        for name, tgt in [("qbr", qbr_tree), ("true", true_tree), ("rev", rev_tree)]:
            try:
                plan, dt = explain_json(rewrite(q_sql, tgt, a2t))
                got = pt.plan_to_tree(plan)
                c = cmp_trees(got, tgt) if got is not None else dict(exact=0, canon=0, seq=0)
                rec[name] = c
                rec[name + "_ms"] = round(dt * 1000, 1)
                if name == "true":
                    est = vsub_rows(plan, v_aliases)
                    rec["vsub_est_rows"] = est
            except Exception as e:
                rec[name] = dict(exact=0, canon=0, seq=0, err=str(e)[:120])
        results.append(rec)
        print(f"{rid} n={rec['n']:2d} | qbr {rec['qbr']} | true {rec['true']} | "
              f"rev {rec['rev']} | {rec.get('true_ms')}ms", flush=True)

    def agg(k, f):
        vals = [r[k][f] for r in results if isinstance(r.get(k), dict) and f in r[k]]
        return np.mean(vals) if vals else float("nan")

    print("\n===== summary (%d samples) =====" % len(results))
    print(f"{'target':6s} {'exact':>7s} {'canon':>7s} {'seq':>7s}")
    for k in ["qbr", "true", "rev"]:
        print(f"{k:6s} {agg(k,'exact'):7.2f} {agg(k,'canon'):7.2f} {agg(k,'seq'):7.2f}")
    ms = [r["true_ms"] for r in results if "true_ms" in r]
    print(f"EXPLAIN wall time: median {np.median(ms):.0f}ms max {np.max(ms):.0f}ms")
    lr = [np.log(max(r['vsub_est_rows'], 1) / max(r['rows_mv'], 1))
          for r in results if r.get("vsub_est_rows") and r.get("rows_mv")]
    if lr:
        print(f"vsub est rows vs rows_mv: log ratio median {np.median(lr):+.2f} "
              f"P90|log ratio| {np.quantile(np.abs(lr), 0.9):.2f}")

    out = os.path.join(FUSION, "data", "collapse_test_results.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)
    print("saved ->", out)

if __name__ == "__main__":
    main()
