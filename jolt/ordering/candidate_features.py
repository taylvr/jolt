from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence

from ..core import plan_tree as pt
from ..core import sql_parse as sp

JOB_TABLES = [
    "aka_name", "aka_title", "cast_info", "char_name", "comp_cast_type", "company_name", "company_type",
    "complete_cast", "info_type", "keyword", "kind_type", "link_type", "movie_companies", "movie_info",
    "movie_info_idx", "movie_keyword", "movie_link", "name", "person_info", "role_type", "title",
]
TABLE_VOCAB = ["<unk>", "<mv>"] + JOB_TABLES
SCAN_VOCAB = ["<none>", "<mv>", "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan", "<other>"]
JOIN_VOCAB = ["<none>", "Nested Loop", "Hash Join", "Merge Join", "<other>"]
NODE_TYPE_VOCAB = [
    "<other>", "Nested Loop", "Hash Join", "Merge Join", "Seq Scan", "Index Scan", "Index Only Scan",
    "Bitmap Heap Scan", "Bitmap Index Scan", "Hash", "Memoize", "Materialize", "Aggregate", "Gather",
    "Gather Merge", "Sort", "Incremental Sort", "Limit", "Result", "Append", "Unique",
]

def vocab_index(vocab: Sequence[str], x: Optional[str], other: str = "<other>") -> int:
    if x in vocab:
        return vocab.index(x)
    if other in vocab:
        return vocab.index(other)
    return 0

def l1p(x) -> float:
    try:
        return math.log1p(max(float(x), 0.0))
    except (TypeError, ValueError):
        return 0.0

LEAF_NUM_FEATURES = [
    "is_mv",
    "log_est_rows",
    "log_act_rows",
    "log_loops",
    "log_scan_cost",
    "log_width",
    "is_param",
    "has_filter",
    "n_filter_terms",
    "has_const_index_cond",
    "qbr_pos_norm",
    "qbr_side",
    "q_join_log_est",
    "q_join_log_act",
    "q_fanout_est",
    "q_fanout_act",
    "q_join_side",
    "degree",
    "n_pk_edges",
    "n_fk_edges",
    "adj_mv",
    "mv_log_v_act",
    "mv_log_v_est",
    "mv_log_v_cost",
    "mv_n_tables",
    "mv_log_qsub_est",
    "mv_log_qsub_act",
    "mv_miss_est",
    "prior_pos_norm",
    "prior_side",
]
LEAF_CAT_FEATURES = ["table", "scan", "q_join_type"]

_ALIAS_REF = re.compile(r"\b([A-Za-z_]\w*)\.")

def _cond_refs_other(cond: Optional[str], alias: str) -> bool:
    if not cond:
        return False
    return any(a != alias for a in _ALIAS_REF.findall(cond))

def _n_terms(cond: Optional[str]) -> int:
    if not cond:
        return 0
    return cond.count(" AND ") + cond.count(" OR ") + 1

def _scan_features(scan: dict, alias: str, use_actual: bool) -> Dict[str, float]:
    idx_cond = scan.get("Index Cond") or scan.get("Recheck Cond")
    if scan.get("Node Type") == "Bitmap Heap Scan":
        for c in scan.get("Plans", []) or []:
            idx_cond = idx_cond or c.get("Index Cond")
    is_param = _cond_refs_other(idx_cond, alias)
    f = {
        "log_est_rows": l1p(scan.get("Plan Rows", 0)),
        "log_act_rows": l1p(scan.get("Actual Rows", 0)) if use_actual else 0.0,
        "log_loops": l1p(scan.get("Actual Loops", 1)) if use_actual else 0.0,
        "log_scan_cost": l1p(scan.get("Total Cost", 0)),
        "log_width": l1p(scan.get("Plan Width", 0)),
        "is_param": float(is_param),
        "has_filter": float(bool(scan.get("Filter"))),
        "n_filter_terms": float(_n_terms(scan.get("Filter"))),
        "has_const_index_cond": float(bool(idx_cond) and not is_param),
    }
    return f

def _join_features(q_plan: dict, aliases, use_actual: bool) -> Dict[str, float]:
    r = pt.join_node_adding(q_plan, aliases)
    if r is None:
        return {"q_join_log_est": 0.0, "q_join_log_act": 0.0, "q_fanout_est": 0.0, "q_fanout_act": 0.0,
                "q_join_side": 0.0, "_join_type": "<none>"}
    jn, own, sib, side = r
    je, se = l1p(jn.get("Plan Rows", 0)), l1p(sib.get("Plan Rows", 0))
    ja = l1p(jn.get("Actual Rows", 0) * jn.get("Actual Loops", 1))
    sa = l1p(sib.get("Actual Rows", 0) * sib.get("Actual Loops", 1))
    return {
        "q_join_log_est": je,
        "q_join_log_act": ja if use_actual else 0.0,
        "q_fanout_est": je - se,
        "q_fanout_act": (ja - sa) if use_actual else 0.0,
        "q_join_side": float(side),
        "_join_type": jn.get("Node Type", "<other>"),
    }

def compute_leaf_features(smp, use_actual: bool = True) -> None:
    q_plan = smp.q_plan["Plan"]
    v_root = smp.v_plan["Plan"]
    n = len(smp.candidates)
    qbr_pos = {a: i for i, a in enumerate(smp.qbr_seq)}
    qbr_side = {a: s for a, s in zip(smp.qbr_seq, smp.qbr_sides)}

    for a in smp.candidates:
        f = {k: 0.0 for k in LEAF_NUM_FEATURES}
        cats = {}
        if a == smp.mv:
            v_act = float(v_root.get("Actual Rows", 0)) * float(v_root.get("Actual Loops", 1))
            if not use_actual:
                v_act = float(v_root.get("Plan Rows", 0))
            qsub = pt.topmost_node_with_aliases(q_plan, set(smp.v_aliases))
            qsub_est = float(qsub.get("Plan Rows", 0)) if qsub else 0.0
            qsub_act = float(qsub.get("Actual Rows", 0)) * float(qsub.get("Actual Loops", 1)) if qsub else 0.0
            f.update({
                "is_mv": 1.0,
                "log_est_rows": l1p(v_act),
                "log_act_rows": l1p(v_act) if use_actual else 0.0,
                "log_loops": 0.0,
                "log_scan_cost": 0.0,
                "log_width": l1p(v_root.get("Plan Width", 0)),
                "mv_log_v_act": l1p(v_act),
                "mv_log_v_est": l1p(v_root.get("Plan Rows", 0)),
                "mv_log_v_cost": l1p(v_root.get("Total Cost", 0)),
                "mv_n_tables": float(len(smp.v_aliases)),
                "mv_log_qsub_est": l1p(qsub_est),
                "mv_log_qsub_act": l1p(qsub_act) if use_actual else 0.0,
                "mv_miss_est": l1p(v_act) - l1p(qsub_est),
            })
            jf = _join_features(q_plan, smp.v_aliases, use_actual)
            cats["table"] = "<mv>"
            cats["scan"] = "<mv>"
        else:
            scan = pt.scan_node(q_plan, a)
            f.update(_scan_features(scan, a, use_actual))
            jf = _join_features(q_plan, a, use_actual)
            cats["table"] = smp.q_alias2table.get(a, "<unk>")
            cats["scan"] = scan.get("Node Type", "<other>")
        cats["q_join_type"] = jf.pop("_join_type")
        f.update(jf)
        f["qbr_pos_norm"] = qbr_pos.get(a, 0) / max(n - 1, 1)
        f["qbr_side"] = float(qbr_side.get(a, 0))
        if smp.prior_seq:
            f["prior_pos_norm"] = smp.prior_seq.index(a) / max(n - 1, 1)
            f["prior_side"] = float(smp.prior_sides[smp.prior_seq.index(a)])
        nb = smp.graph.get(a, {})
        f["degree"] = float(len(nb))
        n_pk = n_fk = 0
        for b, cols in nb.items():
            for (ca, cb) in cols:
                if sp.is_pk_column(ca):
                    n_pk += 1
                else:
                    n_fk += 1
        f["n_pk_edges"], f["n_fk_edges"] = float(n_pk), float(n_fk)
        f["adj_mv"] = float(smp.mv in nb)
        smp.leaf_feats[a] = f
        smp.leaf_cats[a] = cats

STEP_FEATURE_NAMES = [
    "k", "k_norm", "n", "n_left",
    "n_edges_to_S", "cand_pk_into_S", "S_pk_into_cand",
    "qbr_rank", "qbr_is_next", "qbr_rank_norm",
    "mv_in_S", "prev_is_mv", "edge_to_prev",
    "card_rank", "card_minus_min", "card_minus_S",
    "S_in_q", "Sc_in_q", "Sc_q_log_est", "Sc_q_growth",
    "is_first_pick_min_card",
    "r_geqo", "q_geqo", "card_minus_mv",
    "has_prior", "prior_rank", "prior_is_next",
]

GEQO_THRESHOLD = 12

def step_features(smp, chosen: List[str], legal: List[str]) -> Dict[str, List[float]]:
    k = len(chosen)
    n = smp.n
    S = set(chosen)
    remaining_qbr = [a for a in smp.qbr_seq if a not in S]
    rank = {a: i for i, a in enumerate(remaining_qbr)}
    cards = {c: smp.leaf_feats[c]["log_est_rows"] for c in legal}
    min_card = min(cards.values()) if cards else 0.0
    sorted_c = sorted(legal, key=lambda c: (cards[c], c))
    card_rank = {c: i for i, c in enumerate(sorted_c)}
    S_exp = smp.expand(S) if S else frozenset()
    S_in_q = float(S_exp in smp.q_node_sets) if S else 0.0
    S_est = smp.q_node_sets.get(S_exp, (None, None))[0] if S else None
    prev = chosen[-1] if chosen else None
    if smp.prior_seq:
        rem_prior = [a for a in smp.prior_seq if a not in S]
        prank = {a: i for i, a in enumerate(rem_prior)}
    else:
        prank = None

    out = {}
    for c in legal:
        nb = smp.graph.get(c, {})
        n_edges = sum(1 for b in nb if b in S)
        cand_pk = S_pk = 0.0
        for b, cols in nb.items():
            if b in S:
                for (cc, cb) in cols:
                    if sp.is_pk_column(cc):
                        cand_pk = 1.0
                    if sp.is_pk_column(cb):
                        S_pk = 1.0
        Sc_exp = smp.expand(S | {c})
        sc = smp.q_node_sets.get(Sc_exp)
        sc_est = l1p(sc[0]) if sc else -1.0
        growth = (sc_est - l1p(S_est)) if (sc is not None and S_est is not None) else 0.0
        out[c] = [
            float(k), k / max(n - 1, 1), float(n), float(n - k),
            float(n_edges), cand_pk, S_pk,
            float(rank.get(c, n)), float(rank.get(c, n) == 0), rank.get(c, n) / max(n - k, 1),
            float(smp.mv in S), float(prev == smp.mv), float(prev in nb if prev else 0.0),
            float(card_rank[c]), cards[c] - min_card, (cards[c] - l1p(S_est)) if S_est is not None else 0.0,
            S_in_q, float(sc is not None), sc_est, growth,
            float(k == 0 and card_rank[c] == 0),
            float(n >= GEQO_THRESHOLD), float(len(smp.q_alias2table) >= GEQO_THRESHOLD),
            cards[c] - smp.leaf_feats[smp.mv]["log_est_rows"],
            float(prank is not None),
            float(prank.get(c, n)) if prank is not None else 0.0,
            float(prank.get(c, n) == 0) if prank is not None else 0.0,
        ]
    return out

def leaf_vector(smp, c: str) -> List[float]:
    f = smp.leaf_feats[c]
    return [f[k] for k in LEAF_NUM_FEATURES]

def leaf_cat_ids(smp, c: str) -> List[int]:
    cats = smp.leaf_cats[c]
    return [
        vocab_index(TABLE_VOCAB, cats["table"], "<unk>"),
        vocab_index(SCAN_VOCAB, cats["scan"], "<other>"),
        vocab_index(JOIN_VOCAB, cats["q_join_type"], "<other>"),
    ]

NODE_NUM_FEATURES = ["log_plan_rows", "log_total_cost", "log_startup_cost", "log_width",
                     "log_act_rows", "log_loops", "has_filter", "n_filter_terms", "has_join_filter",
                     "is_param", "is_outer_child", "is_inner_child"]

def node_vector(node: dict, use_actual: bool = True) -> List[float]:
    alias = node.get("Alias")
    cond = node.get("Index Cond") or node.get("Recheck Cond")
    rel = node.get("Parent Relationship")
    return [
        l1p(node.get("Plan Rows", 0)), l1p(node.get("Total Cost", 0)), l1p(node.get("Startup Cost", 0)),
        l1p(node.get("Plan Width", 0)),
        l1p(node.get("Actual Rows", 0)) if use_actual else 0.0,
        l1p(node.get("Actual Loops", 1)) if use_actual else 0.0,
        float(bool(node.get("Filter"))), float(_n_terms(node.get("Filter"))),
        float(bool(node.get("Join Filter") or node.get("Hash Cond") or node.get("Merge Cond"))),
        float(_cond_refs_other(cond, alias) if alias else False),
        float(rel == "Outer"), float(rel == "Inner"),
    ]

def node_cat_ids(node: dict, alias2table: Optional[Dict[str, str]] = None) -> List[int]:
    nt = vocab_index(NODE_TYPE_VOCAB, node.get("Node Type"), "<other>")
    rel = node.get("Relation Name")
    if rel is None and alias2table and node.get("Alias") in alias2table:
        rel = alias2table[node["Alias"]]
    tab = vocab_index(TABLE_VOCAB, rel, "<unk>") if rel else 0
    return [nt, tab]
