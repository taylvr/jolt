from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from . import plan_tree as pt
from . import sql_parse as sp

logger = logging.getLogger(__name__)

@dataclass
class JOSample:
    rid: str
    qid: str
    vid: str
    mv: str
    template: str
    q_plan: dict
    v_plan: dict
    q_alias2table: Dict[str, str]
    v_aliases: Set[str]
    v_alias_method: str
    q_tree: pt.JoinTree
    qbr_tree: pt.JoinTree
    candidates: List[str]
    graph: Dict[str, Dict[str, List[Tuple[str, str]]]]
    qbr_seq: List[str] = field(default_factory=list)
    qbr_sides: List[int] = field(default_factory=list)
    q_node_sets: Dict[FrozenSet[str], Tuple[float, float]] = field(default_factory=dict)
    leaf_feats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    leaf_cats: Dict[str, Dict[str, str]] = field(default_factory=dict)
    r_plan: Optional[dict] = None
    r_tree: Optional[pt.JoinTree] = None
    target_seq: Optional[List[str]] = None
    target_sides: Optional[List[int]] = None
    r_is_linear: Optional[bool] = None
    rbq_tree: Optional[pt.JoinTree] = None
    prior_tree: Optional[pt.JoinTree] = None
    prior_seq: Optional[List[str]] = None
    prior_sides: Optional[List[int]] = None

    @property
    def n(self) -> int:
        return len(self.candidates)

    def expand(self, aliases) -> FrozenSet[str]:
        out = set()
        for a in aliases:
            if a == self.mv:
                out |= self.v_aliases
            else:
                out.add(a)
        return frozenset(out)

    def neighbors(self, a: str) -> Set[str]:
        return set(self.graph.get(a, {}).keys())

    def legal_candidates(self, chosen: List[str]) -> List[str]:
        rest = [c for c in self.candidates if c not in chosen]
        if not chosen:
            return rest
        S = set(chosen)
        conn = [c for c in rest if self.neighbors(c) & S]
        return conn if conn else rest

def template_of(qid: str) -> str:
    m = re.match(r"(\d+)", qid)
    return m.group(1) if m else qid

def _mv_name(v_sql: Optional[str], vid: str) -> str:
    if v_sql:
        m = re.search(r"MATERIALIZED\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w]+)", v_sql, re.I)
        if m:
            return m.group(1)
    return f"mv{vid}"

def _filter_sig(scan: Optional[dict]) -> str:
    if not scan:
        return ""
    parts = [scan.get("Filter") or ""]
    ic = scan.get("Index Cond") or ""
    if ic and not re.search(r"\b\w+\.\w+", ic):
        parts.append(ic)
    return re.sub(r"\b[A-Za-z_]\w*\.", "", " & ".join(parts))

def derive_v_aliases(q_alias2table: Dict[str, str], q_tree: pt.JoinTree, v_plan: dict,
                     v_sql: Optional[str], r_sql: Optional[str], mv: str,
                     q_plan: Optional[dict] = None) -> Tuple[Set[str], str]:
    if r_sql:
        r_al = sp.from_aliases(r_sql)
        vset = set(q_alias2table) - (set(r_al) - {mv})
        if vset:
            return vset, "r_sql"
    v_tables = Counter(n["Relation Name"] for n in pt.iter_nodes(v_plan["Plan"]) if "Relation Name" in n)
    matches = []
    for s in pt.subtrees(q_tree):
        ls = pt.leaves(s)
        if Counter(q_alias2table[a] for a in ls) == v_tables:
            matches.append(set(ls))
    if matches:
        if len(matches) > 1 and q_plan is not None:
            v_sigs = Counter(_filter_sig(n) for n in pt.iter_nodes(v_plan["Plan"]) if "Relation Name" in n)
            def score(m):
                q_sigs = Counter(_filter_sig(pt.scan_node(q_plan["Plan"], a)) for a in m)
                return sum((q_sigs & v_sigs).values())
            matches.sort(key=score, reverse=True)
            if len(matches) > 1 and score(matches[0]) == score(matches[1]):
                logger.warning("ambiguous V alias match %s; take the first", matches)
        return matches[0], "q_subtree_match"
    tabs = set(v_tables) if v_tables else set(sp.from_aliases(v_sql).values()) if v_sql else set()
    vset = {a for a, t in q_alias2table.items() if t in tabs}
    logger.warning("fall back to table-name mapping for V: %s", vset)
    return vset, "table_name"

def build_sample(q_plan: dict, q_sql: str, v_plan: dict, v_sql: Optional[str], qid: str, vid: str,
                 r_plan: Optional[dict] = None, r_sql: Optional[str] = None,
                 rbq_plan: Optional[dict] = None, use_actual: bool = True,
                 prior_plan: Optional[dict] = None) -> JOSample:
    from .features import compute_leaf_features

    mv = _mv_name(v_sql, vid)
    q_alias2table = sp.from_aliases(q_sql)
    q_tree = pt.plan_to_tree(q_plan["Plan"])
    if set(pt.leaves(q_tree)) != set(q_alias2table):
        raise ValueError(f"Q plan leaves {sorted(pt.leaves(q_tree))} != Q FROM aliases {sorted(q_alias2table)}")

    v_aliases, method = derive_v_aliases(q_alias2table, q_tree, v_plan, v_sql, r_sql, mv, q_plan)
    qbr_tree, ok = pt.replace_subtree_by_leafset(q_tree, v_aliases, mv)
    if not ok:
        logger.warning("%s_%s: V aliases %s are not a subtree of Q; approximating QBR", qid, vid, v_aliases)
        seq, sides, _ = pt.tree_to_seq(q_tree)
        new_seq, new_sides, placed = [], [], False
        for a, s in zip(seq, sides):
            if a in v_aliases:
                if not placed:
                    new_seq.append(mv); new_sides.append(s); placed = True
            else:
                new_seq.append(a); new_sides.append(s)
        qbr_tree = pt.seq_to_tree(new_seq, new_sides)

    candidates = sorted([a for a in q_alias2table if a not in v_aliases] + [mv])
    collapse = {a: mv for a in v_aliases}
    preds = sp.equi_join_predicates(q_sql, set(q_alias2table))
    graph = sp.build_join_graph(preds, set(candidates), collapse)

    qbr_seq, qbr_sides, _ = pt.tree_to_seq(qbr_tree)

    q_node_sets: Dict[FrozenSet[str], Tuple[float, float]] = {}
    for n in pt.iter_nodes(q_plan["Plan"]):
        s = frozenset(pt.node_aliases(n))
        if s and s not in q_node_sets:
            q_node_sets[s] = (float(n.get("Plan Rows", 0.0)),
                              float(n.get("Actual Rows", 0.0)) * float(n.get("Actual Loops", 1.0)))

    smp = JOSample(rid=f"{qid}_{vid}", qid=qid, vid=vid, mv=mv, template=template_of(qid),
                   q_plan=q_plan, v_plan=v_plan, q_alias2table=q_alias2table, v_aliases=v_aliases,
                   v_alias_method=method, q_tree=q_tree, qbr_tree=qbr_tree, candidates=candidates,
                   graph=graph, qbr_seq=qbr_seq, qbr_sides=qbr_sides, q_node_sets=q_node_sets)

    if prior_plan is not None:
        ptree = pt.plan_to_tree(prior_plan["Plan"])
        if set(pt.leaves(ptree)) == set(candidates):
            smp.prior_tree = ptree
            smp.prior_seq, smp.prior_sides, _ = pt.tree_to_seq(ptree)
        else:
            logger.warning("%s: prior plan leaves %s != candidates %s; ignored", smp.rid,
                           sorted(pt.leaves(ptree)), candidates)

    compute_leaf_features(smp, use_actual=use_actual)

    if r_plan is not None:
        r_tree = pt.plan_to_tree(r_plan["Plan"])
        if set(pt.leaves(r_tree)) != set(candidates):
            raise ValueError(f"R leaves {sorted(pt.leaves(r_tree))} != candidates {candidates}")
        seq, sides, exact = pt.tree_to_seq(r_tree)
        smp.r_plan, smp.r_tree = r_plan, r_tree
        smp.target_seq, smp.target_sides, smp.r_is_linear = seq, sides, exact
    if rbq_plan is not None:
        smp.rbq_tree = pt.plan_to_tree(rbq_plan["Plan"])
    return smp

def load_dataset(root: str, use_actual: bool = True, require_r: bool = True,
                 limit: Optional[int] = None, prior_dir: str = "r_json_fill") -> Tuple[List[JOSample], List[dict]]:
    r_dir = os.path.join(root, "r_json")
    rids = sorted(f[:-5] for f in os.listdir(r_dir) if f.endswith(".json"))
    if limit:
        rids = rids[:limit]
    samples, skipped = [], []
    for rid in rids:
        if "_" not in rid:
            skipped.append({"rid": rid, "reason": "bad name"}); continue
        qid, vid = rid.split("_", 1)
        try:
            q_plan = pt.load_plan_json(os.path.join(root, "q_json", f"{qid}.json"))
            v_plan = pt.load_plan_json(os.path.join(root, "v_json", f"{vid}.json"))
            r_plan = pt.load_plan_json(os.path.join(root, "r_json", f"{rid}.json"))
            q_sql = sp.read_sql(os.path.join(root, "q", f"{qid}.sql"))
            v_path = os.path.join(root, "v", f"{vid}.sql")
            v_sql = sp.read_sql(v_path) if os.path.exists(v_path) else None
            r_path = os.path.join(root, "r", f"{rid}.sql")
            r_sql = sp.read_sql(r_path) if os.path.exists(r_path) else None
            rbq_path = os.path.join(root, "rbq_sql_json", f"{rid}.json")
            rbq_plan = pt.load_plan_json(rbq_path) if os.path.exists(rbq_path) else None
            prior_path = os.path.join(root, prior_dir, f"{rid}.json") if prior_dir else ""
            prior_plan = pt.load_plan_json(prior_path) if prior_dir and os.path.exists(prior_path) else None
            s = build_sample(q_plan, q_sql, v_plan, v_sql, qid, vid, r_plan=r_plan, r_sql=r_sql,
                             rbq_plan=rbq_plan, use_actual=use_actual, prior_plan=prior_plan)
            if s.n < 2:
                skipped.append({"rid": rid, "reason": f"only {s.n} leaf (nothing to order)"}); continue
            samples.append(s)
        except Exception as e:
            logger.exception("skip %s", rid)
            skipped.append({"rid": rid, "reason": repr(e)})
    return samples, skipped
