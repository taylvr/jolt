import re

import numpy as np

from ..core import plan_tree as pt

def seq_of(tree_str):
    t = pt.str_to_tree(tree_str)
    seq, sides, _ = pt.tree_to_seq(t)
    return seq, sides

def kendall_seq(a, b):
    if sorted(a) != sorted(b) or len(a) < 2:
        return 0.0
    pos = {x: i for i, x in enumerate(b)}
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos[a[i]] < pos[a[j]]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc)

def prefix_ratio(a, b):
    if not a or len(a) != len(b):
        return 0.0
    k = 0
    if set(a[:2]) == set(b[:2]):
        k = 2
        while k < len(a) and a[k] == b[k]:
            k += 1
    return k / len(a)

def mv_pos(seq, mv):
    for i, x in enumerate(seq):
        if x == mv:
            return i
    return -1

def first_join_set(tree_strs):
    outs = set()
    for s in tree_strs:
        pairs = re.findall(r"\(([^()]+)\)", s)
        if pairs:
            outs.add(pairs[0])
    return outs

def canon_str(t):
    return pt.tree_to_str(pt.canon(t)) if t is not None else None

def ord_feats(b, s, prefix):
    seq_p, _ = seq_of(b["pred"])
    seq_b, _ = seq_of(b["qbr"])
    beam_strs = [x[0] for x in b["beams"]]
    logps = np.array([x[1] for x in b["beams"]])
    mv = s.mv
    return {
        f"{prefix}top1_eq_qbr": 1.0 if canon_str(pt.str_to_tree(b["pred"])) ==
        canon_str(pt.str_to_tree(b["qbr"])) else 0.0,
        f"{prefix}kendall_vs_qbr": kendall_seq(seq_p, seq_b),
        f"{prefix}prefix_vs_qbr": prefix_ratio(seq_p, seq_b),
        f"{prefix}mv_step_pred": mv_pos(seq_p, mv) / max(len(seq_p) - 1, 1),
        f"{prefix}mv_step_qbr": mv_pos(seq_b, mv) / max(len(seq_b) - 1, 1),
        f"{prefix}mv_step_diff": (mv_pos(seq_p, mv) - mv_pos(seq_b, mv)) / max(len(seq_p), 1),
        f"{prefix}logp_top1": logps[0] if len(logps) else 0.0,
        f"{prefix}logp_gap12": float(logps[0] - logps[1]) if len(logps) > 1 else 0.0,
        f"{prefix}logp_std": float(np.std(logps)) if len(logps) > 1 else 0.0,
        f"{prefix}n_beams_ne_qbr": sum(1.0 for x in beam_strs
                                       if canon_str(pt.str_to_tree(x)) !=
                                       canon_str(pt.str_to_tree(b["qbr"]))),
        f"{prefix}n_firstjoins": float(len(first_join_set(beam_strs))),
        f"{prefix}beam_has_qbr": 1.0 if any(canon_str(pt.str_to_tree(x)) ==
                                            canon_str(pt.str_to_tree(b["qbr"]))
                                            for x in beam_strs) else 0.0,
        f"{prefix}n": float(b["n"]),
        f"{prefix}geqo": 1.0 if b["n"] >= 12 else 0.0,
    }
