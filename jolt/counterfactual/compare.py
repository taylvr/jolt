
from ..core import plan_tree as pt

def cmp_trees(got, tgt):
    return dict(
        exact=int(pt.tree_to_str(got) == pt.tree_to_str(tgt)),
        canon=int(pt.tree_to_str(pt.canon(got)) == pt.tree_to_str(pt.canon(tgt))),
        seq=int(pt.tree_to_seq(got)[0] == pt.tree_to_seq(tgt)[0]),
    )
