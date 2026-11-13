

def replace_leaf(tree, leaf_name, subtree):
    if isinstance(tree, str):
        return subtree if tree == leaf_name else tree
    return (replace_leaf(tree[0], leaf_name, subtree),
            replace_leaf(tree[1], leaf_name, subtree))
