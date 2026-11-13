import re

def leaf_sql(alias, alias2table):
    t = alias2table[alias]
    return t if t == alias else f"{t} AS {alias}"

def nested_from(tree, alias2table):
    if isinstance(tree, str):
        return leaf_sql(tree, alias2table)
    l, r = tree
    return f"({nested_from(l, alias2table)} JOIN {nested_from(r, alias2table)} ON TRUE)"

def rewrite(q_sql, tree, alias2table):
    m = re.search(r"\bFROM\b(.*?)\bWHERE\b", q_sql, re.S | re.I)
    if not m:
        raise ValueError("no FROM..WHERE")
    return q_sql[:m.start()] + "FROM " + nested_from(tree, alias2table) + " WHERE" + q_sql[m.end():]
