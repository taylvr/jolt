

def vsub_rows(plan, v_aliases):
    def aliases_of(n):
        if "Alias" in n:
            return {n["Alias"]}
        s = set()
        for c in n.get("Plans", []):
            s |= aliases_of(c)
        return s
    best = None

    def walk(n):
        nonlocal best
        if aliases_of(n) == set(v_aliases):
            if best is None or n.get("Plan Rows", 0) > best.get("Plan Rows", 0):
                best = n
        for c in n.get("Plans", []):
            walk(c)
    walk(plan)
    return best.get("Plan Rows") if best else None
