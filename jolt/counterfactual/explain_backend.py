import json
import os
import subprocess
import time

PSQL = ["psql", "-h", os.environ.get("PGHOST", "localhost"),
        "-p", os.environ.get("PGPORT", "5432"),
        "-U", os.environ.get("PGUSER", "postgres"),
        "-d", os.environ.get("MVDB", "imdb"), "-X", "-q", "-A", "-t"]

def explain_json(sql):
    full = ("SET join_collapse_limit = 1; SET from_collapse_limit = 1; "
            "SET max_parallel_workers_per_gather = 0; SET max_parallel_workers = 0;\n"
            "EXPLAIN (FORMAT JSON, COSTS) " + sql.rstrip().rstrip(";") + ";")
    t0 = time.time()
    out = subprocess.run(PSQL + ["-c", full], capture_output=True, text=True, timeout=120)
    dt = time.time() - t0
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:200])
    txt = out.stdout.strip()
    j = json.loads(txt[txt.index("["):])
    return j[0]["Plan"], dt
