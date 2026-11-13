# JOLT

Join-order prediction and cost estimation for materialized-view query rewriting.

## Requirements

- Python 3.8+
- numpy, pandas
- `psql` (optional)

## Installation

No build step. Clone the repository and make sure the package root is on
`sys.path` (or run from the repository root):

```bash
pip install numpy pandas
python -c "import jolt; print('ok')"
```

## Package Layout

```
jolt/
├── core/            plan-tree and SQL parsing, sample construction
├── ordering/        join-order model: candidate features, GBDT scorer, beam search
├── counterfactual/  order-pinned EXPLAIN: SQL rewrite, plan substitution, backend
└── estimation/      cost features: plan stats, hint features, subtraction, prior
```

## Quick Start

```python
from jolt.core.sample import build_sample
from jolt.ordering.scorer import GBDTScorer
from jolt.ordering.beam_search import beam_search
from jolt.estimation.subtraction import sub_row
```

## Configuration

| Variable    | Default     | Meaning                        |
|------------|-------------|--------------------------------|
| `PGHOST`   | `localhost` | PostgreSQL host                |
| `PGPORT`   | `5432`      | PostgreSQL port                |
| `PGUSER`   | `postgres`  | PostgreSQL user                |
| `MVDB`     | `imdb`      | Target database name           |
| `MVJO_ROOT`| package dir | Data root for sample artifacts |

## Data

The dataset is hosted on [Google Drive](https://drive.google.com/file/d/1gEP1epha46ohTWF1w5WDaQFe7YM9QjoF/view?usp=drive_link). Download and extract it next
to the package. The extracted `qvr_sql/` directory holds, for each of the three
benchmark workloads, three subfolders of plain `.sql` files:

| Folder    | Content                          | Naming                  |
|-----------|----------------------------------|-------------------------|
| `sql/q/`  | original queries (Q)             | `<q_id>.sql`            |
| `sql/v/`  | view definition queries (V)      | `<v_id>.sql`            |
| `sql/r/`  | rewritten queries (R)            | `<rid>.sql`             |

An `rid` follows the pattern `q_id__v_id`, so each rewritten query can be traced
back to its original query and view.
