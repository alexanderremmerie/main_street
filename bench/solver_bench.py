"""Bench the exact solver on a fixed grid of specs.

Usage: `uv run python -m bench.solver_bench` (or `python bench/solver_bench.py`).

Each spec is solved with a fresh `Solver` (no warm cache across specs). The grid
is fixed so before/after numbers are directly comparable. CSV output lands at
`bench/solver_bench.csv` next to this script.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Make `src/` importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from main_street.core import GameSpec, GameState  # noqa: E402
from main_street.solve import Solver  # noqa: E402

SPECS: list[tuple[int, tuple[int, ...]]] = [
    (8, (3, 4)),
    (10, (3, 3, 3)),
    (12, (4, 4)),
    (12, (1, 4, 4)),
    (14, (2, 4, 4)),
    (16, (4, 5)),
    (16, (1, 1, 1, 1, 1, 1, 1, 1)),
    (18, (3, 4, 3)),
    (20, (5, 6)),
]


def main() -> None:
    rows: list[dict[str, object]] = []
    for n, schedule in SPECS:
        spec = GameSpec(n=n, schedule=schedule)
        state = GameState.initial(spec)
        solver = Solver()
        t0 = time.perf_counter()
        result = solver.solve(state)
        dt = time.perf_counter() - t0
        tt_size = solver.tt_size
        rows.append(
            {
                "n": n,
                "schedule": "-".join(map(str, schedule)),
                "value": result.value,
                "best_cell": result.best_cell,
                "tt_size": tt_size,
                "seconds": f"{dt:.4f}",
            }
        )
        print(
            f"n={n:>3} sched={'-'.join(map(str, schedule)):<18} "
            f"value={result.value:>+d} best={result.best_cell:>2} "
            f"tt={tt_size:>9} t={dt:.3f}s"
        )

    out = Path(__file__).resolve().parent / "solver_bench.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
