"""`python -m main_street ...` entry points.

Currently dispatches to:
- `solve` — compute and persist the solved policy table for a (N, schedule).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .core import GameSpec
from .solve import DEFAULT_TABLE_DIR, build_table, save_table


def _build_spec(n: int, schedule_text: str) -> GameSpec:
    try:
        parts = tuple(int(p) for p in schedule_text.split(",") if p.strip())
        return GameSpec(n=n, schedule=parts)
    except (ValueError, ValidationError) as e:
        raise argparse.ArgumentTypeError(f"invalid spec: {e}") from e


def _cmd_solve(args: argparse.Namespace) -> int:
    spec = _build_spec(args.n, args.schedule)
    out_dir = Path(args.out)
    table = build_table(spec)
    path = save_table(table, root=out_dir)
    print(
        f"solved {spec.n=} schedule={list(spec.schedule)} "
        f"value={table.value:+d} states={len(table.entries)} -> {path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m main_street")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve", help="solve a (N, schedule) and persist the table")
    p_solve.add_argument("--n", type=int, required=True, help="board length")
    p_solve.add_argument(
        "--schedule", type=str, required=True, help="comma-separated positive ints, e.g. 2,2,1"
    )
    p_solve.add_argument(
        "--out", type=str, default=str(DEFAULT_TABLE_DIR), help="output directory"
    )
    p_solve.set_defaults(func=_cmd_solve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
