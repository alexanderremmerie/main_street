from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import build_preset_dataset, export_position_set
from .train import train_from_config_path


def cmd_export(args: argparse.Namespace) -> int:
    return_code = 0
    if args.preset:
        out = build_preset_dataset(
            args.preset,
            out_path=args.out,
            prompt_style=args.prompt_style,
            limit=args.limit,
            seed=args.seed,
        )
    else:
        from ..eval.positions import PositionSet

        ps = PositionSet.load(args.position_set, root=Path(args.eval_root))
        out = export_position_set(
            ps,
            out_path=args.out,
            prompt_style=args.prompt_style,
            limit=args.limit,
            seed=args.seed,
        )
    print(f"wrote {out}")
    return return_code


def cmd_train(args: argparse.Namespace) -> int:
    return train_from_config_path(args.config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m main_street.grpo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser(
        "export",
        help="export GRPO prompt dataset from solver-labeled states",
    )
    p_export.add_argument("--preset", help="built-in preset name such as train_small")
    p_export.add_argument("--position-set", help="prebuilt PositionSet name under data/eval")
    p_export.add_argument("--eval-root", default="data/eval", help="root for --position-set loads")
    p_export.add_argument("--out", required=True, help="output JSONL path")
    p_export.add_argument("--prompt-style", default="json_v1")
    p_export.add_argument("--limit", type=int)
    p_export.add_argument("--seed", type=int, default=0)
    p_export.set_defaults(func=cmd_export)

    p_train = sub.add_parser("train", help="run GRPO training from a JSON config")
    p_train.add_argument("--config", required=True, help="path to a GRPO training config JSON")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    if (
        args.cmd == "export"
        and not getattr(args, "preset", None)
        and not getattr(args, "position_set", None)
    ):
        parser.error("export requires either --preset or --position-set")
    if getattr(args, "preset", None) and getattr(args, "position_set", None):
        parser.error("export accepts only one of --preset or --position-set")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
