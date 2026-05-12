"""CLI for the trainer.

  uv run python -m main_street.nn --config experiments/smoke.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .train import TrainConfig, Trainer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="main_street.nn")
    p.add_argument("--config", required=True, help="Path to a TrainConfig JSON.")
    args = p.parse_args(argv)
    cfg = TrainConfig.model_validate_json(Path(args.config).read_text())
    final = Trainer(cfg).run()
    print(f"final checkpoint: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
