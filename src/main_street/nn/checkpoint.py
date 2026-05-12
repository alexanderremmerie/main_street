"""Checkpoint format.

A `.pt` file that fully reconstructs the `(encoder, model)` pair so a trained
agent can be loaded anywhere with just the checkpoint path.

Saved fields:
  encoder_config (dict)  — EncoderConfig.model_dump()
  model_name (str)
  model_params (dict)
  state_dict
  run_id (str), iter (int)

`load_checkpoint(path) -> (model, encoder, meta)` instantiates everything from
the recorded names + params. The model is in eval mode and on CPU.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .encode import Encoder, EncoderConfig, build_encoder
from .models import Model, build_model


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    run_id: str
    iter: int


@dataclass(frozen=True, slots=True)
class DiscoveredCheckpoint:
    path: Path
    run_id: str
    run_name: str
    iter: int | None
    is_final: bool


def _parse_iter(stem: str) -> int | None:
    if not stem.startswith("iter_"):
        return None
    try:
        return int(stem.removeprefix("iter_"))
    except ValueError:
        return None


def discover_checkpoints(runs_dir: Path) -> list[DiscoveredCheckpoint]:
    """Walk `runs_dir` and list every `.pt` under each run's `checkpoints/`.
    `final.pt` sorts first, then `iter_<n>.pt` by descending n; runs sort
    newest-first by directory name."""
    if not runs_dir.exists():
        return []
    out: list[DiscoveredCheckpoint] = []
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
        run_name = run_dir.name
        config_path = run_dir / "config.json"
        if config_path.exists():
            with suppress(OSError, ValueError):
                run_name = str(json.loads(config_path.read_text()).get("name") or run_name)

        def _sort_key(p: Path) -> tuple[int, int]:
            if p.name == "final.pt":
                return (1, 10**9)
            return (0, _parse_iter(p.stem) or -1)

        for ckpt in sorted(ckpt_dir.glob("*.pt"), key=_sort_key, reverse=True):
            out.append(
                DiscoveredCheckpoint(
                    path=ckpt,
                    run_id=run_dir.name,
                    run_name=run_name,
                    iter=_parse_iter(ckpt.stem),
                    is_final=ckpt.name == "final.pt",
                )
            )
    return out


def save_checkpoint(
    path: Path,
    model: Model,
    model_name: str,
    model_params: dict[str, Any],
    encoder_config: EncoderConfig,
    meta: CheckpointMeta,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_config": encoder_config.model_dump(),
        "model_name": model_name,
        "model_params": model_params,
        "state_dict": model.state_dict(),
        "run_id": meta.run_id,
        "iter": meta.iter,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> tuple[Model, Encoder, CheckpointMeta]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    encoder_config = EncoderConfig(**payload["encoder_config"])
    encoder = build_encoder(encoder_config)
    model = build_model(payload["model_name"], encoder, payload["model_params"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = CheckpointMeta(run_id=payload["run_id"], iter=int(payload["iter"]))
    return model, encoder, meta
