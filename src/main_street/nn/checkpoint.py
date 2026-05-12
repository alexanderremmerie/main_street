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
