from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import GRPOTrainConfig
from .rewards import build_reward_funcs


def _lazy_import_training_stack() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise RuntimeError(
            "GRPO training dependencies are not installed. "
            "Use `uv run --group rl ...` or install the `rl` dependency group."
        ) from e
    return load_dataset, GRPOConfig, GRPOTrainer, None


def load_train_config(path: str | Path) -> GRPOTrainConfig:
    return GRPOTrainConfig.model_validate(json.loads(Path(path).read_text()))


def _build_peft_config(cfg: GRPOTrainConfig) -> Any | None:
    if not cfg.lora.enabled:
        return None
    try:
        from peft import LoraConfig
    except ImportError as e:
        raise RuntimeError(
            "LoRA was requested but `peft` is not installed. "
            "Use `uv run --group rl ...` or install the `rl` dependency group."
        ) from e
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=list(cfg.lora.target_modules) if cfg.lora.target_modules else None,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )


def train_from_config_path(path: str | Path) -> int:
    cfg = load_train_config(path)
    load_dataset, GRPOConfig, GRPOTrainer, _ = _lazy_import_training_stack()

    reward_funcs, reward_weights = build_reward_funcs(cfg.reward)
    trainer_kwargs = dict(cfg.trainer_args)
    trainer_kwargs.setdefault("output_dir", cfg.output_dir)
    trainer_kwargs.setdefault("remove_unused_columns", False)
    trainer_kwargs.setdefault("report_to", "none")
    trainer_kwargs.setdefault("save_only_model", True)
    trainer_kwargs["reward_weights"] = reward_weights

    training_args = GRPOConfig(**trainer_kwargs)
    train_dataset = load_dataset("json", data_files=str(cfg.train_dataset_path), split="train")
    eval_dataset = None
    if cfg.eval_dataset_path:
        eval_dataset = load_dataset("json", data_files=str(cfg.eval_dataset_path), split="train")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_train_config.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2)
    )

    trainer = GRPOTrainer(
        model=cfg.model_name_or_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=_build_peft_config(cfg),
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    return 0

