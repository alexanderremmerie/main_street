from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_valid_reward: float = 0.1
    format_invalid_reward: float = 0.0
    legal_reward: float = 0.5
    illegal_reward: float = -1.0
    optimal_reward: float = 1.0
    suboptimal_reward: float = 0.0
    invalid_move_reward: float = -1.0
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: tuple[str, ...] | None = None
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"


class GRPOTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name_or_path: str = Field(min_length=1)
    train_dataset_path: str = Field(min_length=1)
    eval_dataset_path: str | None = None
    output_dir: str = Field(min_length=1)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    trainer_args: dict[str, Any] = Field(default_factory=dict)
    resume_from_checkpoint: str | None = None

