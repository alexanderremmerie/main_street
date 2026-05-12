"""End-to-end training loop.

Single process. One iter:
  1. Each `SampleSource` populates its pool (e.g. self-play generates games).
  2. Run `steps_per_iter` gradient steps; each batch is drawn by weighted mix
     from the sources via `ReplayBuffer.sample`.
  3. Every `eval_every` iters: raw + PUCT oracle agreement on starter +
     holdout + diagnostics; checkpoint; log to wandb.

`TrainConfig` is the full configurable surface. Drop it as JSON and re-run
to reproduce. Sources are a list — that's the whole story for "what data
does this run train on".
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from pydantic import BaseModel, ConfigDict, Field

from ..core import X
from ..eval.positions import PositionSet
from .buffer import ReplayBuffer, Sample
from .checkpoint import CheckpointMeta, save_checkpoint
from .encode import Encoder, EncoderConfig, build_encoder
from .mcts import puct_search, select_action
from .models import build_model
from .sources import SourceConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = "simple_conv"
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    sources: list[SourceConfig] = Field(
        description="Mixed sources. Each contributes by weight to every batch."
    )


class LoopConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    iters: int = 50
    steps_per_iter: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 5
    eval_n_simulations: int = 64


class WandbConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    project: str = "main_street"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    seed: int = 0
    model: ModelConfig = Field(default_factory=ModelConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    data: DataConfig
    loop: LoopConfig = Field(default_factory=LoopConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


# ---------- Loss + step ------------------------------------------------------


def _build_targets(batch: list[Sample], max_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build via numpy then one `from_numpy` per tensor — scalar torch writes
    in a Python loop are orders of magnitude slower on this hot path."""
    B = len(batch)
    pi = np.zeros((B, max_n), dtype=np.float32)
    z = np.empty(B, dtype=np.float32)
    for i, s in enumerate(batch):
        for a, p in s.pi.items():
            pi[i, a] = p
        # Value target is in current player's frame (matches the model's
        # canonicalized tanh output).
        z[i] = s.z if s.state.current_player == X else -s.z
    return torch.from_numpy(pi), torch.from_numpy(z)


def _train_step(
    model,
    optimizer: torch.optim.Optimizer,
    batch: list[Sample],
    encoder: Encoder,
) -> dict[str, float]:
    states = [s.state for s in batch]
    inputs = encoder(states)
    pi_target, z_target = _build_targets(batch, encoder.max_n)

    logits, value = model(inputs)
    log_p = F.log_softmax(logits, dim=-1)
    # 0 * -inf = NaN; mask explicitly.
    log_p = torch.where(inputs["legal_mask"], log_p, torch.zeros_like(log_p))
    policy_loss = -(pi_target * log_p).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, z_target)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }


# ---------- Eval --------------------------------------------------------------


def _in_range_indices(ps: PositionSet, encoder: Encoder) -> list[int]:
    """Indices into `ps` whose spec fits the encoder. Stable across the run."""
    return [
        i
        for i in range(len(ps))
        if (spec := ps.specs[int(ps.spec_idx[i])]).n <= encoder.max_n
        and len(spec.schedule) <= encoder.max_turns
    ]


def _score_raw(
    model, encoder: Encoder, ps: PositionSet, in_range: list[int]
) -> float:
    if not in_range:
        return 0.0
    states = [ps.state(i) for i in in_range]
    with torch.no_grad():
        inputs = encoder(states)
        logits, _ = model(inputs)
    argmax = logits.argmax(dim=-1).cpu().numpy()
    correct = sum(
        bool(ps.optimal_mask[in_range[k], int(a)]) for k, a in enumerate(argmax)
    )
    return correct / len(in_range)


def _score_puct(
    model,
    encoder: Encoder,
    ps: PositionSet,
    in_range: list[int],
    n_simulations: int,
) -> tuple[float, dict[str, bool]]:
    if not in_range:
        return 0.0, {}
    correct = 0
    per_label: dict[str, bool] = {}
    for i in in_range:
        root = puct_search(
            ps.state(i), model, encoder, n_simulations=n_simulations, c_puct=1.5
        )
        a = select_action(root, temperature=0.0)
        ok = bool(ps.optimal_mask[i, a])
        if ok:
            correct += 1
        if ps.labels[i]:
            per_label[ps.labels[i]] = ok
    return correct / len(in_range), per_label


def _eval_metrics(
    model,
    encoder: Encoder,
    sets: dict[str, PositionSet],
    in_range_by_set: dict[str, list[int]],
    puct_sims: int,
    puct_sets: tuple[str, ...] = ("diagnostics",),
) -> dict[str, Any]:
    """Raw-policy agreement on every set (cheap, batched). PUCT agreement only
    on `puct_sets` — search is expensive per position and meaningfully diverges
    from raw only on hard positions, which is what diagnostics holds."""
    out: dict[str, Any] = {}
    for name, ps in sets.items():
        in_range = in_range_by_set[name]
        out[f"raw/{name}/agreement"] = _score_raw(model, encoder, ps, in_range)
        out[f"raw/{name}/skipped"] = len(ps) - len(in_range)

        if name in puct_sets:
            puct_agr, per_label = _score_puct(
                model, encoder, ps, in_range, puct_sims
            )
            out[f"puct/{name}/agreement"] = puct_agr
            for lbl, ok in per_label.items():
                out[f"diag/{lbl}"] = ok
    return out


# ---------- Trainer ----------------------------------------------------------


@dataclass
class Trainer:
    cfg: TrainConfig
    run_dir: Path = field(init=False)
    run_id: str = field(init=False)

    def __post_init__(self) -> None:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        self.run_id = f"{ts}_{self.cfg.name}_{uuid.uuid4().hex[:6]}"
        self.run_dir = Path("data") / "runs" / self.run_id
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(self.cfg.model_dump(), indent=2)
        )

    def run(self) -> Path:
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)

        encoder = build_encoder(self.cfg.encoder)

        eval_sets = {
            "starter": PositionSet.load("starter"),
            "holdout": PositionSet.load("holdout"),
            "diagnostics": PositionSet.load("diagnostics"),
        }
        in_range_by_set = {
            name: _in_range_indices(ps, encoder) for name, ps in eval_sets.items()
        }

        sources = [cfg.build() for cfg in self.cfg.data.sources]
        buffer = ReplayBuffer(sources, rng=rng)
        print(
            f"[{self.run_id}] sources: "
            + ", ".join(f"#{i}={s.size}@w={s.weight}" for i, s in enumerate(sources))
        )

        model = build_model(self.cfg.model.name, encoder, self.cfg.model.params)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.loop.lr,
            weight_decay=self.cfg.loop.weight_decay,
        )

        wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            mode=self.cfg.wandb.mode,
            name=self.run_id,
            config=self.cfg.model_dump(),
            dir=str(self.run_dir),
        )

        try:
            final_iter = self.cfg.loop.iters
            for it in range(1, final_iter + 1):
                model.eval()
                t0 = time.perf_counter()
                added = buffer.populate(model, encoder)
                t_pop = time.perf_counter() - t0

                model.train()
                t1 = time.perf_counter()
                losses: list[dict[str, float]] = []
                for _ in range(self.cfg.loop.steps_per_iter):
                    batch = buffer.sample(self.cfg.loop.batch_size)
                    losses.append(_train_step(model, optimizer, batch, encoder))
                t_train = time.perf_counter() - t1

                avg = {
                    k: float(np.mean([li[k] for li in losses])) for k in losses[0]
                }
                log_d: dict[str, Any] = {
                    "iter": it,
                    "time/populate_s": t_pop,
                    "time/train_s": t_train,
                }
                log_d.update({f"buffer/{k}": v for k, v in buffer.sizes.items()})
                log_d.update({f"buffer/{k}": v for k, v in added.items()})
                log_d.update({f"train/{k}": v for k, v in avg.items()})

                do_eval = it % self.cfg.loop.eval_every == 0 or it == final_iter
                if do_eval:
                    model.eval()
                    log_d.update(
                        _eval_metrics(
                            model=model,
                            encoder=encoder,
                            sets=eval_sets,
                            in_range_by_set=in_range_by_set,
                            puct_sims=self.cfg.loop.eval_n_simulations,
                        )
                    )
                    ckpt = self.run_dir / "checkpoints" / f"iter_{it:04d}.pt"
                    save_checkpoint(
                        ckpt,
                        model=model,
                        model_name=self.cfg.model.name,
                        model_params=self.cfg.model.params,
                        encoder_config=self.cfg.encoder,
                        meta=CheckpointMeta(run_id=self.run_id, iter=it),
                    )

                wandb.log(log_d)
                msg = (
                    f"iter {it:>3}  loss={avg['loss']:.3f}  "
                    f"pi={avg['policy_loss']:.3f}  v={avg['value_loss']:.3f}  "
                    f"pop_t={t_pop:.1f}s  tr_t={t_train:.1f}s"
                )
                if do_eval:
                    msg += (
                        f"  raw/starter={log_d['raw/starter/agreement']:.3f}"
                        f"  puct/diag={log_d['puct/diagnostics/agreement']:.3f}"
                    )
                print(msg)

            final = self.run_dir / "checkpoints" / "final.pt"
            save_checkpoint(
                final,
                model=model,
                model_name=self.cfg.model.name,
                model_params=self.cfg.model.params,
                encoder_config=self.cfg.encoder,
                meta=CheckpointMeta(run_id=self.run_id, iter=final_iter),
            )

            # Append a one-line summary to the runs index so we can grep
            # "what have I trained" without spelunking through run_dir/*.
            _append_runs_index(
                self.run_dir.parent / "_index.jsonl",
                run_id=self.run_id,
                name=self.cfg.name,
                final_iter=final_iter,
                final_checkpoint=str(final),
            )
            return final
        finally:
            wandb.finish()


def _append_runs_index(
    path: Path,
    run_id: str,
    name: str,
    final_iter: int,
    final_checkpoint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "name": name,
        "final_iter": final_iter,
        "final_checkpoint": final_checkpoint,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
