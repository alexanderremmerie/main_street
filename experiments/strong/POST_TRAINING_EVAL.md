# Post-Training Evaluation Plan

Use this after the `experiments/strong/*.json` checkpoints finish training.
Training is frozen. The goal is twofold: (1) measure size generalization and
(2) test whether trained models learned the specific structural strategies the
paper proves are optimal.

---

## Why this game lets us measure generalization

Main Street has **closed-form solutions for two-turn schedules**, so we know
exactly which moves are optimal and why — no approximation needed. This lets
the eval go beyond "did the agent win?" to "did the agent apply the correct
structural reasoning?"

Three strategy regimes map directly to paper §3.3:

| Strategy | Paper section | Probe set |
|---|---|---|
| **Rightmost dominance** (γ₁ ≥ γ₂ → X always wins by playing rightmost cells) | §3.3.1 Case 1 | `probe_rightmost` |
| **Fragmentation / partitioning** (γ₁ < γ₂, N ≤ T → sparse winning pattern) | §3.3.1 Case 2 | `probe_partition` |
| **Threshold awareness** (same schedule, N sweeps across T(γ₁)) | §3.3.1 Case 2 | `probe_threshold` |
| **Schedule-order sensitivity** (permuted schedule → different winner) | §3.3.2 | `probe_order` |
| **Perturbation sensitivity** (+1 O mark flips outcome) | §3.3.3 | `probe_perturbation` |
| **Tie-break awareness** | game rules | `probe_tiebreak` |
| **Fork / open-end awareness** | mid-game | `probe_fork` |
| **Potential denial** | mid-game | `probe_potential` |

The **minimal story for the paper** is the first four rows. The rest are
supporting evidence.

---

## 1. Build Eval Sets Once

```sh
uv run python -m main_street.eval build starter
uv run python -m main_street.eval build holdout
uv run python -m main_street.eval build diagnostics
uv run python -m main_street.eval build probe_partition
uv run python -m main_street.eval build probe_threshold
uv run python -m main_street.eval build probe_rightmost
uv run python -m main_street.eval build probe_perturbation
uv run python -m main_street.eval build probe_order
uv run python -m main_street.eval build probe_tiebreak
uv run python -m main_street.eval build probe_fork
uv run python -m main_street.eval build probe_potential
```

### Strategy probe descriptions

- **`probe_rightmost`** — 5 initial positions where γ₁ ≥ γ₂ (3 equal-mark,
  2 advantage). The dominant strategy is to place the first mark in the rightmost
  γ₁ cells. Oracle agreement tests whether the model prefers right-side placements.
  "equal" cases (γ₁=γ₂) are the clearest: the only way to win is by tie-break,
  which requires playing rightmost.

- **`probe_partition`** — 5 initial positions at exactly N=T(γ₁) for γ₁ ∈ {2,3,4,5,6}.
  The winning strategy is the k-singleton + rightmost-block fragmentation pattern.
  Winning placements are sparse: 1 of C(N, γ₁) for the first move.

- **`probe_threshold`** — 9 initial positions: (5,6) at N∈{12,14,16,18,20} and
  (4,5) at N∈{10,12,14,16}. Labels indicate "xwins" (N ≤ T) or "owins" (N > T).
  Tests whether oracle agreement flips correctly when N crosses T(γ₁). A model
  that learned the threshold rule should score near 1.0 on both sides; a model
  that only pattern-matches on size will show a drop at the unfamiliar side of T.

- **`probe_order`** — 3 initial positions with the same schedule multiset
  but different orderings on N=10. Different permutations yield different winners.
  Tests schedule-order sensitivity.

- **`probe_perturbation`** — 2 initial positions: (2,3,2) and (2,4,2) on N=8.
  Adding one mark to O's turn flips the outcome. Note: (8,(2,3,2)) is in
  TRAIN_SMALL — it tests in-distribution correctness; (8,(2,4,2)) is OOD.

---

## 2. Pick Checkpoints

At minimum, evaluate these final checkpoints:

- `04_large_mixed_12_to_100` — main mixed-data model, broad training
- `06_train_12_to_48_arch100` — restricted lower N range
- `07_train_24_to_72_arch100` — mid range
- `08_train_48_to_100_arch100` — restricted upper N range
- `09_full_range_8_to_100_arch100` — full range including smallest boards
- `10_train_64_to_100_arch100` — large-board specialisation
- `21_deep_arch` — depth ablation
- `22_wide_kernel_arch` — capacity ablation

Also collect 3 intermediate checkpoints per flagship run (04, 09):
- early: first checkpoint with stable non-trivial holdout score
- middle: ~50% of training
- final: last checkpoint

---

## 3. Oracle-Agreement Scores

Score all four strategy probes for every checkpoint:

```sh
CKPT=data/runs/<run_id>/checkpoints/final.pt
NAME=<run_shortname>

uv run python -m main_street.eval score starter      --ckpt $CKPT --label ${NAME}_starter
uv run python -m main_street.eval score holdout      --ckpt $CKPT --label ${NAME}_holdout
uv run python -m main_street.eval score diagnostics  --ckpt $CKPT --label ${NAME}_diag
uv run python -m main_street.eval score probe_rightmost    --ckpt $CKPT --label ${NAME}_rightmost
uv run python -m main_street.eval score probe_partition    --ckpt $CKPT --label ${NAME}_partition
uv run python -m main_street.eval score probe_threshold    --ckpt $CKPT --label ${NAME}_threshold
uv run python -m main_street.eval score probe_order        --ckpt $CKPT --label ${NAME}_order
uv run python -m main_street.eval score probe_perturbation --ckpt $CKPT --label ${NAME}_perturbation
uv run python -m main_street.eval score probe_tiebreak     --ckpt $CKPT --label ${NAME}_tiebreak
uv run python -m main_street.eval score probe_fork         --ckpt $CKPT --label ${NAME}_fork
uv run python -m main_street.eval score probe_potential    --ckpt $CKPT --label ${NAME}_potential
```

Score the same probes on baselines for the comparison table:

```sh
for PROBE in probe_rightmost probe_partition probe_threshold probe_order probe_perturbation; do
  for AGENT in random greedy rightmost forkaware potentialaware alphabeta; do
    uv run python -m main_street.eval score $PROBE $AGENT
  done
done
```

This establishes the baseline ceiling (alphabeta = 1.0 on all probes) and floor
(random) so model scores can be placed in context.

---

## 4. Threshold Sweep Analysis

`probe_threshold` produces per-label scores. The key comparison is:

- **xwins labels** (N ≤ T): does the model find the fragmentation move?
  These are the hard cases — winning placements are sparse.
- **owins labels** (N > T): does the model find X's best defense?
  All legal moves have value -1 for X, so this tests the quality of X's
  best-losing play.

Expected pattern for a model that learned the threshold rule:
- High agreement on both sides of T (correct play in each regime).

Expected pattern for a model that only memorised training positions:
- High agreement for N in the training range, drop outside it.

Plot a bar chart with x-axis = N, y-axis = oracle agreement, grouped by model.
The threshold T should be marked as a vertical line.

---

## 5. Match Play vs Baselines

```python
from main_street.agents import AlphaZeroAgentSpec, GreedyAgentSpec, ForkAwareAgentSpec
from main_street.core import GameSpec
from main_street.records import EvalConfig
from main_street.runner import run_tournament
from main_street.store import bootstrap, connect

bootstrap()
conn = connect()
cfg = EvalConfig(
    agent_a=AlphaZeroAgentSpec(checkpoint_path="<CKPT>", n_simulations=64),
    agent_b=ForkAwareAgentSpec(seed=0),
    specs=(
        GameSpec(n=10, schedule=(2, 3, 3)),
        GameSpec(n=10, schedule=(3, 4, 2, 1)),
        GameSpec(n=10, schedule=(2, 4, 3, 1)),
    ),
    n_games_per_spec=20,
    swap_sides=True,
    seed=0,
)
rec = run_tournament(conn, cfg)
print(rec.summary)
```

Recommended baseline opponents: `greedy`, `forkaware`, `potentialaware`,
`alphabeta` (only on small exact arenas, N ≤ 12).

Recommended model-vs-model:
- `04` vs `06` — does broad training hurt small-board strategy?
- `04` vs `08` — does large-board training hurt small-board strategy?
- `06` vs `10` — do training ranges that don't overlap transfer to each other?
- `09` vs `04` — does including N=8-11 improve small-board probes?

---

## 6. Main Questions Per Experiment

### A. Board-size generalization (`06`, `07`, `08`, `09`, `10`)

- Does training on a restricted N range transfer upward, downward, or both?
- Which probes fail first as N moves OOD?

**Outputs**: heatmap — rows = training N range, cols = {starter, holdout,
probe_partition, probe_threshold}. Color = oracle agreement.

### B. Strategy acquisition (`04` + train-range models, with checkpoints)

- Which of the four structural skills appear, and in what order?
- Does fragmentation (probe_partition) appear before or after tie-break
  awareness (probe_tiebreak)?

**Outputs**: line plot — x = training iteration, y = probe score. One curve
per probe. Use intermediate checkpoints from `04` or `09`.

### C. Schedule-distribution sensitivity (`11`–`16`)

- Do many-turn vs few-big-turn training regimes teach different skills?
- Does training on random schedules hurt performance on structured probes?

**Outputs**: grouped bar chart — model vs probe family.

### D. Search / exploration / architecture (`17`–`22` vs `04`)

- Are strategy gains from better targets (more sims), better exploration
  (Dirichlet), or better representation (deeper / wider)?

**Outputs**: ablation table — each run vs `04` on the four core probes.

---

## 7. Skill Claims to Keep Narrow

Make a claim only when a specific probe supports it.

| Claim | Required evidence |
|---|---|
| Agent learned rightmost dominance | probe_rightmost ≥ 0.85 |
| Agent learned fragmentation | probe_partition ≥ 0.60 (random << 0.05 due to sparsity) |
| Agent is threshold-aware | probe_threshold: xwins ≥ 0.70 AND owins ≥ 0.70 |
| Agent is schedule-order sensitive | probe_order ≥ 0.70 (three distinct schedules) |
| Agent generalises OOD | holdout ≥ starter − 0.10 |

Do not claim a skill unless the corresponding probe clears its bar.

---

## 8. Suggested Paper Figures

- **Figure 1**: Train-range generalization heatmap (rows=model, cols=eval family).
- **Figure 2**: Threshold sweep bar chart (x=N, y=oracle agreement, line at T).
- **Figure 3**: Skill acquisition curves over checkpoints (one curve per probe).
- **Figure 4**: Schedule-regime ablation grouped bar chart.
- **Table 1**: Exact small-board oracle agreement (all probes × key models).
- **Table 2**: Win rate vs baselines on matched arenas.

---

## 9. Minimal Deliverable If Time Is Tight

1. Evaluate `04`, `06`, `08`, `09`, `10` (final checkpoints only).
2. Score on `holdout`, `probe_partition`, `probe_threshold`, `probe_rightmost`, `probe_order`.
3. Run match play vs `greedy`, `forkaware`, `potentialaware`.
4. Report: Table 1 (oracle agreement) + Figure 1 (generalization heatmap) + Figure 2 (threshold sweep).

That is enough for a defensible paper section on structural generalization.
