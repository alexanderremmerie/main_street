# Strong Experiment Suite

These configs are the paper-grade replacement for the quick `experiments/overnight`
sweep. All runs train from scratch and use online W&B.

Main design rule: every cross-distribution comparison model uses
`encoder.max_n=100` and `encoder.max_turns=16`, even when its self-play data is
restricted to small boards. This lets us evaluate the same checkpoint on held-out
larger boards. `05_small_arch32_control` is the deliberate exception: it controls
for the old small-encoder setup and cannot play boards with `N > 32`.

Common budget unless a config explicitly varies compute:

- `iters=600`
- `steps_per_iter=250`
- `batch_size=512`
- self-play `n_simulations=64`
- self-play `games_per_iter=40`
- `eval_every=25`
- replay `capacity=500000`

## Runs

1. `01_small_selfplay_arch100.json` — small-board self-play only, but with
   `max_n=100`. Tests whether small-board strategies extrapolate when the
   architecture can encode larger boards.
2. `02_small_mixed_arch100.json` — small-board supervised + self-play. Tests
   whether exact small-board oracle data improves small-to-large transfer.
3. `03_large_selfplay_12_to_100.json` — broad N=12-100 self-play baseline.
4. `04_large_mixed_12_to_100.json` — broad N=12-100 supervised + self-play;
   the main mixed-data model.
5. `05_small_arch32_control.json` — small-board self-play with `max_n=32`.
   Controls for architectural support and documents why old strong small models
   are not comparable on N>32.
6. `06_train_12_to_48_arch100.json` — trains on small/mid boards only. Measures
   extrapolation to much larger boards.
7. `07_train_24_to_72_arch100.json` — trains on mid boards. Tests transfer both
   down to small boards and up to large boards.
8. `08_train_48_to_100_arch100.json` — trains only on large boards. Tests whether
   large specialization sacrifices small-board play.
9. `09_full_range_8_to_100_arch100.json` — trains on the full 8-100 range. Tests
   whether including the smallest boards helps broad generalization.
10. `10_train_64_to_100_arch100.json` — trains only on very large boards. Tests
    large-board specialization.
11. `11_many_turns.json` — many small turns. Tests long-horizon tactical play.
12. `12_few_big_turns.json` — few large turns. Tests large placement-turn
    combinatorics.
13. `13_high_fill.json` — high board occupancy. Tests crowded, blocking-heavy
    regimes.
14. `14_low_fill.json` — low board occupancy. Tests sparse/open-board regimes.
15. `15_arc_heavy.json` — mostly arc-structured schedules. Tests whether
    structured schedule families teach transferable strategy.
16. `16_random_schedule_heavy.json` — mostly random schedules. Tests broad
    robustness against unstructured schedules.
17. `17_low_sims_32.json` — broad mixed model with 32 self-play sims. Search
    compute ablation.
18. `18_high_sims_128.json` — broad mixed model with 128 self-play sims and 24
    games/iter. Search-target quality ablation.
19. `19_no_dirichlet.json` — broad mixed model with no root noise. Exploration
    ablation.
20. `20_high_exploration.json` — broad mixed model with higher c_puct, Dirichlet
    noise, and temperature. Exploration ablation.
21. `21_deep_arch.json` — 64 channels, 8 residual blocks. Depth ablation.
22. `22_wide_kernel_arch.json` — 128 channels, 4 blocks, kernel size 7. Capacity
    and local receptive-field ablation.
