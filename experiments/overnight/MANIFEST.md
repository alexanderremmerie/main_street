# Overnight Experiment Grid

Goal: produce a paper-ready sweep over data regime, size generalization to
`N=100`, model capacity, and self-play exploration.

Run each config with:

```sh
uv run python -m main_street.nn --config experiments/overnight/<config>.json
```

Highest priority if time is limited:

1. `07_mixed_24_to_100.json`
2. `17_arch_wide_kernel_100.json`
3. `04_mixed_sampled_small.json`
4. `06_mixed_hard_small.json`
5. `08_selfplay_24_to_100.json`
6. `20_heuristic_forkaware_100.json`

## Blocks

- `01`-`06`: small/solvable data-regime baselines for oracle agreement.
- `07`-`12`: size-generalization runs up to `N=100`.
- `13`-`17`: architecture ablations, all using the `mixed_24_to_100` data regime.
- `18`-`20`: exploration/opponent ablations on the same large mixed regime.
- `21`: seed repeat of the flagship large mixed run.

The flagship large regime is sampled self-play on `N in [12,100]`, mixed with a
small amount of oracle supervision from `train_small`.
