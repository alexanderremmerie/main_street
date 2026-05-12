import numpy as np
import pytest

from main_street.core import GameSpec
from main_street.spec_sampling import (
    SpecSamplerConfig,
    _bounded_composition,
    _weighted_bounded_composition,
    sample_unique_specs,
)


def assert_valid_sample(spec: GameSpec, cfg: SpecSamplerConfig) -> None:
    assert cfg.n_min <= spec.n <= cfg.n_max
    assert 1 <= len(spec.schedule) <= min(cfg.turns_max, spec.n)
    assert sum(spec.schedule) <= spec.n
    assert all(1 <= k <= cfg.max_marks_per_turn for k in spec.schedule)


def test_sampler_handles_training_8h_range() -> None:
    cfg = SpecSamplerConfig(
        n_min=8,
        n_max=32,
        turns_min=4,
        turns_max=12,
        fill_min=0.55,
        fill_max=0.95,
        max_marks_per_turn=8,
    )
    rng = np.random.default_rng(20260512)

    for _ in range(10_000):
        assert_valid_sample(cfg.sample(rng), cfg)


def test_sampler_handles_turn_max_above_small_n() -> None:
    cfg = SpecSamplerConfig(
        n_min=3,
        n_max=5,
        turns_min=4,
        turns_max=12,
        fill_min=0.8,
        fill_max=0.95,
        max_marks_per_turn=2,
    )
    rng = np.random.default_rng(7)

    for _ in range(1000):
        spec = cfg.sample(rng)
        assert cfg.n_min <= spec.n <= cfg.n_max
        assert len(spec.schedule) <= spec.n
        assert sum(spec.schedule) <= spec.n


def test_sample_unique_specs_returns_valid_unique_specs() -> None:
    cfg = SpecSamplerConfig(
        n_min=8,
        n_max=12,
        turns_min=3,
        turns_max=8,
        fill_min=0.5,
        fill_max=0.95,
        max_marks_per_turn=5,
    )

    specs = sample_unique_specs(cfg, 100, seed=17)
    assert len(specs) == 100
    assert len({(s.n, s.schedule) for s in specs}) == 100
    for spec in specs:
        assert_valid_sample(spec, cfg)


@pytest.mark.parametrize(
    ("total", "parts", "max_part"),
    [(2, 3, 2), (7, 3, 2), (0, 1, 1), (3, 0, 2)],
)
def test_bounded_composition_rejects_impossible_inputs(
    total: int, parts: int, max_part: int
) -> None:
    with pytest.raises(ValueError):
        _bounded_composition(total, parts, max_part, np.random.default_rng(0))


@pytest.mark.parametrize(
    ("total", "weights", "max_part"),
    [(2, [1, 1, 1], 2), (7, [1, 1, 1], 2), (0, [1], 1), (3, [], 2)],
)
def test_weighted_bounded_composition_rejects_impossible_inputs(
    total: int, weights: list[float], max_part: int
) -> None:
    with pytest.raises(ValueError):
        _weighted_bounded_composition(
            total, weights, max_part, np.random.default_rng(0)
        )
