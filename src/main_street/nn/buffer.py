"""Sample sources + replay buffer.

A `SampleSource` produces training tuples on demand. The `ReplayBuffer` holds
one or more sources and draws weighted mini-batches from them. The trainer
calls `populate` once per iter and `sample` per gradient step; it doesn't
know what kind of source it has.

Concrete sources live in `sources.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..core import GameState
from .encode import Encoder
from .models import Model


@dataclass(frozen=True, slots=True)
class Sample:
    state: GameState
    pi: dict[int, float]  # target policy: action -> probability
    z: float  # value target from X's perspective, in [-1, 1]


class SampleSource(ABC):
    """Produces samples and lets the buffer draw from them."""

    weight: float

    @abstractmethod
    def populate(
        self, model: Model, encoder: Encoder, rng: np.random.Generator
    ) -> int:
        """Add fresh samples to the source's internal pool. Returns the count
        added. Called once per training iter. Static sources return 0."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> list[Sample]:
        """Draw `n` samples from the current pool. Returns fewer if pool is empty."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Current pool size (for monitoring + weighting)."""


class CyclicBuffer:
    """Pre-allocated FIFO with O(1) random indexing. Used by `SelfPlay` so
    `sample` doesn't pay the deque-indexing cost on large buffers."""

    __slots__ = ("capacity", "_data", "_write", "_full")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: list[Sample] = []
        self._write = 0
        self._full = False

    def extend(self, samples: Iterable[Sample]) -> None:
        for s in samples:
            if not self._full:
                self._data.append(s)
                if len(self._data) >= self.capacity:
                    self._full = True
            else:
                self._data[self._write] = s
                self._write = (self._write + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> Sample:
        return self._data[i]


class ReplayBuffer:
    """Weighted mix of `SampleSource`s. Empty sources are temporarily ignored
    (their weight drops to 0 for that batch), which is what lets self-play
    sources behave correctly during early iters before they've generated games."""

    def __init__(
        self, sources: list[SampleSource], rng: np.random.Generator
    ) -> None:
        if not sources:
            raise ValueError("ReplayBuffer needs at least one source")
        self.sources = sources
        self._rng = rng

    def populate(self, model: Model, encoder: Encoder) -> dict[str, int]:
        return {
            f"source_{i}_added": s.populate(model, encoder, self._rng)
            for i, s in enumerate(self.sources)
        }

    @property
    def sizes(self) -> dict[str, int]:
        return {f"source_{i}_size": s.size for i, s in enumerate(self.sources)}

    def sample(self, batch_size: int) -> list[Sample]:
        weights = np.array(
            [s.weight if s.size > 0 else 0.0 for s in self.sources],
            dtype=np.float64,
        )
        total = weights.sum()
        if total == 0:
            raise RuntimeError("all replay-buffer sources are empty")
        weights = weights / total

        counts = np.floor(weights * batch_size).astype(int)
        leftover = batch_size - int(counts.sum())
        if leftover > 0:
            fracs = weights * batch_size - counts
            order = np.argsort(-fracs)
            for k in range(leftover):
                counts[order[k % len(counts)]] += 1

        out: list[Sample] = []
        for src, n in zip(self.sources, counts.tolist(), strict=False):
            if n > 0:
                out.extend(src.sample(n, self._rng))
        return out
