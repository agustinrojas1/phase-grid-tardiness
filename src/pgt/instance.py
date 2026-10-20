"""Problem instances for 1 | r_j, p_j = p | sum w_j T_j, plus random generators."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

INT = np.int64
INT_MAX = int(np.iinfo(INT).max)


def _int_array(name: str, values: Any) -> np.ndarray:
    raw = list(values) if not isinstance(values, np.ndarray) else values
    arr = np.asarray(raw)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size and np.issubdtype(arr.dtype, np.unsignedinteger) and int(arr.max()) > INT_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == object:
            if not all(isinstance(v, (int, np.integer)) for v in arr):
                raise ValueError(f"{name} must contain integers")
            if any(not -INT_MAX - 1 <= int(v) <= INT_MAX for v in arr):
                raise OverflowError(f"{name} does not fit in int64")
        elif not np.all(np.mod(arr, 1) == 0):
            raise ValueError(f"{name} must contain integers")
    out = arr.astype(INT)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class Instance:
    """N jobs with release r_j, due date d_j, weight w_j and common processing time p."""

    p: int
    r: np.ndarray
    d: np.ndarray
    w: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.p, bool) or not isinstance(self.p, (int, np.integer)):
            raise ValueError("p must be an integer")
        object.__setattr__(self, "p", int(self.p))
        for field in ("r", "d", "w"):
            object.__setattr__(self, field, _int_array(field, getattr(self, field)))
        if self.p <= 0:
            raise ValueError("p must be positive")
        if not (len(self.r) == len(self.d) == len(self.w)):
            raise ValueError("r, d and w must have equal lengths")
        if len(self.r) == 0:
            raise ValueError("an instance needs at least one job")
        if (self.r < 0).any() or (self.d < 0).any():
            raise ValueError("release and due dates must be non-negative")
        if (self.w <= 0).any():
            raise ValueError("weights must be positive")

    @property
    def n(self) -> int:
        return len(self.r)

    @property
    def W(self) -> int:
        return sum(int(x) for x in self.w)

    @property
    def sum_w2(self) -> int:
        return sum(int(x) ** 2 for x in self.w)

    def horizon(self) -> int:
        """Strict upper bound on every start any of our methods considers: max r + N p."""
        return int(self.r.max()) + self.n * self.p

    def max_cost(self) -> int:
        """Upper bound on a single term w_j (t + p - d_j)^+ over starts t < horizon()."""
        return int(self.w.max()) * max(self.horizon() + self.p - int(self.d.min()), 0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instance):
            return NotImplemented
        return (
            self.p == other.p
            and np.array_equal(self.r, other.r)
            and np.array_equal(self.d, other.d)
            and np.array_equal(self.w, other.w)
        )

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "p": self.p,
            "r": [int(x) for x in self.r],
            "d": [int(x) for x in self.d],
            "w": [int(x) for x in self.w],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instance:
        return cls(p=data["p"], r=data["r"], d=data["d"], w=data["w"], name=data.get("name", ""))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Instance:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Generators. Every generator takes a numpy Generator so experiments are seedable.


def uniform(
    n: int, p: int, rho: float, tau: float, rng: np.random.Generator, w_max: int = 10
) -> Instance:
    """r ~ U[0, rho N p], d = r + p + U[0, tau N p], w ~ U[1, w_max] (all integer)."""
    r = rng.integers(0, int(rho * n * p) + 1, size=n)
    d = r + p + rng.integers(0, int(tau * n * p) + 1, size=n)
    w = rng.integers(1, w_max + 1, size=n)
    return Instance(p, r, d, w, name=f"uniform(n={n},p={p},rho={rho},tau={tau})")


def tight_family(n: int) -> Instance:
    """Proposition 3: p = N, r_i = i(2N+1), d_i = r_i + N, w_i = 1 for i = 1..N."""
    r = np.arange(1, n + 1, dtype=INT) * (2 * n + 1)
    return Instance(n, r, r + n, np.ones(n, dtype=INT), name=f"tight(n={n})")


def perturbed_tight(
    n: int,
    rng: np.random.Generator,
    jitter: float = 0.0,
    slack: float = 0.0,
    weight_spread: int = 0,
) -> Instance:
    """Proposition 3 instance with noise.

    jitter: releases move by U[-jitter N, jitter N] (jobs may start to interact once
        jitter is large enough), due dates move with them.
    slack: due dates get U[0, slack N] extra slack.
    weight_spread: weights become U[1, 1 + weight_spread].
    """
    base = tight_family(n)
    j = int(round(jitter * n))
    r = np.maximum(base.r + rng.integers(-j, j + 1, size=n), 0)
    d = r + n + rng.integers(0, int(round(slack * n)) + 1, size=n)
    w = rng.integers(1, weight_spread + 2, size=n)
    return Instance(
        n, r, d, w, name=f"tight(n={n},jitter={jitter},slack={slack},ws={weight_spread})"
    )


def few_phase(
    n: int, p: int, phases: int, rho: float, tau: float, rng: np.random.Generator, w_max: int = 10
) -> Instance:
    """Like `uniform` but releases use only `phases` distinct residues mod p."""
    if not 1 <= phases <= p:
        raise ValueError("need 1 <= phases <= p")
    residues = rng.choice(p, size=phases, replace=False)
    blocks = rng.integers(0, max(int(math.ceil(rho * n)), 1), size=n)
    r = residues[rng.integers(0, phases, size=n)] + p * blocks
    d = r + p + rng.integers(0, int(tau * n * p) + 1, size=n)
    w = rng.integers(1, w_max + 1, size=n)
    return Instance(p, r, d, w, name=f"few_phase(n={n},p={p},k={phases},rho={rho},tau={tau})")
