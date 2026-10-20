"""Schedules: feasibility, objective evaluation, left-justification."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pgt.instance import INT, Instance


@dataclass(frozen=True, eq=False)
class Schedule:
    """Start time of every job, indexed like the instance arrays."""

    starts: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.starts, dtype=INT).copy()
        arr.setflags(write=False)
        object.__setattr__(self, "starts", arr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schedule):
            return NotImplemented
        return np.array_equal(self.starts, other.starts)

    __hash__ = None  # type: ignore[assignment]

    def order(self) -> np.ndarray:
        """Jobs sorted by start time (ties by index)."""
        return np.argsort(self.starts, kind="stable")

    def completions(self, inst: Instance) -> np.ndarray:
        return self.starts + inst.p

    def tardiness(self, inst: Instance) -> np.ndarray:
        return np.maximum(self.starts + inst.p - inst.d, 0)

    def F(self, inst: Instance) -> int:
        """Total weighted tardiness, in exact (Python) integer arithmetic."""
        return sum(int(w) * int(t) for w, t in zip(inst.w, self.tardiness(inst), strict=True))

    def Phi(self, inst: Instance) -> int:
        """Shifted objective F + p W."""
        return self.F(inst) + inst.p * inst.W

    def violations(self, inst: Instance) -> list[str]:
        if len(self.starts) != inst.n:
            return [f"{len(self.starts)} starts for {inst.n} jobs"]
        out = [
            f"job {j} starts at {self.starts[j]} < r = {inst.r[j]}"
            for j in np.flatnonzero(self.starts < inst.r)
        ]
        order = self.order()
        s = self.starts[order]
        for k in np.flatnonzero(np.diff(s) < inst.p):
            out.append(f"jobs {order[k]} and {order[k + 1]} overlap ({s[k]}, {s[k + 1]})")
        return out

    def is_feasible(self, inst: Instance) -> bool:
        return not self.violations(inst)

    def to_dict(self) -> dict[str, Any]:
        return {"starts": [int(x) for x in self.starts]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schedule:
        return cls(np.asarray(data["starts"], dtype=INT))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Schedule:
        return cls.from_dict(json.loads(Path(path).read_text()))


def left_justify(inst: Instance, order: Any) -> Schedule:
    """Earliest schedule processing jobs in `order`: s_1 = r, s_k = max(r, s_{k-1} + p)."""
    order = np.asarray(order, dtype=np.intp)
    if sorted(order.tolist()) != list(range(inst.n)):
        raise ValueError("order must be a permutation of the jobs")
    starts = np.empty(inst.n, dtype=INT)
    free = None
    for j in order:
        s = int(inst.r[j]) if free is None else max(int(inst.r[j]), free)
        starts[j] = s
        free = s + inst.p
    return Schedule(starts)
