"""Non-delay dispatch rules, for context only (no guarantee).

At every decision time the machine is free; among released unscheduled jobs the rule
picks the one with the smallest key. If nothing is released, time jumps to the next release.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np

from pgt.instance import INT, Instance
from pgt.schedule import Schedule

Key = Callable[[int, int], tuple]  # (job, time) -> sort key


def dispatch(inst: Instance, key: Key) -> Schedule:
    r = inst.r.tolist()
    pending = set(range(inst.n))
    starts = np.empty(inst.n, dtype=INT)
    t = 0
    while pending:
        ready = [j for j in pending if r[j] <= t]
        if not ready:
            t = min(r[j] for j in pending)
            continue
        j = min(ready, key=lambda j: key(j, t))
        starts[j] = t
        pending.remove(j)
        t += inst.p
    return Schedule(starts)


def wspt(inst: Instance) -> Schedule:
    """Weighted shortest processing time; with equal p this is largest weight first."""
    w, d = inst.w.tolist(), inst.d.tolist()
    return dispatch(inst, lambda j, t: (-w[j], d[j], j))


def edd(inst: Instance) -> Schedule:
    """Earliest due date."""
    w, d = inst.w.tolist(), inst.d.tolist()
    return dispatch(inst, lambda j, t: (d[j], -w[j], j))


def atc(inst: Instance, k: float = 2.0) -> Schedule:
    """Apparent tardiness cost: priority (w_j / p) exp(-max(d_j - p - t, 0) / (k p))."""
    w, d, p = inst.w.tolist(), inst.d.tolist(), inst.p

    def key(j: int, t: int) -> tuple:
        return (-(w[j] / p) * math.exp(-max(d[j] - p - t, 0) / (k * p)), j)

    return dispatch(inst, key)


def atc_best(inst: Instance, ks: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 4.0)) -> Schedule:
    """ATC with the best look-ahead parameter from `ks` (first wins ties)."""
    return min((atc(inst, k) for k in ks), key=lambda s: s.F(inst))


HEURISTICS: dict[str, Callable[[Instance], Schedule]] = {
    "WSPT": wspt,
    "EDD": edd,
    "ATC": atc_best,
}
