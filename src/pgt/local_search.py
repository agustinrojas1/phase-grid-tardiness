"""Local search over job orders, and `solve_plus`: the algorithm plus everything cheap.

A job order determines its best schedule (left-justify, since the objective is regular),
so the search space is permutations. The neighbourhood is every swap of two positions and
every insertion (move one job to another position). First improvement: the first
improving neighbour, in a fixed scan order, is applied and the scan restarts, until no
neighbour improves. Neighbours are evaluated in vectorised batches the way
`pgt.exact.brute_force` evaluates orders, so the result is deterministic.

`solve_plus` keeps the algorithm as specified untouched and only adds to it: it starts the
search from the best of the left-justified order of every phase and ATC, and returns the
best schedule seen together with the round-down lower bound. Because the winning phase's
left-justified schedule is among the candidates, F* <= F_plus <= F_lj <= F_alg, so the
Theorem 2 guarantee carries over to F_plus.
"""

import functools
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pgt.heuristics import atc_best
from pgt.instance import Instance
from pgt.lower_bound import LowerBound, lower_bound_detail
from pgt.phase_grid import Result, solve
from pgt.schedule import Schedule, left_justify

CHUNK = 20_000  # neighbours evaluated per batch (bounds memory at N x CHUNK int64 x 4)


@functools.lru_cache(maxsize=64)
def neighbour_moves(n: int) -> np.ndarray:
    """Position permutations of every swap and insertion, one per row, in scan order.

    Row m applied to an order gives neighbour `order[moves[m]]`. Swaps (i, j), i < j, come
    first in lexicographic order; then insertions of position i to position j for |i-j| > 1
    (adjacent insertions are swaps). Independent of the order itself, so cached per N.
    """
    base = np.arange(n, dtype=np.intp)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            m = base.copy()
            m[i], m[j] = j, i
            rows.append(m)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                m = base.tolist()
                m.insert(j, m.pop(i))
                rows.append(np.array(m, dtype=np.intp))
    if not rows:
        return np.empty((0, n), dtype=np.intp)
    out = np.stack(rows)
    out.setflags(write=False)
    return out


def order_costs(inst: Instance, P: np.ndarray) -> np.ndarray:
    """F of the left-justified schedule of every order (row) in P, vectorised."""
    rP, dP, wP = inst.r[P], inst.d[P], inst.w[P]
    s = np.empty_like(rP)
    s[:, 0] = rP[:, 0]
    for k in range(1, P.shape[1]):
        s[:, k] = np.maximum(rP[:, k], s[:, k - 1] + inst.p)
    return (wP * np.maximum(s + inst.p - dP, 0)).sum(axis=1)


def first_improving(inst: Instance, order: np.ndarray, F: int) -> np.ndarray | None:
    """The first neighbour of `order` (in scan order) with objective < F, or None."""
    moves = neighbour_moves(len(order))
    for lo in range(0, len(moves), CHUNK):
        block = moves[lo : lo + CHUNK]
        better = np.flatnonzero(order_costs(inst, order[block]) < F)
        if len(better):
            return order[block[better[0]]]
    return None


def local_search(inst: Instance, order: Any, max_steps: int | None = None) -> Schedule:
    """First-improvement swap + insertion descent from `order`; never worse than its start."""
    if inst.n * inst.max_cost() >= 2**62:
        raise OverflowError("costs too large for int64 local search")
    order = np.asarray(order, dtype=np.intp)
    if sorted(order.tolist()) != list(range(inst.n)):
        raise ValueError("order must be a permutation of the jobs")
    F = int(order_costs(inst, order[None, :])[0])
    steps = 0
    while max_steps is None or steps < max_steps:
        nxt = first_improving(inst, order, F)
        if nxt is None:
            break
        order, steps = nxt, steps + 1
        F = int(order_costs(inst, order[None, :])[0])
    schedule = left_justify(inst, order)
    assert schedule.F(inst) == F
    return schedule


@dataclass(frozen=True)
class PlusResult:
    result: Result  # the algorithm as specified, with its left-justified schedule
    schedule: Schedule  # best schedule found
    F_plus: int
    Phi_plus: int
    start: str  # which candidate the search started from ("phase a" or "ATC")
    F_start: int
    lb: LowerBound
    seconds: float

    @property
    def F_alg(self) -> int:
        return self.result.F_alg

    @property
    def lower_bound(self) -> int:
        return self.lb.value

    @property
    def gap(self) -> int:
        """Certified a-posteriori bound on F_plus - F*."""
        return self.F_plus - self.lb.value

    @property
    def gap_alg(self) -> int:
        """Certified a-posteriori bound on F_alg - F*."""
        return self.result.F_alg - self.lb.value


def candidates(inst: Instance, result: Result) -> list[tuple[str, Schedule]]:
    """Left-justified order of every phase, then ATC with its best look-ahead."""
    out = [(f"phase {o.phase}", left_justify(inst, o.schedule.order())) for o in result.phases]
    out.append(("ATC", atc_best(inst)))
    return out


def solve_plus(
    inst: Instance,
    lb_phases: Iterable[int] | None = None,
    search: bool = True,
) -> PlusResult:
    """Phase-grid algorithm, left-justify every phase, ATC, local search from the best,
    plus the round-down lower bound (`lb_phases=range(p)` for the strongest one)."""
    t0 = time.perf_counter()
    result = solve(inst, left_justify=True)
    cands = candidates(inst, result)
    name, start = min(cands, key=lambda c: c[1].F(inst))  # first wins ties
    F_start = start.F(inst)
    best = local_search(inst, start.order()) if search else start
    F_plus = best.F(inst)
    if F_plus > F_start:  # cannot happen; keep the guarantee independent of the search
        best, F_plus = start, F_start
    lb = lower_bound_detail(inst, lb_phases)
    return PlusResult(
        result=result,
        schedule=best,
        F_plus=F_plus,
        Phi_plus=F_plus + inst.p * inst.W,
        start=name,
        F_start=F_start,
        lb=lb,
        seconds=time.perf_counter() - t0,
    )
