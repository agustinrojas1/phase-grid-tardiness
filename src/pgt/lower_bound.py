"""An a-posteriori lower bound on F* from rounding starts DOWN to a grid.

Fix a phase a in {0, ..., p-1}. For any feasible schedule, moving every start down to the
grid a + pZ keeps the jobs pairwise disjoint (starts p apart stay p apart), lowers every
cost term, and leaves job j at or above R_j^down(a) = a + p * floor((r_j - a) / p), the last
grid point <= r_j. Left-justifying that grid schedule (still on the grid, still above the
rounded releases, still no costlier) lands every start in {R_i^down(a) + k p : k < N}. So
the min-cost assignment of jobs to those slots, with job j eligible for t >= R_j^down(a),
is at most F*. The bound is the maximum over the phases considered; any subset of residues
is valid and the release residues are the default.

Other residues never help: if a is not a release residue, moving the grid up by one unit
raises every R_j^down by one, so every slot and every cost term go up and the phase a + 1
bound is at least the phase a bound. Repeating until a release residue is reached shows
that every phase is dominated by the next release residue above it (cyclically), hence
max over range(p) == max over the release residues. `phases=range(p)` is still accepted,
for inspecting the per-phase values.

The assignment is the same rectangular problem the algorithm solves (`pgt.phase_grid`),
only with rounded-down releases in place of rounded-up ones.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from pgt.bounds import check_float_exact
from pgt.instance import Instance
from pgt.phase_grid import cost_matrices, residues, slots


@dataclass(frozen=True)
class LowerBound:
    value: int
    per_phase: dict[int, int]  # phase -> its bound, in the order the phases were given
    seconds: float

    @property
    def best_phase(self) -> int:
        """The first phase attaining the maximum."""
        return next(a for a, v in self.per_phase.items() if v == self.value)


def round_down(r: np.ndarray, a: int, p: int) -> np.ndarray:
    """a + p * floor((r - a) / p), the last grid point <= r, in exact integer arithmetic."""
    return a + p * ((r - a) // p)


def lower_bound_phase(inst: Instance, a: int) -> int:
    """Min cost of assigning jobs to the slots of the rounded-down grid a + pZ."""
    p = inst.p
    R = round_down(inst.r, a % p, p)
    # R >= r - (p - 1) can be negative; shift releases, due dates and slots together by p,
    # which changes no cost term, so `cost_matrices` (edge iff t >= release) applies as is.
    relaxed = Instance(p, R + p, inst.d + p, inst.w)
    t = slots(relaxed.r, inst.n, p)
    C, C_int = cost_matrices(relaxed, t)
    rows, cols = linear_sum_assignment(C)
    return int(C_int[rows, cols].sum())


def normalize_phases(inst: Instance, phases: Iterable[int] | None) -> list[int]:
    """Residues mod p, deduplicated, order kept; None means the release residues."""
    if phases is None:
        return [int(a) for a in residues(inst)]
    out: list[int] = []
    for a in phases:
        a = int(a) % inst.p
        if a not in out:
            out.append(a)
    if not out:
        raise ValueError("need at least one phase")
    return out


def lower_bound_detail(inst: Instance, phases: Iterable[int] | None = None) -> LowerBound:
    """Per-phase bounds and their maximum. `phases=range(p)` uses every residue."""
    check_float_exact(inst)
    t0 = time.perf_counter()
    per_phase = {a: lower_bound_phase(inst, a) for a in normalize_phases(inst, phases)}
    return LowerBound(max(per_phase.values()), per_phase, time.perf_counter() - t0)


def lower_bound(inst: Instance, phases: Iterable[int] | None = None) -> int:
    """max over phases of the round-down assignment bound; never exceeds F*."""
    return lower_bound_detail(inst, phases).value


def certified_gap(inst: Instance, F: int, phases: Iterable[int] | None = None) -> int:
    """F - LB: how far a schedule with objective F can be from the optimum, at most.

    `F` is typically `Result.F_alg` or `Result.F_lj`; the result is an a-posteriori
    additive bound on F - F* that needs no exact oracle.
    """
    return F - lower_bound(inst, phases)
