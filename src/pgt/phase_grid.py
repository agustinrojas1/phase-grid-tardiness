"""The phase-grid assignment algorithm (Chen & Hall, Section 6.2).

For every release residue a mod p, round releases up to the grid a + pZ, offer the slots
R_i(a) + k p (k < N) and solve a rectangular min-cost assignment of jobs to slots. Slots on
one grid are disjoint length-p intervals, so every assignment is a feasible schedule, and
it is optimal among schedules on that grid (Lemma 6). Keep the best phase.
"""

import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from pgt.bounds import check_float_exact
from pgt.instance import INT, Instance
from pgt.schedule import Schedule, left_justify


@dataclass(frozen=True)
class PhaseOutcome:
    phase: int
    schedule: Schedule
    F: int
    n_slots: int
    seconds: float


@dataclass(frozen=True)
class Result:
    best_phase: int
    schedule: Schedule  # the algorithm as specified
    F_alg: int
    Phi_alg: int
    phases: tuple[PhaseOutcome, ...]
    seconds: float
    # Filled when solve(..., left_justify=True): the winning order, left-justified.
    lj_schedule: Schedule | None = None
    F_lj: int | None = None
    Phi_lj: int | None = None

    @property
    def per_phase(self) -> dict[int, int]:
        return {o.phase: o.F for o in self.phases}


def residues(inst: Instance) -> np.ndarray:
    """Sorted distinct r_j mod p."""
    return np.unique(inst.r % inst.p)


def round_up(r: np.ndarray, a: int, p: int) -> np.ndarray:
    """a + p * ceil((r - a) / p), the first grid point >= r, in exact integer arithmetic."""
    return a + p * -((a - r) // p)


def slots(R: np.ndarray, n: int, p: int) -> np.ndarray:
    """Distinct candidate starts {R_i + k p : 0 <= k < n}, sorted."""
    return np.unique(R[:, None] + p * np.arange(n, dtype=INT)[None, :])


def cost_matrices(inst: Instance, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(float cost with inf on forbidden edges, exact int64 twin) of shape (N, len(t))."""
    C_int = inst.w[:, None] * np.maximum(t[None, :] + inst.p - inst.d[:, None], 0)
    C = np.where(t[None, :] >= inst.r[:, None], C_int.astype(np.float64), np.inf)
    return C, C_int


def solve_phase(inst: Instance, a: int) -> PhaseOutcome:
    """Optimal schedule among those whose starts all lie on a + pZ."""
    t0 = time.perf_counter()
    # Every slot is >= R_i >= r_i >= min r, so "drop slots below min r" removes nothing;
    # pruning anything else would break Lemma 6.
    t = slots(round_up(inst.r, a, inst.p), inst.n, inst.p)
    C, C_int = cost_matrices(inst, t)
    rows, cols = linear_sum_assignment(C)
    assert np.array_equal(rows, np.arange(inst.n))
    F = int(C_int[rows, cols].sum())
    return PhaseOutcome(int(a), Schedule(t[cols]), F, len(t), time.perf_counter() - t0)


def solve(inst: Instance, left_justify: bool = False) -> Result:
    """Run the algorithm over all release phases; ties go to the smaller phase."""
    check_float_exact(inst)
    t0 = time.perf_counter()
    outcomes = tuple(solve_phase(inst, int(a)) for a in residues(inst))
    best = min(outcomes, key=lambda o: o.F)  # min() keeps the first, i.e. smallest phase
    result = Result(
        best_phase=best.phase,
        schedule=best.schedule,
        F_alg=best.F,
        Phi_alg=best.F + inst.p * inst.W,
        phases=outcomes,
        seconds=time.perf_counter() - t0,
    )
    if not left_justify:
        return result
    lj = left_justify_schedule(inst, best.schedule)
    F_lj = lj.F(inst)
    return replace(
        result,
        seconds=time.perf_counter() - t0,
        lj_schedule=lj,
        F_lj=F_lj,
        Phi_lj=F_lj + inst.p * inst.W,
    )


def left_justify_schedule(inst: Instance, schedule: Schedule) -> Schedule:
    """Post-pass of Section 6.4: keep the job order, start every job as early as possible."""
    return left_justify(inst, schedule.order())
