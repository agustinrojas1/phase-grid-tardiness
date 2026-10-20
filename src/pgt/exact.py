"""Exact oracles: permutation brute force and a time-indexed MIP solved by HiGHS.

Both rely on the same fact: the objective is regular, so for any job order the
left-justified schedule is optimal for that order, and its starts are all of the form
r_i + k p with k < N.
"""

import itertools
import time
from dataclasses import dataclass

import highspy
import numpy as np
import scipy.sparse as sp

from pgt.bounds import check_float_exact
from pgt.instance import INT, Instance
from pgt.schedule import Schedule, left_justify


@dataclass(frozen=True)
class ExactResult:
    F: int
    schedule: Schedule
    optimal: bool
    status: str
    lower_bound: float  # proven lower bound on F* (== F when optimal)
    seconds: float


class NoIncumbent(RuntimeError):
    """The MIP stopped (time limit) before finding any feasible schedule; pass a
    warm start to rule this out."""


def brute_force(inst: Instance, max_n: int = 10, chunk: int = 200_000) -> ExactResult:
    """Enumerate all N! orders, left-justify each, vectorised over chunks of orders."""
    n = inst.n
    if n > max_n:
        raise ValueError(f"brute force refuses N = {n} > {max_n}")
    if inst.n * inst.max_cost() >= 2**62:
        raise OverflowError("costs too large for int64 brute force")
    t0 = time.perf_counter()
    perms = itertools.permutations(range(n))
    best_F, best_order = None, None
    while block := list(itertools.islice(perms, chunk)):
        P = np.array(block, dtype=np.intp)
        rP, dP, wP = inst.r[P], inst.d[P], inst.w[P]
        s = np.empty_like(rP)
        s[:, 0] = rP[:, 0]
        for k in range(1, n):
            s[:, k] = np.maximum(rP[:, k], s[:, k - 1] + inst.p)
        cost = (wP * np.maximum(s + inst.p - dP, 0)).sum(axis=1)
        i = int(np.argmin(cost))
        if best_F is None or cost[i] < best_F:
            best_F, best_order = int(cost[i]), P[i]
    schedule = left_justify(inst, best_order)
    assert schedule.F(inst) == best_F
    return ExactResult(best_F, schedule, True, "optimal", float(best_F), time.perf_counter() - t0)


def candidate_starts(inst: Instance) -> np.ndarray:
    """T = {r_i + k p : 1 <= i <= N, 0 <= k < N}: contains every left-justified start."""
    return np.unique(inst.r[:, None] + inst.p * np.arange(inst.n, dtype=INT)[None, :])


def milp(
    inst: Instance,
    time_limit: float = 60.0,
    warm_start: Schedule | None = None,
    threads: int = 1,
    verbose: bool = False,
) -> ExactResult:
    """Time-indexed MIP over the candidate start set T, solved with HiGHS.

    x_{j,t} = 1 iff job j starts at t (only t >= r_j). Each job starts once; for every
    t in T at most one job starts in (t - p, t], which forbids overlap because all jobs
    have length p.

    `warm_start` hands HiGHS an incumbent (its order is left-justified first so every
    start lies in T), so a time-limited run still returns a schedule and a gap. HiGHS
    defaults to one thread per core; keep `threads=1` when running many solves in
    parallel processes.
    """
    check_float_exact(inst)
    t0 = time.perf_counter()
    T = candidate_starts(inst)
    jobs, cols = np.nonzero(T[None, :] >= inst.r[:, None])
    n_var = len(jobs)
    cost = inst.w[jobs] * np.maximum(T[cols] + inst.p - inst.d[jobs], 0)

    # Rows 0..N-1: assignment. Rows N + u: window (T[u] - p, T[u]]. Variable on slot c
    # appears in window rows u = c .. hi[c] - 1, hi[c] = first slot with T >= T[c] + p.
    hi = np.searchsorted(T, T + inst.p, side="left")
    span = hi[cols] - cols
    win_var = np.repeat(np.arange(n_var), span)
    offset = np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span)
    win_row = np.repeat(cols, span) + offset
    rows = np.concatenate([jobs, inst.n + win_row])
    vars_ = np.concatenate([np.arange(n_var), win_var])
    A = sp.csc_array((np.ones(len(rows)), (rows, vars_)), shape=(inst.n + len(T), n_var))

    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n_var, A.shape[0]
    lp.col_cost_ = cost.astype(np.float64)
    lp.col_lower_, lp.col_upper_ = np.zeros(n_var), np.ones(n_var)
    lp.row_lower_ = np.concatenate([np.ones(inst.n), np.full(len(T), -highspy.kHighsInf)])
    lp.row_upper_ = np.ones(A.shape[0])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_ = A.indptr, A.indices
    lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n_var

    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    h.setOptionValue("threads", threads)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.passModel(lp)
    if warm_start is not None:
        ws = left_justify(inst, warm_start.order()).starts
        slot = {int(t): k for k, t in enumerate(T)}
        index = {(int(j), int(c)): v for v, (j, c) in enumerate(zip(jobs, cols, strict=True))}
        x0 = np.zeros(n_var)
        x0[[index[j, slot[int(ws[j])]] for j in range(inst.n)]] = 1.0
        sol = highspy.HighsSolution()
        sol.col_value = x0
        h.setSolution(sol)
    h.run()
    seconds = time.perf_counter() - t0

    model_status = h.getModelStatus()
    info = h.getInfo()
    if info.primal_solution_status != highspy.SolutionStatus.kSolutionStatusFeasible:
        raise NoIncumbent(f"HiGHS: {h.modelStatusToString(model_status)}")
    chosen = np.asarray(h.getSolution().col_value) > 0.5
    starts = np.empty(inst.n, dtype=INT)
    starts[jobs[chosen]] = T[cols[chosen]]
    assert np.bincount(jobs[chosen], minlength=inst.n).tolist() == [1] * inst.n
    schedule = Schedule(starts)
    assert schedule.is_feasible(inst), schedule.violations(inst)
    F = schedule.F(inst)
    optimal = model_status == highspy.HighsModelStatus.kOptimal
    if optimal:
        status, lower = "optimal", float(F)
    else:
        status = (
            "time_limit"
            if model_status == highspy.HighsModelStatus.kTimeLimit
            else (h.modelStatusToString(model_status))
        )
        lower = max(float(info.mip_dual_bound), 0.0)
    return ExactResult(F, schedule, optimal, status, lower, seconds)
