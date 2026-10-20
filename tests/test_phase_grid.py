import itertools

import numpy as np
import pytest
from helpers import random_small

from pgt import Instance, exact, solve
from pgt.phase_grid import residues, round_up, slots, solve_phase


def grid_brute_force(inst: Instance, a: int) -> int:
    """Best schedule with every start on a + pZ, by enumerating orders (independent of
    the assignment): the earliest grid start >= r_j is R_j(a), then left-justify on grid."""
    R = round_up(inst.r, a, inst.p).tolist()
    best = None
    for order in itertools.permutations(range(inst.n)):
        free, F = None, 0
        for j in order:
            s = R[j] if free is None else max(R[j], free)
            free = s + inst.p
            F += int(inst.w[j]) * max(s + inst.p - int(inst.d[j]), 0)
        best = F if best is None else min(best, F)
    return best


def test_round_up():
    r = np.array([0, 1, 4, 5, 6, 10])
    assert round_up(r, 0, 5).tolist() == [0, 5, 5, 5, 10, 10]
    assert round_up(r, 1, 5).tolist() == [1, 1, 6, 6, 6, 11]
    assert round_up(r, 4, 5).tolist() == [4, 4, 4, 9, 9, 14]
    for p in range(1, 7):
        for a in range(p):
            R = round_up(np.arange(40), a, p)
            assert ((R - a) % p == 0).all() and (R >= np.arange(40)).all()
            assert (R - np.arange(40) < p).all()


def test_slots_and_residues():
    inst = Instance(p=4, r=[1, 9, 6], d=[0, 0, 0], w=[1, 1, 1])
    assert residues(inst).tolist() == [1, 2]
    R = round_up(inst.r, 1, 4)
    assert R.tolist() == [1, 9, 9]
    assert slots(R, 3, 4).tolist() == [1, 5, 9, 13, 17]


def test_every_schedule_feasible(rng):
    for _ in range(300):
        inst = random_small(rng, max_n=12)
        res = solve(inst, left_justify=True)
        assert res.schedule.is_feasible(inst)
        assert res.lj_schedule.is_feasible(inst)
        for o in res.phases:
            assert o.schedule.is_feasible(inst)
            assert ((o.schedule.starts - o.phase) % inst.p == 0).all()
            assert o.F == o.schedule.F(inst)
        assert res.F_alg == min(res.per_phase.values()) == res.schedule.F(inst)
        assert res.F_lj <= res.F_alg
        assert res.Phi_alg == res.F_alg + inst.p * inst.W


def test_lemma6_phase_is_grid_optimal(rng):
    for _ in range(150):
        inst = random_small(rng, max_n=6, max_p=5)
        for a in residues(inst):
            assert solve_phase(inst, int(a)).F == grid_brute_force(inst, int(a))


def test_tie_break_smallest_phase():
    # p = 2, two isolated jobs with ample slack: both phases give F = 0
    inst = Instance(p=2, r=[0, 11], d=[100, 100], w=[1, 1])
    res = solve(inst)
    assert res.per_phase == {0: 0, 1: 0}
    assert res.best_phase == 0


def test_deterministic(rng):
    for _ in range(30):
        inst = random_small(rng)
        a, b = solve(inst, True), solve(inst, True)
        assert a.best_phase == b.best_phase and a.schedule == b.schedule
        assert a.lj_schedule == b.lj_schedule


# --- edge cases -------------------------------------------------------------


def test_single_job():
    inst = Instance(p=5, r=[3], d=[4], w=[7])
    res = solve(inst, True)
    assert res.F_alg == 7 * 4 == res.F_lj
    assert res.schedule.starts.tolist() == [3]


def test_all_released_at_zero_is_exact(rng):
    for _ in range(50):
        n = int(rng.integers(1, 8))
        inst = Instance(
            int(rng.integers(1, 6)), np.zeros(n, int), rng.integers(0, 30, n), rng.integers(1, 9, n)
        )
        assert solve(inst).F_alg == exact.brute_force(inst).F


def test_unavoidable_tardiness():
    inst = Instance(p=4, r=[5, 5, 6], d=[2, 3, 1], w=[1, 2, 3])
    res = solve(inst, True)
    assert res.F_alg == exact.brute_force(inst).F > 0


def test_p_equals_one_is_exact(rng):
    for _ in range(50):
        inst = random_small(rng, max_p=1)
        assert inst.p == 1 and solve(inst).F_alg == exact.brute_force(inst).F


def test_duplicate_jobs():
    inst = Instance(p=3, r=[2, 2, 2, 2], d=[5, 5, 5, 5], w=[4, 4, 4, 4])
    res = solve(inst)
    assert sorted(res.schedule.starts.tolist()) == [2, 5, 8, 11]
    assert res.F_alg == 4 * (3 + 6 + 9) == exact.brute_force(inst).F


def test_large_weights_stay_exact():
    w = 2**40 + 1
    inst = Instance(p=3, r=[0, 1, 2], d=[0, 0, 0], w=[w, w - 1, w - 2])
    res = solve(inst)
    assert res.F_alg == exact.brute_force(inst).F
    assert res.F_alg == res.schedule.F(inst)


def test_huge_weights_rejected_not_silently_wrong():
    big = int(np.iinfo(np.int64).max) // 2
    inst = Instance(p=3, r=[0, 1, 2], d=[0, 0, 0], w=[big, big, big])
    with pytest.raises(OverflowError):
        solve(inst)
    with pytest.raises(OverflowError):
        exact.milp(inst)
    with pytest.raises(OverflowError):
        exact.brute_force(inst)
