import numpy as np
import pytest
from helpers import random_small

from pgt import Instance, exact
from pgt.heuristics import HEURISTICS, atc, dispatch, edd, wspt


def test_brute_force_small_example():
    # p = 2; job 0 heavy and due early but released late
    inst = Instance(p=2, r=[1, 0], d=[3, 4], w=[10, 1])
    res = exact.brute_force(inst)
    # best: idle until 1, run job 0 at [1,3), job 1 at [3,5) -> T = (0, 1)
    assert res.F == 1 and res.schedule.starts.tolist() == [1, 3]


def test_candidate_starts():
    inst = Instance(p=3, r=[0, 4], d=[0, 0], w=[1, 1])
    assert exact.candidate_starts(inst).tolist() == [0, 3, 4, 7]


def test_oracles_agree_on_500_instances(rng):
    """Definition of done: brute force and MIP agree on >= 500 random small instances."""
    for _ in range(500):
        inst = random_small(rng, max_n=8)
        bf, mip = exact.brute_force(inst), exact.milp(inst)
        assert mip.optimal
        assert bf.F == mip.F, inst.to_dict()
        assert bf.schedule.is_feasible(inst) and mip.schedule.is_feasible(inst)
        assert mip.schedule.F(inst) == mip.F


def test_brute_force_size_guard():
    inst = Instance(p=1, r=list(range(11)), d=[0] * 11, w=[1] * 11)
    with pytest.raises(ValueError):
        exact.brute_force(inst)


def test_milp_reports_time_limit_status():
    inst = Instance(p=1, r=[0], d=[5], w=[1])
    res = exact.milp(inst, time_limit=10)
    assert res.optimal and res.status == "optimal" and res.lower_bound == res.F == 0


def test_heuristics_feasible_and_not_better_than_optimum(rng):
    for _ in range(100):
        inst = random_small(rng)
        F_opt = exact.brute_force(inst).F
        for name, h in HEURISTICS.items():
            s = h(inst)
            assert s.is_feasible(inst), name
            assert s.F(inst) >= F_opt, name


def test_dispatch_rules_pick_expected_job():
    inst = Instance(p=1, r=[0, 0, 0], d=[5, 1, 3], w=[1, 2, 9])
    assert np.argsort(wspt(inst).starts).tolist() == [2, 1, 0]
    assert np.argsort(edd(inst).starts).tolist() == [1, 2, 0]
    # the machine never idles while a job is waiting
    late = Instance(p=2, r=[5, 0], d=[0, 0], w=[1, 1])
    assert dispatch(late, lambda j, t: (j,)).starts.tolist() == [5, 0]
    assert atc(inst).is_feasible(inst)
