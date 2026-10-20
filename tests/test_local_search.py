"""Local search and solve_plus: feasible, never worse than the algorithm, locally optimal."""

import numpy as np
from helpers import random_small
from hypothesis import given, settings
from hypothesis import strategies as st

from pgt import Instance, bounds, exact, left_justify, local_search, lower_bound, solve
from pgt import instance as gen
from pgt.local_search import neighbour_moves, order_costs, solve_plus


def test_neighbour_moves_are_distinct_permutations():
    for n in range(1, 8):
        M = neighbour_moves(n)
        n_swaps = n * (n - 1) // 2
        n_ins = sum(1 for i in range(n) for j in range(n) if abs(i - j) > 1)
        assert M.shape == (n_swaps + n_ins, n)
        rows = {tuple(m) for m in M.tolist()}
        assert len(rows) == len(M)  # no duplicate neighbour
        assert tuple(range(n)) not in rows  # no identity move
        assert all(sorted(m) == list(range(n)) for m in M.tolist())


def test_order_costs_matches_left_justify(rng):
    for _ in range(50):
        inst = random_small(rng, max_n=8)
        P = np.stack([rng.permutation(inst.n) for _ in range(5)])
        expected = [left_justify(inst, o).F(inst) for o in P]
        assert order_costs(inst, P).tolist() == expected


def test_local_search_never_worse_and_locally_optimal(rng):
    for _ in range(100):
        inst = random_small(rng, max_n=8)
        start = rng.permutation(inst.n)
        s = local_search(inst, start)
        assert s.is_feasible(inst)
        assert s.F(inst) <= left_justify(inst, start).F(inst)
        assert s == left_justify(inst, s.order())  # a left-justified order
        # no single swap or insertion improves it
        order = s.order()
        if inst.n > 1:
            neigh = order_costs(inst, order[neighbour_moves(inst.n)])
            assert (neigh >= s.F(inst)).all()


def test_local_search_from_optimal_order_stays_optimal(rng):
    for _ in range(50):
        inst = random_small(rng, max_n=7)
        ex = exact.brute_force(inst)
        assert local_search(inst, ex.schedule.order()).F(inst) == ex.F


def test_max_steps_is_a_cap():
    rng = np.random.default_rng(3)
    inst = gen.uniform(12, 5, 1.0, 0.0, rng)
    start = rng.permutation(12)
    F0 = left_justify(inst, start).F(inst)
    s1 = local_search(inst, start, max_steps=1)
    assert s1.F(inst) <= F0
    assert local_search(inst, start, max_steps=0).F(inst) == F0


def test_solve_plus_sandwiched_between_optimum_and_algorithm(rng):
    for _ in range(300):
        inst = random_small(rng, max_n=8)
        pr = solve_plus(inst)
        F_opt = exact.brute_force(inst).F
        assert pr.schedule.is_feasible(inst)
        assert F_opt <= pr.F_plus <= pr.F_start <= pr.result.F_lj <= pr.F_alg, inst.to_dict()
        assert pr.F_plus == pr.schedule.F(inst)
        assert pr.Phi_plus == pr.F_plus + inst.p * inst.W
        assert pr.lower_bound <= F_opt
        assert pr.gap == pr.F_plus - pr.lower_bound >= 0
        assert pr.gap_alg == pr.F_alg - pr.lower_bound >= pr.gap
        # the algorithm as specified is untouched by the extras
        assert pr.result.schedule == solve(inst).schedule
        assert pr.result.F_alg == solve(inst).F_alg
        # Theorem 2 carries over to F_plus
        assert all(bounds.check_bounds(inst, pr.F_plus, F_opt))


def test_solve_plus_without_search_is_best_candidate(rng):
    for _ in range(30):
        inst = random_small(rng, max_n=8)
        pr = solve_plus(inst, search=False)
        assert pr.F_plus == pr.F_start
        assert pr.start.startswith("phase") or pr.start == "ATC"


def test_solve_plus_tight_family_is_exact():
    for n in range(1, 10):
        inst = gen.tight_family(n)
        pr = solve_plus(inst)
        assert pr.F_plus == 0 == pr.lower_bound and pr.F_alg == n * (n - 1) // 2
        assert pr.gap == 0 and pr.gap_alg == pr.F_alg


def test_lb_phases_option():
    rng = np.random.default_rng(5)
    inst = gen.uniform(8, 6, 1.0, 0.0, rng)
    assert solve_plus(inst, lb_phases=range(6)).lower_bound == lower_bound(inst, range(6))
    assert solve_plus(inst).lower_bound == lower_bound(inst)


@st.composite
def instances(draw, max_n: int = 7) -> Instance:
    n = draw(st.integers(1, max_n))
    p = draw(st.integers(1, 6))
    ints = lambda hi: st.lists(st.integers(0, hi), min_size=n, max_size=n)  # noqa: E731
    r = draw(ints(4 * p * n))
    d = draw(ints(5 * p * n))
    w = draw(st.lists(st.integers(1, 6), min_size=n, max_size=n))
    return Instance(p, r, d, w)


@settings(max_examples=300, deadline=None)
@given(instances())
def test_plus_feasible_bounded_and_deterministic(inst):
    pr = solve_plus(inst, lb_phases=range(inst.p))
    assert pr.schedule.is_feasible(inst)
    F_opt = exact.brute_force(inst).F
    assert pr.lower_bound <= F_opt <= pr.F_plus <= pr.result.F_lj <= pr.F_alg
    assert lower_bound(inst) <= pr.lower_bound
    assert all(bounds.check_bounds(inst, pr.F_plus, F_opt))
    again = solve_plus(inst, lb_phases=range(inst.p))
    assert again.schedule == pr.schedule and again.start == pr.start
    assert again.lb.per_phase == pr.lb.per_phase
