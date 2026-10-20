from hypothesis import given, settings
from hypothesis import strategies as st

from pgt import Instance, bounds, exact, solve


@st.composite
def instances(draw, max_n: int = 7) -> Instance:
    n = draw(st.integers(1, max_n))
    p = draw(st.integers(1, 6))
    ints = lambda hi: st.lists(st.integers(0, hi), min_size=n, max_size=n)  # noqa: E731
    r = draw(ints(4 * p * n))
    d = draw(ints(5 * p * n))
    w = draw(st.lists(st.integers(1, 6), min_size=n, max_size=n))
    return Instance(p, r, d, w)


@settings(max_examples=400, deadline=None)
@given(instances())
def test_feasible_bounded_and_deterministic(inst):
    res = solve(inst, left_justify=True)
    assert res.schedule.is_feasible(inst) and res.lj_schedule.is_feasible(inst)
    F_opt = exact.brute_force(inst).F
    assert F_opt <= res.F_lj <= res.F_alg
    assert all(bounds.check_bounds(inst, res.F_alg, F_opt))
    again = solve(inst, left_justify=True)
    assert again.schedule == res.schedule and again.best_phase == res.best_phase


@settings(max_examples=150, deadline=None)
@given(instances(max_n=6))
def test_milp_matches_brute_force(inst):
    assert exact.milp(inst).F == exact.brute_force(inst).F


@settings(max_examples=200, deadline=None)
@given(instances(), st.integers(0, 50))
def test_shifting_time_shifts_nothing(inst, shift):
    """Translating all releases and due dates by a multiple of p leaves F_alg unchanged."""
    moved = Instance(inst.p, inst.r + shift * inst.p, inst.d + shift * inst.p, inst.w)
    assert solve(moved).F_alg == solve(inst).F_alg
