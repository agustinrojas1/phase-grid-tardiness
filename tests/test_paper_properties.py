"""Statements of the paper checked numerically. A failure here is a bug in our code."""

from fractions import Fraction

import numpy as np
import pytest
from helpers import random_small

from pgt import Instance, bounds, exact, solve
from pgt import instance as gen


@pytest.mark.parametrize("n", range(1, 13))
def test_tight_family(n):
    """Proposition 3: F* = 0, F_alg = N(N-1)/2, ratio exactly 3/2 - 1/(2N)."""
    inst = gen.tight_family(n)
    res = solve(inst, left_justify=True)
    assert len(res.per_phase) == n  # every residue appears
    assert set(res.per_phase.values()) == {n * (n - 1) // 2}  # every phase loses the same
    assert res.F_alg == n * (n - 1) // 2
    assert bounds.ratio(inst, res.F_alg, 0) == bounds.ratio_bound_n(n) == bounds.ratio_bound(inst)
    assert Fraction(res.F_alg) == bounds.additive_gap(inst)  # additive bound is tight
    assert res.F_lj == 0
    if n <= 9:
        assert exact.brute_force(inst).F == 0
    assert exact.milp(inst).F == 0


def test_single_residue_is_exact(rng):
    for _ in range(200):
        n, p = int(rng.integers(1, 9)), int(rng.integers(1, 8))
        a = int(rng.integers(0, p))
        r = a + p * rng.integers(0, n + 1, size=n)
        inst = Instance(p, r, r + rng.integers(0, 3 * p, size=n), rng.integers(1, 10, size=n))
        res = solve(inst)
        assert len(res.phases) == 1
        assert res.F_alg == exact.brute_force(inst).F


def test_theorem2_bounds_vs_brute_force(rng):
    worst = Fraction(0)
    for _ in range(600):
        inst = random_small(rng, max_n=8)
        F_opt = exact.brute_force(inst).F
        res = solve(inst, left_justify=True)
        additive_ok, ratio_ok = bounds.check_bounds(inst, res.F_alg, F_opt)
        assert additive_ok and ratio_ok, inst.to_dict()
        assert F_opt <= res.F_lj <= res.F_alg
        assert bounds.ratio_bound(inst) <= bounds.ratio_bound_n(inst.n)
        worst = max(worst, bounds.ratio(inst, res.F_alg, F_opt))
    assert worst <= Fraction(3, 2)


def test_bound_identities():
    inst = Instance(p=4, r=[0, 1, 2], d=[0, 0, 0], w=[1, 2, 3])
    W, S = 6, 14
    assert bounds.additive_gap(inst) == Fraction(4 * (W * W - S), 2 * W)
    assert bounds.ratio_bound(inst) == Fraction(3, 2) - Fraction(S, 2 * W * W)
    # Additive bound implies the ratio bound: ratio RHS = additive RHS + slack * F*.
    for F_opt in (0, 5, 100):
        F_alg = F_opt + bounds.additive_gap(inst)
        assert bounds.ratio(inst, F_alg, F_opt) <= bounds.ratio_bound(inst)


def test_unweighted_bound_collapses_to_n_form():
    for n in range(1, 10):
        inst = Instance(1, np.arange(n), np.arange(n), np.ones(n, int))
        assert bounds.ratio_bound(inst) == bounds.ratio_bound_n(n)
