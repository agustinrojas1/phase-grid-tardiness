"""The round-down assignment lower bound: valid, tight where it should be, monotone."""

import itertools

import numpy as np
import pytest
from helpers import random_small

from pgt import Instance, certified_gap, exact, lower_bound, solve
from pgt.lower_bound import (
    lower_bound_detail,
    lower_bound_phase,
    normalize_phases,
    round_down,
)
from pgt.phase_grid import residues


def rounded_down_brute_force(inst: Instance, a: int) -> int:
    """Best schedule on a + pZ for the instance with releases R^down(a), by enumerating
    orders and left-justifying on the grid (independent of the assignment)."""
    R = round_down(inst.r, a, inst.p).tolist()
    best = None
    for order in itertools.permutations(range(inst.n)):
        free, F = None, 0
        for j in order:
            s = R[j] if free is None else max(R[j], free)
            free = s + inst.p
            F += int(inst.w[j]) * max(s + inst.p - int(inst.d[j]), 0)
        best = F if best is None else min(best, F)
    return best


def test_round_down():
    r = np.array([0, 1, 4, 5, 6, 10])
    assert round_down(r, 0, 5).tolist() == [0, 0, 0, 5, 5, 10]
    assert round_down(r, 1, 5).tolist() == [-4, 1, 1, 1, 6, 6]
    assert round_down(r, 4, 5).tolist() == [-1, -1, 4, 4, 4, 9]
    for p in range(1, 7):
        for a in range(p):
            R = round_down(np.arange(40), a, p)
            assert ((R - a) % p == 0).all() and (R <= np.arange(40)).all()
            assert (np.arange(40) - R < p).all()


def test_lower_bound_never_exceeds_optimum(rng):
    for _ in range(500):
        inst = random_small(rng, max_n=8)
        F_opt = exact.brute_force(inst).F
        lb = lower_bound(inst)
        assert 0 <= lb <= F_opt, inst.to_dict()
        assert lower_bound(inst, range(inst.p)) <= F_opt, inst.to_dict()


def test_phase_bound_is_grid_optimal_for_rounded_releases(rng):
    for _ in range(150):
        inst = random_small(rng, max_n=6, max_p=5)
        for a in range(inst.p):
            assert lower_bound_phase(inst, a) == rounded_down_brute_force(inst, a)


def test_single_residue_is_tight(rng):
    """One residue: rounding down changes no release, so the bound is the algorithm's own
    (exact) phase value."""
    for _ in range(100):
        n, p = int(rng.integers(1, 9)), int(rng.integers(1, 8))
        a = int(rng.integers(0, p))
        r = a + p * rng.integers(0, n + 1, size=n)
        inst = Instance(p, r, r + rng.integers(0, 3 * p, size=n), rng.integers(1, 10, size=n))
        assert len(residues(inst)) == 1
        F_opt = exact.brute_force(inst).F
        assert lower_bound(inst) == solve(inst).F_alg == F_opt
        assert certified_gap(inst, solve(inst).F_alg) == 0


def test_p_equals_one_is_tight(rng):
    for _ in range(50):
        inst = random_small(rng, max_p=1)
        assert lower_bound(inst) == exact.brute_force(inst).F


def test_monotone_in_phase_set(rng):
    for _ in range(100):
        inst = random_small(rng, max_n=7)
        detail = lower_bound_detail(inst, range(inst.p))
        rel = normalize_phases(inst, None)
        assert set(rel) <= set(detail.per_phase)
        assert detail.value == max(detail.per_phase.values())
        assert lower_bound(inst) == max(detail.per_phase[a] for a in rel)
        assert lower_bound(inst) <= detail.value
        # any subset is bounded by any superset
        subset = rel[: max(1, len(rel) // 2)]
        assert lower_bound(inst, subset) <= lower_bound(inst, rel) <= detail.value
        # per-phase values do not depend on which other phases were asked for
        for a in rel:
            assert lower_bound_detail(inst, [a]).per_phase == {a: detail.per_phase[a]}
        assert detail.per_phase[detail.best_phase] == detail.value


def test_all_residues_equal_release_residues(rng):
    """A non-release phase is dominated by the next release residue above it: shifting
    the grid up by one past a non-release residue raises every rounded-down release and
    every slot, so no cost term can fall. Hence the max over range(p) is already attained
    on the release residues."""
    for _ in range(200):
        inst = random_small(rng, max_n=7)
        detail = lower_bound_detail(inst, range(inst.p))
        rel = normalize_phases(inst, None)
        assert detail.value == lower_bound(inst)
        for a in range(inst.p):
            nxt = next(b for b in rel + [rel[0] + inst.p] if b >= a) % inst.p
            assert detail.per_phase[a] <= detail.per_phase[nxt]
            if a not in rel:
                assert detail.per_phase[a] <= detail.per_phase[(a + 1) % inst.p]


def test_phases_normalised_mod_p():
    inst = Instance(p=4, r=[1, 9, 6], d=[0, 0, 0], w=[1, 1, 1])
    assert normalize_phases(inst, None) == [1, 2]
    assert normalize_phases(inst, [5, 1, -3, 2]) == [1, 2]
    assert lower_bound(inst, [5]) == lower_bound(inst, [1])
    with pytest.raises(ValueError):
        lower_bound(inst, [])


def test_hand_example():
    # Two unit-weight jobs, p = 4, r = (1, 2), d = (5, 6). Optimum: job 0 at 1 (on time),
    # job 1 at 5 (C = 9, d = 6, T = 3), so F* = 3.
    # Phase 1 rounds r = 2 down to 1, both jobs available at 1 on grid 1 + 4Z: order (0, 1)
    # costs 3, order (1, 0) puts job 0 at 5 (C = 9, d = 5, T = 4), so the phase bound is 3.
    # Phase 2 rounds r = 1 down to -2: job 0 at -2 (C = 2) and job 1 at 2 (C = 6) are both
    # on time, so that phase bound is 0.
    inst = Instance(p=4, r=[1, 2], d=[5, 6], w=[1, 1])
    assert exact.brute_force(inst).F == 3
    assert lower_bound_phase(inst, 1) == 3
    assert lower_bound_phase(inst, 2) == 0
    assert lower_bound(inst) == 3 == lower_bound(inst, range(4))


def test_unperturbed_tight_family_bound_is_zero():
    from pgt import instance as gen

    for n in range(1, 10):
        inst = gen.tight_family(n)
        assert lower_bound(inst) == 0 == lower_bound(inst, range(inst.p))


def test_huge_weights_rejected():
    big = int(np.iinfo(np.int64).max) // 2
    inst = Instance(p=3, r=[0, 1, 2], d=[0, 0, 0], w=[big, big, big])
    with pytest.raises(OverflowError):
        lower_bound(inst)
