"""The Theorem 2 guarantee and helpers around the shifted objective Phi = F + p W.

Everything returns exact `Fraction`s so bound checks never suffer from rounding.
"""

from fractions import Fraction

from pgt.instance import Instance

FLOAT_EXACT = 2**53


def phi(inst: Instance, F: int) -> int:
    return F + inst.p * inst.W


def additive_gap(inst: Instance) -> Fraction:
    """D(I) = p / (2W) * (W^2 - sum w_j^2): F_alg <= F* + D(I)."""
    W = inst.W
    return Fraction(inst.p * (W * W - inst.sum_w2), 2 * W)


def ratio_bound(inst: Instance) -> Fraction:
    """3/2 - sum w_j^2 / (2 W^2): Phi_alg / Phi* is at most this."""
    W = inst.W
    return Fraction(3, 2) - Fraction(inst.sum_w2, 2 * W * W)


def ratio_bound_n(n: int) -> Fraction:
    """Instance-independent version 3/2 - 1/(2N)."""
    return Fraction(3, 2) - Fraction(1, 2 * n)


def ratio(inst: Instance, F_alg: int, F_opt: int) -> Fraction:
    """Phi_alg / Phi*."""
    return Fraction(phi(inst, F_alg), phi(inst, F_opt))


def normalized_loss(inst: Instance, F_alg: int, F_opt: int) -> float:
    """(F_alg - F*) / D(I), in [0, 1] whenever Theorem 2 holds. 0 when D(I) = 0 (N = 1)."""
    D = additive_gap(inst)
    if D == 0:
        return 0.0 if F_alg == F_opt else float("inf")
    return float((F_alg - F_opt) / D)


def check_bounds(inst: Instance, F_alg: int, F_opt: int) -> tuple[bool, bool]:
    """(additive bound holds, ratio bound holds)."""
    return (
        F_alg <= F_opt + additive_gap(inst),
        ratio(inst, F_alg, F_opt) <= ratio_bound(inst),
    )


def check_float_exact(inst: Instance) -> None:
    """Raise unless any sum of N costs over starts < horizon() is exactly representable
    in float64 (and therefore also fits int64)."""
    if inst.n * inst.max_cost() >= FLOAT_EXACT:
        raise OverflowError(
            "cost magnitudes exceed 2^53: float64 assignment would not be exact "
            f"(N * max cost = {inst.n * inst.max_cost()})"
        )
