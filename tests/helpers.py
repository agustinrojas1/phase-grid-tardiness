import numpy as np

from pgt import instance as gen


def random_small(rng: np.random.Generator, max_n: int = 8, max_p: int = 7) -> gen.Instance:
    """Small instance with mixed congestion and slack, the shape most of the tests use."""
    n = int(rng.integers(1, max_n + 1))
    p = int(rng.integers(1, max_p + 1))
    rho = float(rng.choice([0.2, 0.5, 1.0, 2.0]))
    tau = float(rng.choice([0.0, 0.3, 1.0]))
    return gen.uniform(n, p, rho, tau, rng, w_max=int(rng.choice([1, 3, 10])))
