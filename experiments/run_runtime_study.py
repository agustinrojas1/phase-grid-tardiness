"""Runtime of the algorithm (and of the exact MIP) as N grows.

    uv run python experiments/run_runtime_study.py alg   # sequential, for clean timings
    uv run python experiments/run_runtime_study.py mip [--workers 6]

p = 1000 so (almost) every job has its own release residue, i.e. ~N phases.
rho = "spread" places releases N p apart, which makes every phase offer the full N^2 slots
(the worst case of the O(N^5) bound) even though the instance itself is trivial.
"""

import argparse
import time
import zlib

import numpy as np
from common import RESULTS, run

from pgt import exact, solve
from pgt import instance as gen
from pgt.heuristics import HEURISTICS

ALG_NS = (10, 20, 30, 40, 60, 80, 100, 140, 200, 280, 400)
SPREAD_NS = (10, 20, 30, 40, 60, 80, 100, 140)
MIP_NS = (10, 20, 30, 40, 50, 60, 80)
REPS = 5


def build(n: int, p: int, rho: float | str, tau: float, rep: int) -> gen.Instance:
    rng = np.random.default_rng(zlib.crc32(f"runtime({n},{p},{rho},{tau})#{rep}".encode()))
    return gen.uniform(n, p, float(n) if rho == "spread" else rho, tau, rng)


def alg_work(arg: tuple) -> dict:
    n, p, rho, tau, rep = arg
    inst = build(*arg)
    t0 = time.perf_counter()
    res = solve(inst, left_justify=True)
    seconds = time.perf_counter() - t0
    return {
        "n": n, "p": p, "rho": rho, "tau": tau, "rep": rep,
        "phases": len(res.phases),
        "slots_max": max(o.n_slots for o in res.phases),
        "slots_mean": float(np.mean([o.n_slots for o in res.phases])),
        "t_alg": seconds,
        "t_phase_max": max(o.seconds for o in res.phases),
        "F_alg": res.F_alg, "F_lj": res.F_lj,
    }  # fmt: skip


def mip_work(arg: tuple) -> dict:
    (n, p, rho, tau, rep), time_limit = arg
    inst = build(n, p, rho, tau, rep)
    res = solve(inst, left_justify=True)
    warm = min([res.lj_schedule, *(h(inst) for h in HEURISTICS.values())], key=lambda s: s.F(inst))
    ex = exact.milp(inst, time_limit, warm_start=warm, threads=1)
    T = exact.candidate_starts(inst)
    return {
        "n": n, "p": p, "rho": rho, "tau": tau, "rep": rep,
        "mip_vars": int((T[None, :] >= inst.r[:, None]).sum()),
        "t_alg": res.seconds, "t_mip": ex.seconds, "status": ex.status,
        "F_alg": res.F_alg, "F_lj": res.F_lj, "F_opt": ex.F, "lower_bound": ex.lower_bound,
    }  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["alg", "mip"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=600.0)
    args = ap.parse_args()
    if args.part == "alg":
        cfgs = [
            (n, 1000, rho, 0.25, rep)
            for n in ALG_NS
            for rho in (0.25, 1.0, 4.0)
            for rep in range(REPS)
        ]
        cfgs += [(n, 1000, "spread", 0.25, rep) for n in SPREAD_NS for rep in range(REPS)]
        # one worker by default: timings are not disturbed by neighbours
        run(((str(c), c) for c in cfgs), alg_work, RESULTS / "runtime_alg.csv", args.workers or 1)
    else:
        cfgs = [
            (n, 20, rho, 0.25, rep) for n in MIP_NS for rho in (0.25, 1.0) for rep in range(REPS)
        ]
        tasks = ((str(c), (c, args.time_limit)) for c in sorted(cfgs, key=lambda c: -c[0]))
        run(tasks, mip_work, RESULTS / "runtime_mip.csv", args.workers or 6)
