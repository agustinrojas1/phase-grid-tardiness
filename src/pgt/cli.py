"""Command line: pgt solve | bench | verify | gen."""

import argparse
import json
import statistics
import sys
import time

import numpy as np

from pgt import bounds, exact
from pgt import instance as gen
from pgt.instance import Instance
from pgt.local_search import solve_plus
from pgt.phase_grid import solve


def cmd_solve(args: argparse.Namespace) -> int:
    inst = Instance.load(args.instance)
    res = solve(inst, left_justify=args.left_justify or args.exact)
    out: dict = {
        "n": inst.n,
        "p": inst.p,
        "phases": len(res.phases),
        "best_phase": res.best_phase,
        "F_alg": res.F_alg,
        "Phi_alg": res.Phi_alg,
        "seconds": round(res.seconds, 6),
        "starts": res.schedule.to_dict()["starts"],
    }
    if args.left_justify:
        lj = res.lj_schedule.to_dict()["starts"]
        out |= {"F_lj": res.F_lj, "Phi_lj": res.Phi_lj, "starts_lj": lj}
    plus = None
    if args.plus:
        plus = solve_plus(inst)
        out |= {
            "F_plus": plus.F_plus,
            "Phi_plus": plus.Phi_plus,
            "plus_start": plus.start,
            "LB": plus.lower_bound,
            "LB_phases": len(plus.lb.per_phase),
            "gap_certified": plus.gap,  # F_plus - LB >= F_plus - F*
            "gap_certified_alg": plus.gap_alg,  # F_alg - LB >= F_alg - F*
            "seconds_plus": round(plus.seconds, 6),
            "starts_plus": plus.schedule.to_dict()["starts"],
        }
    if args.exact:
        if inst.n <= 8:
            ex = exact.brute_force(inst)
        else:
            ex = exact.milp(inst, args.time_limit, warm_start=res.lj_schedule, threads=0)
        out |= {
            "F_opt": ex.F,
            "opt_status": ex.status,
            "ratio": float(bounds.ratio(inst, res.F_alg, ex.F)),
            "ratio_bound": float(bounds.ratio_bound(inst)),
            "opt_lower_bound": ex.lower_bound,
            "D": float(bounds.additive_gap(inst)),
        }
        if plus is not None:
            out |= {"gap_plus": plus.F_plus - ex.F, "lb_valid": plus.lower_bound <= ex.F}
    if args.out:
        best = plus.schedule if plus else res.lj_schedule if args.left_justify else res.schedule
        best.save(args.out)
    if args.json:
        print(json.dumps(out))
    else:
        width = max(map(len, out))
        for k, v in out.items():
            print(f"{k:<{width}}  {v}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    print(f"{'N':>5} {'phases':>7} {'median s':>10} {'max s':>10}")
    for n in args.n:
        times, phases = [], []
        for _ in range(args.count):
            inst = gen.uniform(n, args.p, args.rho, args.tau, rng)
            t0 = time.perf_counter()
            res = solve(inst, left_justify=True)
            times.append(time.perf_counter() - t0)
            phases.append(len(res.phases))
        med = statistics.median(times)
        print(f"{n:>5} {statistics.median(phases):>7.0f} {med:>10.4f} {max(times):>10.4f}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Cross-check both oracles and Theorem 2 on random small instances."""
    rng = np.random.default_rng(args.seed)
    failures = 0
    for i in range(args.count):
        n = int(rng.integers(1, args.max_n + 1))
        p = int(rng.integers(1, 8))
        rho, tau = float(rng.choice([0.2, 0.5, 1, 2])), float(rng.choice([0, 0.3, 1]))
        inst = gen.uniform(n, p, rho, tau, rng)
        bf, mip = exact.brute_force(inst), exact.milp(inst)
        res = solve(inst, left_justify=True)
        problems = []
        if bf.F != mip.F:
            problems.append(f"oracles disagree: brute {bf.F} vs MIP {mip.F}")
        if not res.schedule.is_feasible(inst):
            problems.append("infeasible schedule")
        if not all(bounds.check_bounds(inst, res.F_alg, bf.F)):
            problems.append(f"Theorem 2 violated: F_alg {res.F_alg}, F* {bf.F}")
        if not bf.F <= res.F_lj <= res.F_alg:
            problems.append("left-justify order violated")
        if problems:
            failures += 1
            print(f"instance {i}: {'; '.join(problems)}\n  {json.dumps(inst.to_dict())}")
    print(f"{args.count - failures}/{args.count} instances OK")
    return 1 if failures else 0


def cmd_gen(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    if args.family == "tight":
        inst = gen.tight_family(args.n)
    else:
        inst = gen.uniform(args.n, args.p, args.rho, args.tau, rng)
    if args.out:
        inst.save(args.out)
    else:
        print(json.dumps(inst.to_dict()))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pgt", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("solve", help="run the phase-grid algorithm on an instance JSON")
    s.add_argument("instance")
    s.add_argument("--left-justify", action="store_true", help="apply the left-justify post-pass")
    s.add_argument(
        "--plus",
        action="store_true",
        help="also run solve_plus (left-justify every phase, ATC, local search) and print "
        "F_plus, the round-down lower bound LB and the certified gap F_plus - LB",
    )
    s.add_argument("--exact", action="store_true", help="also compute F* (brute force or MIP)")
    s.add_argument("--time-limit", type=float, default=60.0, help="MIP time limit in seconds")
    s.add_argument("--out", help="write the schedule JSON here")
    s.add_argument("--json", action="store_true", help="print one JSON object")
    s.set_defaults(func=cmd_solve)

    b = sub.add_parser("bench", help="runtime of the algorithm on random uniform instances")
    b.add_argument("--n", type=int, nargs="+", default=[10, 20, 40])
    b.add_argument("--count", type=int, default=5)
    b.add_argument("--p", type=int, default=20)
    b.add_argument("--rho", type=float, default=1.0)
    b.add_argument("--tau", type=float, default=0.5)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_bench)

    v = sub.add_parser("verify", help="cross-check oracles and Theorem 2 on random instances")
    v.add_argument("--count", type=int, default=500)
    v.add_argument("--max-n", type=int, default=8)
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=cmd_verify)

    g = sub.add_parser("gen", help="write a random instance JSON")
    g.add_argument("family", choices=["uniform", "tight"])
    g.add_argument("--n", type=int, default=10)
    g.add_argument("--p", type=int, default=5)
    g.add_argument("--rho", type=float, default=1.0)
    g.add_argument("--tau", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", "-o")
    g.set_defaults(func=cmd_gen)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
