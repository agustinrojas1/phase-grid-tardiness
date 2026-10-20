"""Algorithm vs exact optimum (time-indexed MIP) on every family in generate.py.

    uv run python experiments/run_ratio_study.py [--workers K] [--time-limit S] [--max-n N]

Writes experiments/results/ratio_<family>.csv (resumable: rerun to continue an
interrupted study).
"""

import argparse
import time
from pathlib import Path

from common import DEFAULT_WORKERS, RESULTS, run
from generate import Spec, all_specs

from pgt import bounds, exact, solve
from pgt.heuristics import HEURISTICS
from pgt.schedule import left_justify

TIME_LIMIT = 300.0


def work(arg: tuple[Spec, float]) -> dict:
    spec, time_limit = arg
    inst = spec.build()
    res = solve(inst, left_justify=True)
    # Left-justify every phase's schedule, not only the winner's (a cheap variant).
    F_lj_all = min(left_justify(inst, o.schedule.order()).F(inst) for o in res.phases)
    heur = {name: h(inst) for name, h in HEURISTICS.items()}
    warm = min([res.lj_schedule, *heur.values()], key=lambda s: s.F(inst))
    ex = exact.milp(inst, time_limit, warm_start=warm, threads=1)
    F_opt = ex.F
    add_ok, ratio_ok = bounds.check_bounds(inst, res.F_alg, F_opt) if ex.optimal else ("", "")
    return {
        "family": spec.family,
        **spec.params,
        "rep": spec.rep,
        "phases": len(res.phases),
        "W": inst.W,
        "sum_w2": inst.sum_w2,
        "D": float(bounds.additive_gap(inst)),
        "ratio_bound": float(bounds.ratio_bound(inst)),
        "F_alg": res.F_alg,
        "F_lj": res.F_lj,
        "F_lj_all": F_lj_all,
        **{f"F_{name}": s.F(inst) for name, s in heur.items()},
        "F_opt": F_opt,
        "opt_status": ex.status,
        "opt_lower_bound": ex.lower_bound,
        "ratio_phi": float(bounds.ratio(inst, res.F_alg, F_opt)),
        "ratio_phi_lj": float(bounds.ratio(inst, res.F_lj, F_opt)),
        "loss_norm": bounds.normalized_loss(inst, res.F_alg, F_opt),
        "loss_norm_lj": bounds.normalized_loss(inst, res.F_lj, F_opt),
        "additive_ok": add_ok,
        "ratio_ok": ratio_ok,
        "t_alg": res.seconds,
        "t_mip": ex.seconds,
        "t_wall": time.time(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--max-n", type=int, default=10**9)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--family", nargs="*", default=["tight", "few_phase", "uniform"])
    args = ap.parse_args()
    for family in args.family:
        # Large instances first so the slowest MIPs do not straggle at the end.
        specs = [s for s in all_specs() if s.family == family and s.n <= args.max_n]
        specs.sort(key=lambda s: -s.n)
        tasks = ((s.id, (s, args.time_limit)) for s in specs)
        run(tasks, work, args.out_dir / f"ratio_{family}.csv", args.workers)
