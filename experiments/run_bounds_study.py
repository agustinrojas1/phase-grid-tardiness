"""Round-down lower bound and local search (solve_plus) on every family in generate.py.

    uv run python experiments/run_bounds_study.py [--workers K] [--family F ...]

Writes experiments/results/bounds_<family>.csv, keyed by the same instance ids as
ratio_<family>.csv (resumable). F_opt and opt_status are copied from the ratio CSV when it
exists, so each row carries everything needed to judge the bound and the search.
"""

import argparse
import csv
import time
from pathlib import Path

from common import DEFAULT_WORKERS, RESULTS, run
from generate import Spec, all_specs

from pgt import bounds
from pgt.heuristics import atc_best
from pgt.local_search import solve_plus
from pgt.lower_bound import lower_bound_detail
from pgt.schedule import left_justify


def work(arg: tuple[Spec, dict]) -> dict:
    spec, opt = arg
    inst = spec.build()
    plus = solve_plus(inst)  # LB over the release residues
    res = plus.result
    lb_all = lower_bound_detail(inst, range(inst.p))
    F_lj_all = min(left_justify(inst, o.schedule.order()).F(inst) for o in res.phases)
    F_atc = atc_best(inst).F(inst)
    row = {
        "family": spec.family,
        **spec.params,
        "rep": spec.rep,
        "p": inst.p,
        "phases": len(res.phases),
        "W": inst.W,
        "D": float(bounds.additive_gap(inst)),
        "F_alg": res.F_alg,
        "F_lj": res.F_lj,
        "F_lj_all": F_lj_all,
        "F_ATC": F_atc,
        "F_start": plus.F_start,
        "start": plus.start,
        "F_plus": plus.F_plus,
        "LB_res": plus.lower_bound,
        "LB_all": lb_all.value,
        "LB_all_phase": lb_all.best_phase,
        "gap_certified": plus.gap,
        "gap_certified_alg": plus.gap_alg,
        "F_opt": opt.get("F_opt", ""),
        "opt_status": opt.get("opt_status", ""),
        "t_alg": res.seconds,
        "t_plus": plus.seconds,
        "t_lb_res": plus.lb.seconds,
        "t_lb_all": lb_all.seconds,
        "t_wall": time.time(),
    }
    derived = ("ratio_phi_plus", "loss_norm_plus", "lb_tight_res", "lb_tight_all", "lb_valid")
    row |= dict.fromkeys(derived, "")
    if row["F_opt"] != "":  # the ratio study closed this instance
        F_opt = int(row["F_opt"])
        row["ratio_phi_plus"] = float(bounds.ratio(inst, plus.F_plus, F_opt))
        row["loss_norm_plus"] = bounds.normalized_loss(inst, plus.F_plus, F_opt)
        row["lb_tight_res"] = plus.lower_bound == F_opt
        row["lb_tight_all"] = lb_all.value == F_opt
        row["lb_valid"] = lb_all.value <= F_opt
    return row


def optimum_by_key(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        return {
            r["key"]: {"F_opt": r["F_opt"], "opt_status": r["opt_status"]}
            for r in csv.DictReader(f)
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--max-n", type=int, default=10**9)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--family", nargs="*", default=["tight", "few_phase", "uniform"])
    args = ap.parse_args()
    for family in args.family:
        opt = optimum_by_key(args.out_dir / f"ratio_{family}.csv")
        specs = [s for s in all_specs() if s.family == family and s.n <= args.max_n]
        specs.sort(key=lambda s: -s.n)
        tasks = ((s.id, (s, opt.get(s.id, {}))) for s in specs)
        run(tasks, work, args.out_dir / f"bounds_{family}.csv", args.workers)
