"""Instance families for the ratio study. Every instance is identified by a string id and
built deterministically from it, so a CSV row is enough to reproduce its instance:

    uv run python experiments/generate.py --dump DIR      # write every instance as JSON
    uv run python experiments/generate.py --show ID       # print one instance
"""

import argparse
import itertools
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pgt import instance as gen
from pgt.instance import Instance

NS = (6, 10, 15, 20, 30)
REPS = 20


@dataclass(frozen=True)
class Spec:
    family: str
    params: dict = field(hash=False)
    rep: int

    @property
    def id(self) -> str:
        body = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.family}({body})#{self.rep}"

    @property
    def n(self) -> int:
        return self.params["n"]

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(zlib.crc32(self.id.encode()))

    def build(self) -> Instance:
        rng, q = self.rng(), self.params
        match self.family:
            case "uniform":
                inst = gen.uniform(q["n"], q["p"], q["rho"], q["tau"], rng)
            case "tight":
                inst = gen.perturbed_tight(
                    q["n"], rng, jitter=q["jitter"], slack=q["slack"], weight_spread=q["ws"]
                )
            case "few_phase":
                inst = gen.few_phase(q["n"], q["p"], q["k"], q["rho"], q["tau"], rng)
            case _:
                raise ValueError(self.family)
        return Instance(inst.p, inst.r, inst.d, inst.w, name=self.id)


def uniform_specs(ns=NS, reps=REPS) -> list[Spec]:
    """Congestion rho x slack tau x processing time p (p controls how many residues appear)."""
    grid = itertools.product(ns, (1, 5, 20, 100), (0.25, 0.5, 1.0, 2.0), (0.0, 0.25, 1.0))
    return [
        Spec("uniform", dict(n=n, p=p, rho=rho, tau=tau), rep)
        for n, p, rho, tau in grid
        for rep in range(reps)
    ]


def tight_specs(ns=(6, 10, 20, 30), reps=REPS) -> list[Spec]:
    """Proposition 3 with one noise knob turned at a time (and the unperturbed instance)."""
    knobs = [dict(jitter=0.0, slack=0.0, ws=0)]
    knobs += [dict(jitter=j, slack=0.0, ws=0) for j in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
    knobs += [dict(jitter=0.0, slack=s, ws=0) for s in (0.1, 0.25, 0.5, 1.0)]
    knobs += [dict(jitter=0.0, slack=0.0, ws=w) for w in (1, 4, 9)]
    return [
        Spec("tight", dict(n=n, **k), rep)
        for n in ns
        for k in knobs
        for rep in range(1 if k == knobs[0] else reps)
    ]


def few_phase_specs(ns=(10, 20, 30), reps=REPS) -> list[Spec]:
    """Releases on k residues only, p = 20 (k = 20 is close to `uniform`)."""
    grid = itertools.product(ns, (1, 2, 3, 5, 10, 20), (0.25, 1.0), (0.0, 0.25))
    return [
        Spec("few_phase", dict(n=n, p=20, k=k, rho=rho, tau=tau), rep)
        for n, k, rho, tau in grid
        for rep in range(reps)
    ]


def all_specs() -> list[Spec]:
    return uniform_specs() + tight_specs() + few_phase_specs()


def by_id() -> dict[str, Spec]:
    return {s.id: s for s in all_specs()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, help="write every instance to DIR/<n>.json")
    ap.add_argument("--show", help="print the instance with this id")
    args = ap.parse_args()
    specs = all_specs()
    if args.show:
        print(json.dumps(by_id()[args.show].build().to_dict()))
    elif args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        index = {}
        for i, s in enumerate(specs):
            s.build().save(args.dump / f"{i:05d}.json")
            index[f"{i:05d}"] = s.id
        (args.dump / "index.json").write_text(json.dumps(index, indent=1))
        print(f"wrote {len(specs)} instances to {args.dump}")
    else:
        from collections import Counter

        print(Counter(s.family for s in specs), len(specs))
