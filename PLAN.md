# Implementation plan: phase-grid assignment algorithm for 1 | r_j, p_j = p | Σ w_j T_j

Source paper: Chen & Hall, "Strong NP-Hardness and Approximation Algorithm for Weighted
Tardiness with Release Dates and Identical Processing Times", arXiv 2609.28751 (Sep 2026).
https://www.alphaxiv.org/abs/2609.28751

The paper states it ships no code and no computational results. This project is the first
implementation, plus an empirical study of how the algorithm behaves against the exact optimum.

---

## 1. The problem

Single machine, N jobs, no preemption. Every job j has

| symbol | meaning                         | type              |
|--------|---------------------------------|-------------------|
| r_j    | release date                    | non-negative int  |
| d_j    | due date                        | non-negative int  |
| w_j    | weight                          | positive int      |
| p      | processing time, same for all   | positive int      |

A schedule assigns start times s_j with s_j >= r_j and processing intervals [s_j, s_j + p)
pairwise disjoint. Completion C_j = s_j + p, tardiness T_j = max(C_j - d_j, 0).

Objectives:

- F(S)   = Σ_j w_j T_j                   (the real objective; strongly NP-hard, Theorem 1)
- Φ(S)   = F(S) + p·W, where W = Σ_j w_j  (the shifted objective the guarantee is stated for)

F and Φ have the same minimizers. The approximation guarantee (Theorem 2) is

    Φ_alg / Φ*  <=  3/2 - Σ_j w_j² / (2 W²)  <=  3/2 - 1/(2N)

equivalently the additive bound  F_alg <= F* + D(I),  D(I) = p/(2W) · (W² - Σ_j w_j²).

## 2. The algorithm (Section 6.2 of the paper, verbatim logic)

Let A = { r_j mod p : 1 <= j <= N } be the set of distinct release residues. For each phase a in A:

1. Round every release up to the phase-a grid:  R_j(a) = a + p · ceil((r_j - a) / p).
2. Candidate start slots:  T_a = { R_i(a) + k·p : 1 <= i <= N, 0 <= k < N }  (deduplicated).
3. Bipartite graph: jobs on one side, distinct slots of T_a on the other.
   Edge (j, t) exists iff t >= r_j, with cost w_j · max(t + p - d_j, 0).
   Solve a minimum-cost matching that covers every job (slots may stay unmatched = idle time).
4. Read the matching as a schedule (job j starts at its matched slot). Keep the cheapest schedule
   over all phases.

Facts to rely on:

- Slots on one grid are disjoint length-p intervals, so every matching is feasible (Lemma 6).
- A full matching always exists: job j has the N distinct eligible slots R_j(a) + k·p, k < N.
- For a fixed phase the matching is optimal among all schedules on that grid (Lemma 6).
- If all releases share one residue the algorithm is exact.
- Cost: at most N phases × N² slots × rectangular Hungarian O(N²·M) → O(N⁵) arithmetic ops.
- Post-pass (noted in 6.4): left-justify the final job order in continuous time, i.e.
  s_π(1) = r_π(1), s_π(k) = max(r_π(k), s_π(k-1) + p). Never worse, and it kills the loss on the
  paper's tight family. We implement it as an optional, clearly separated step so we can report
  both "algorithm as specified" and "algorithm + left-justify".

## 3. Language and technology: Python 3.14 + uv, with SciPy doing the heavy lifting

Recommendation: **Python, managed with uv, numerical core from NumPy/SciPy.**

Why:

- The inner loop is a rectangular min-cost assignment. `scipy.optimize.linear_sum_assignment`
  is a C implementation of exactly that (rectangular matrices, `np.inf` for forbidden edges).
  Writing our own Hungarian first would be the bug-prone part, and it is not what the paper
  contributes.
- We need an exact baseline to measure the ratio. `scipy.optimize.milp` (HiGHS) gives a MIP
  solver with zero extra dependencies, so one package covers both the algorithm and the oracle.
- The deliverable is as much an empirical study as a library: instance generators, ratio
  distributions, runtime curves, plots. Python is the shortest path there.
- Realistic N for this algorithm is tens of jobs (O(N⁵)); SciPy's C assignment handles N ≈ 60
  comfortably. Speed beyond that is a stretch goal, not a requirement.

Rejected alternatives:

- Rust: faster and self-contained, but we would have to write and verify the Hungarian and
  find a MIP backend. Keep as an optional stretch (Section 7) once the Python version is the
  reference oracle.
- OR-Tools CP-SAT: excellent exact solver, but wheels for Python 3.14 may lag and it is a
  heavy dependency. HiGHS via SciPy is enough for N <= 40; revisit only if the MIP is too slow.

Dependencies: `numpy`, `scipy` (>= 1.14 for `milp`), `pytest`, `hypothesis`, `matplotlib`
(experiments only), `typer` or plain `argparse` for the CLI.

## 4. Project layout

```
phase-grid-tardiness/
  PLAN.md
  README.md
  pyproject.toml                  # uv-managed, src layout
  src/pgt/
    __init__.py
    instance.py                   # Instance dataclass, validation, I/O (JSON), generators
    schedule.py                   # Schedule dataclass, feasibility check, F and Φ evaluation,
                                  #   left_justify(order)
    phase_grid.py                 # the algorithm: residues, rounding, slots, cost matrix, solve
    exact.py                      # exact oracles: brute force (N <= 9), time-indexed MIP (HiGHS)
    heuristics.py                 # dispatch rules for context: WSPT, EDD, ATC
    bounds.py                     # D(I), ratio bound, Φ helpers
    cli.py                        # `pgt solve`, `pgt bench`, `pgt verify`
  tests/
    test_schedule.py
    test_phase_grid.py
    test_exact.py
    test_paper_properties.py      # tight family, exactness on single phase, bound checks
    test_property_based.py        # hypothesis
  experiments/
    generate.py                   # instance families
    run_ratio_study.py
    run_runtime_study.py
    plots.py
    results/                      # CSV + PNG, committed
  REPORT.md                       # findings
```

## 5. Milestones

### M0. Scaffold (half a day)
- `uv init --lib`, src layout, pin Python 3.14, add deps, `pytest` config, `ruff`.
- Verify `scipy.optimize.milp` imports and solves a toy problem on this interpreter.
  If a SciPy wheel for 3.14 is missing, pin the project to 3.12 via `uv python pin`.

### M1. Data model (half a day)
- `Instance(p, r, d, w)` with NumPy int64 arrays; validate positivity and equal lengths.
- `Schedule(starts)`: `is_feasible(inst)` (releases respected, intervals disjoint), `F(inst)`,
  `Phi(inst)`, `order()` (jobs sorted by start), `left_justify(inst, order) -> Schedule`.
- JSON read/write for instances and schedules so experiments are reproducible.
- All arithmetic in integers. Costs are at most w·(t + p) with t <= max r + N·p; assert this fits
  in int64 and that the float64 conversion inside SciPy stays exact (values < 2^53).

### M2. The algorithm (one day)
`phase_grid.solve(inst, left_justify=False) -> Result`

- `residues(inst)` → sorted distinct `r mod p`.
- `round_up(r, a, p)` vectorised: `a + p * ceil((r - a) / p)`, with exact integer ceil
  (`-((a - r) // p)`), no floats.
- `slots(R, N, p)` → `np.unique(R[:, None] + p * np.arange(N)[None, :])`.
- Cost matrix C of shape (N, M): `w[:, None] * max(t[None, :] + p - d[:, None], 0)`, and
  `np.inf` where `t < r_j`. Store as float64; keep an int64 twin for exact objective evaluation.
- `linear_sum_assignment(C)` → row_ind, col_ind; starts = t[col_ind].
- Evaluate F with the integer twin, keep the best over phases; tie-break deterministically by
  smaller phase so runs are reproducible.
- Optional `left_justify=True` applies the post-pass to the winning order and records both
  values. `Result` carries: best phase, F_alg, Φ_alg, per-phase F values, schedule(s), timings.
- Optimisation that preserves correctness: drop slots that no job can use (t < min r) and
  nothing else. Resist the temptation to prune "late" slots; Lemma 6 needs all R_i + k·p, k < N.

### M3. Exact oracles (one day)
Two independent oracles so they can cross-check each other.

- `exact.brute_force(inst)`: enumerate all N! job orders, left-justify each (the paper notes an
  optimum exists among left-justified orders with starts r_i + k·p). Use for N <= 9 in tests.
- `exact.milp(inst, time_limit)`: time-indexed MIP over the candidate start set
  T = { r_i + k·p : i, 0 <= k < N } (all phases together, which is exactly why it is not an
  assignment problem). Binary x_{j,t} for t >= r_j; Σ_t x_{j,t} = 1 per job; for every slot t in
  T, Σ_j Σ_{s in T, t - p < s <= t} x_{j,s} <= 1 (no overlap for equal lengths); minimise
  Σ w_j (t + p - d_j)^+ x_{j,t}. Solve with `scipy.optimize.milp`. Return status, F*, schedule.
- Validate milp == brute_force on a few hundred random instances with N <= 8.

### M4. Tests (one day, written alongside M2 and M3)
- Feasibility: every schedule the algorithm returns passes `is_feasible`.
- Lemma 6 sanity: for a fixed phase, brute-force over grid schedules on tiny instances equals the
  assignment value.
- Exactness when all releases share one residue: F_alg == F* (brute force / MIP).
- Tight family (Proposition 3): p = N, r_i = i(2N+1), d_i = r_i + N, w_i = 1. Expect F* = 0,
  F_alg = N(N-1)/2, Φ_alg/Φ* = 3/2 - 1/(2N) exactly for N = 1..12, and F = 0 after left-justify.
- Bound checks on random instances with N <= 8 vs brute force: F_alg <= F* + D(I) and
  Φ_alg/Φ* <= 3/2 - Σw²/(2W²). A violation is a bug in our code, not the paper.
- Hypothesis property tests: random p, r, d, w with small ranges; feasibility, bound, and
  determinism (same instance → same result).
- Edge cases: N = 1; all releases 0; d_j < r_j + p (unavoidable tardiness); p = 1; duplicate
  jobs; huge weights near int64 limits.

### M5. Experiments (two days)
Instance families (parameters recorded in `experiments/generate.py`):
- Uniform: r ~ U[0, ρ·N·p], d = r + p + U[0, τ·N·p], w ~ U[1, 10]; sweep ρ (congestion) and τ
  (slack). Also integer p in {1, 5, 20, 100} to vary how many residues appear.
- Tight-family perturbations: add noise to the Proposition 3 instance to see how fast the loss
  disappears.
- Few-phase instances: releases drawn from 2 or 3 residues only (paper hints at
  "number of release phases" as a direction for stronger results).

Measurements:
- Φ_alg/Φ* and (F_alg - F*)/D(I) distributions vs exact MIP for N in {6, 10, 15, 20, 30}.
- Same with the left-justify post-pass; how often it reaches F*.
- Runtime vs N for N up to 60–80 (MIP omitted beyond what finishes in a few minutes).
- Comparison against WSPT / EDD / ATC dispatch rules to show where the guarantee buys something.
- Write everything to CSV; plots via `plots.py`; summarise in `REPORT.md`.

### M6. Polish (half a day)
- CLI: `pgt solve instance.json [--left-justify] [--exact]`, `pgt bench ...`, `pgt verify`.
- README with the algorithm summary, how to reproduce experiments, and the main plot.
- Type hints throughout, `ruff` clean, CI-less but `uv run pytest` must pass.

## 6. Definition of done
- `uv run pytest` green, including the Proposition 3 tight family and bound checks.
- Brute force and MIP oracles agree on >= 500 random small instances.
- REPORT.md contains the ratio and runtime plots and a short discussion answering:
  how far below the 3/2 bound is the algorithm in practice, and does left-justification make
  the gap vanish on random data as it does on the tight family.

## 7. Stretch goals (only after section 6)
- Rust port of `phase_grid` with an in-house rectangular Hungarian, checked against the Python
  version on thousands of instances; aim for N in the hundreds.
- Exploit structure: slots and jobs are both time-ordered, so the cost matrix has a Monge-like
  structure; investigate whether a faster assignment method applies.
- Experiment with the paper's open questions: restricted classes (few phases, agreeable
  release/due orders) where the unshifted objective might admit a multiplicative guarantee.

## 8. Risks
- SciPy/HiGHS wheels on Python 3.14: fall back to 3.12 (uv makes this a one-line change).
- `linear_sum_assignment` raises on infeasible matrices; infeasibility is impossible by the
  N-eligible-slots argument, so treat an exception as a bug and let the test suite catch it.
- MIP scale: T has up to N² starts, so N = 30 gives ~900 start times and ~27k binaries; HiGHS
  should cope, but set a time limit and record the MIP gap when it stops early.
- Float conversion inside SciPy: assert all costs < 2^53 before solving.
