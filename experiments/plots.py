"""Figures for REPORT.md from experiments/results/*.csv.

uv run python experiments/plots.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).parent / "results"

# Reference categorical palette (light), fixed order; sequential blue ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150,
})  # fmt: skip


def load(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def num(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def optimal(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["opt_status"] == "optimal"]


def group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(
        text,
        (x, y),
        xytext=(6, dy),
        textcoords="offset points",
        color=INK2,
        va="center",
        fontsize=8.5,
    )
    ax.plot([x], [y], "o", color=color, ms=4)


def fig_ratio_vs_n(uniform: list[dict]) -> None:
    """Worst and typical Phi ratio and normalized loss vs N, both variants."""
    rows = optimal(uniform)
    by_n = group(rows, "n")
    ns = sorted(int(k[0]) for k in by_n)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bound = [1.5 - 1 / (2 * n) for n in ns]
    a.plot(ns, bound, color=INK2, lw=1.2, ls=(0, (4, 3)))
    label_end(a, ns[-1], bound[-1], "Theorem 2 bound\n3/2 − 1/(2N)", INK2)
    for col, color, name in (
        ("ratio_phi", SERIES[0], "as specified"),
        ("ratio_phi_lj", SERIES[1], "+ left-justify"),
    ):
        worst = [num(by_n[(str(n),)], col).max() for n in ns]
        p99 = [np.percentile(num(by_n[(str(n),)], col), 99) for n in ns]
        a.plot(ns, worst, color=color, marker="o", ms=4, label=f"{name}: max")
        a.plot(ns, p99, color=color, lw=1.2, ls=(0, (1, 2)), label=f"{name}: 99th pct")
    a.set(
        title="Φ_alg / Φ* on random uniform instances",
        xlabel="N",
        ylabel="ratio",
        ylim=(0.99, 1.52),
        xticks=ns,
    )

    for col, color, name in (
        ("loss_norm", SERIES[0], "as specified"),
        ("loss_norm_lj", SERIES[1], "+ left-justify"),
    ):
        worst = [num(by_n[(str(n),)], col).max() for n in ns]
        mean = [num(by_n[(str(n),)], col).mean() for n in ns]
        b.plot(ns, worst, color=color, marker="o", ms=4, label=f"{name}: max")
        b.plot(ns, mean, color=color, lw=1.2, ls=(0, (1, 2)), label=f"{name}: mean")
    b.set(
        title="Loss as a share of the additive bound",
        xlabel="N",
        ylabel="(F_alg − F*) / D(I)",
        ylim=(0, None),
        xticks=ns,
    )
    handles, _ = a.get_legend_handles_labels()
    labels = [
        "as specified: max",
        "as specified: 99th pct / mean",
        "+ left-justify: max",
        "+ left-justify: 99th pct / mean",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS / "ratio_vs_n.png")
    plt.close(fig)


def fig_heatmap(uniform: list[dict]) -> None:
    """Mean normalized loss over (rho, tau), one panel per p, at the largest N."""
    n = max(int(r["n"]) for r in uniform)
    rows = [r for r in optimal(uniform) if int(r["n"]) == n]
    ps = sorted({int(r["p"]) for r in rows})
    rhos = sorted({float(r["rho"]) for r in rows})
    taus = sorted({float(r["tau"]) for r in rows})
    fig, axes = plt.subplots(
        1, len(ps), figsize=(2.6 * len(ps) + 0.8, 3.1), squeeze=False, sharey=True
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = max(0.05, max((float(r["loss_norm"]) for r in rows), default=0))
    for ax, p in zip(axes[0], ps, strict=True):
        M = np.zeros((len(taus), len(rhos)))
        for i, tau in enumerate(taus):
            for j, rho in enumerate(rhos):
                cell = [
                    float(r["loss_norm"])
                    for r in rows
                    if int(r["p"]) == p and float(r["rho"]) == rho and float(r["tau"]) == tau
                ]
                M[i, j] = np.mean(cell) if cell else np.nan
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="auto")
        for i in range(len(taus)):
            for j in range(len(rhos)):
                ink = SURFACE if M[i, j] > 0.55 * vmax else INK
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8, color=ink)
        ax.set_xticks(range(len(rhos)), [f"{x:g}" for x in rhos])
        ax.set_yticks(range(len(taus)), [f"{x:g}" for x in taus])
        ax.set(title=f"p = {p}", xlabel="ρ (release spread)")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].set_ylabel("τ (due-date slack)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="mean (F_alg − F*) / D")
    fig.suptitle(
        f"Where the loss lives: uniform family, N = {n}",
        x=0.01,
        y=1.06,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.savefig(RESULTS / "loss_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_tight(tight: list[dict]) -> None:
    """Normalized loss as each perturbation knob of Proposition 3 is turned."""
    rows = optimal(tight)
    ns = sorted({int(r["n"]) for r in rows})
    n = ns[-1]
    rows = [r for r in rows if int(r["n"]) == n]
    base = [
        r for r in rows if float(r["jitter"]) == 0 and float(r["slack"]) == 0 and int(r["ws"]) == 0
    ]
    base_loss = np.mean([float(r["loss_norm"]) for r in base])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    knobs = (
        ("jitter", "release jitter (× N)"),
        ("slack", "extra due-date slack (× N)"),
        ("ws", "weight spread"),
    )
    for ax, (knob, xlabel) in zip(axes, knobs, strict=True):
        others = [k for k, _ in knobs if k != knob]
        sel = [r for r in rows if all(float(r[o]) == 0 for o in others)]
        levels = sorted({float(r[knob]) for r in sel})
        for col, color, name in (
            ("loss_norm", SERIES[0], "as specified"),
            ("loss_norm_lj", SERIES[1], "+ left-justify"),
        ):
            mean = [np.mean([float(r[col]) for r in sel if float(r[knob]) == v]) for v in levels]
            ax.plot(levels, mean, color=color, marker="o", ms=4, label=name)
        ax.set(xlabel=xlabel, ylim=(-0.03, 1.05))
    axes[0].set_ylabel("mean (F_alg − F*) / D")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle(
        f"Perturbing the tight family (N = {n}; unperturbed loss = {base_loss:.2f})",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "tight_perturbation.png")
    plt.close(fig)


def fig_few_phase(few: list[dict]) -> None:
    rows = optimal(few)
    ns = sorted({int(r["n"]) for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for color, n in zip(SERIES, ns, strict=False):
        sel = [r for r in rows if int(r["n"]) == n]
        ks = sorted({int(r["k"]) for r in sel})
        mean = [np.mean([float(r["loss_norm"]) for r in sel if int(r["k"]) == k]) for k in ks]
        ax.plot(ks, mean, color=color, marker="o", ms=4)
        label_end(ax, ks[-1], mean[-1], f"N = {n}", color)
    ax.set(
        title="Loss vs number of release phases (p = 20)",
        xlabel="distinct release residues k",
        ylabel="mean (F_alg − F*) / D",
        xscale="log",
    )
    ax.set_xticks([1, 2, 3, 5, 10, 20], ["1", "2", "3", "5", "10", "20"])
    ax.set_xlim(0.9, 30)
    fig.tight_layout()
    fig.savefig(RESULTS / "few_phase.png")
    plt.close(fig)


def fig_heuristics(rows: list[dict]) -> None:
    """Share of instances solved optimally, and mean excess of Phi, per method."""
    rows = optimal(rows)
    methods = [
        ("F_alg", "phase-grid"),
        ("F_lj", "phase-grid\n+ left-justify"),
        ("F_ATC", "ATC (best k)"),
        ("F_EDD", "EDD"),
        ("F_WSPT", "WSPT"),
    ]
    F_opt = num(rows, "F_opt")
    # the tight family has p = N and no p column
    pW = num(rows, "W") * np.array([float(r.get("p") or r["n"]) for r in rows])
    hit = [100 * np.mean(num(rows, c) == F_opt) for c, _ in methods]
    excess = [100 * np.mean((num(rows, c) - F_opt) / (F_opt + pW)) for c, _ in methods]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    y = np.arange(len(methods))[::-1]
    for ax, vals, title, fmt in (
        (a, hit, "Instances solved to optimality", "{:.0f}%"),
        (b, excess, "Mean Φ excess over optimum", "{:.2f}%"),
    ):
        ax.barh(y, vals, height=0.62, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals, strict=True):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [m for _, m in methods])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(vals) * 1.22)
    a.set_xlabel("% of instances")
    b.set_xlabel("mean of (F − F*) / (F* + pW), %")
    fig.suptitle(
        f"All families, {len(rows)} instances with a proven optimum",
        x=0.01,
        ha="left",
        fontsize=9.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "heuristics.png")
    plt.close(fig)


def fig_runtime(alg: list[dict], mip: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    series = [
        ("0.25", "algorithm, ρ = 0.25"),
        ("1.0", "algorithm, ρ = 1"),
        ("4.0", "algorithm, ρ = 4"),
        ("spread", "algorithm, N² slots per phase"),
    ]
    for color, (rho, name) in zip(SERIES, series, strict=False):
        sel = group([r for r in alg if r["rho"] == rho], "n")
        ns = sorted(int(k[0]) for k in sel)
        if ns:
            med = [np.median(num(sel[(str(n),)], "t_alg")) for n in ns]
            ax.plot(ns, med, color=color, marker="o", ms=4, label=name)
    if mip:
        sel = group(mip, "n")
        ns = sorted(int(k[0]) for k in sel)
        med = [np.median(num(sel[(str(n),)], "t_mip")) for n in ns]
        ax.plot(ns, med, color=SERIES[4], marker="s", ms=4, label="exact MIP (HiGHS, 1 thread)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="N (jobs)",
        ylabel="median seconds",
        title="Runtime vs N (p = 1000 for the algorithm, p = 20 for the MIP)",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "runtime.png")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        RESULTS = Path(sys.argv[1])
    uniform, tight, few = (
        load("ratio_uniform.csv"),
        load("ratio_tight.csv"),
        load("ratio_few_phase.csv"),
    )
    if uniform:
        fig_ratio_vs_n(uniform)
        fig_heatmap(uniform)
    if tight:
        fig_tight(tight)
    if few:
        fig_few_phase(few)
    if uniform or tight or few:
        fig_heuristics(uniform + tight + few)
    alg, mip = load("runtime_alg.csv"), load("runtime_mip.csv")
    if alg or mip:
        fig_runtime(alg, mip)
    print("figures written to", RESULTS)
