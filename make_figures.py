"""Figures for the README (called by train.py; can also be re-run on its own)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK2, MUTED, SURFACE = "#b9b8b3", "#0b0b0b", "#52514e", "#898781", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e8e7e2", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlelocation": "left",
})


def _smooth(x, w=100):
    k = np.ones(w) / w
    return np.array([np.convolve(r, k, mode="valid") for r in np.atleast_2d(x)])


def learning_curves(root, res):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    opt = -50 * res["baselines"]["Optimal (value iteration)"]["mean"]
    for algo, color, label in (("q_learning", BLUE, "Q-learning"), ("monte_carlo", ORANGE, "Monte Carlo")):
        curves = _smooth(np.load(root / "results" / f"curves_{algo}.npy"))
        x = np.arange(curves.shape[1]) + 100
        ax.fill_between(x, curves.min(0), curves.max(0), color=color, alpha=0.15, linewidth=0)
        ax.plot(x, curves.mean(0), color=color, lw=2)
        ax.annotate(label, (x[-1], curves.mean(0)[-1]), xytext=(6, 0 if algo == "q_learning" else -12),
                    textcoords="offset points", color=INK2, va="center", fontsize=10)
    ax.axhline(opt, color=AQUA, lw=2, ls=(0, (5, 3)))
    ax.annotate("Optimal policy (expected)", (0.99, opt), xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points", ha="right", color=INK2, fontsize=10)
    ax.set_title("Learning curves: episode return (100-episode moving average)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Return per 50-step episode")
    ax.set_xlim(0, curves.shape[1] + 100 + 900)
    ax.text(0, -0.2, "Line: mean of seeds · band: min–max across seeds. Return = −(cars waiting), summed over the episode.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "docs" / "figures" / "learning_curves.png", dpi=160)
    plt.close(fig)


def evaluation(root, res):
    rows = [(k, v["mean"], v["ci95"]) for k, v in res["baselines"].items() if "overflow" not in k]
    for algo, label in (("q_learning", "Q-learning"), ("monte_carlo", "Monte Carlo")):
        e = res["experiments"]["baseline"][algo]
        rows.append((label, e["mean_waiting"], e["std_across_seeds"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    color = {"Q-learning": BLUE, "Monte Carlo": ORANGE, "Optimal (value iteration)": AQUA}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.62, color=[color.get(r[0], GREY) for r in rows],
            edgecolor=SURFACE, linewidth=2)
    ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for yi, (_, m, e) in zip(y, rows):
        ax.text(m + e + 0.1, yi, f"{m:.2f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [r[0] for r in rows], color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average cars waiting per step  (lower is better)")
    ax.set_title("Evaluation on 2,000 identical held-out traffic episodes")
    ax.text(0, -0.2, "Whiskers: 95% CI for fixed policies; ±1 sd across 5 training seeds for the learners.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.12)
    fig.tight_layout()
    fig.savefig(root / "docs" / "figures" / "evaluation.png", dpi=160)
    plt.close(fig)


def policies(root):
    names = (("q_learning", "Q-learning"), ("monte_carlo", "Monte Carlo"), ("optimal", "Optimal (value iteration)"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    cmap = matplotlib.colors.ListedColormap(["#d6e6f8", YELLOW])
    pi_star = np.array(json.loads((root / "models" / "optimal.json").read_text())["Q"]).argmax(2)
    for ax, (f, title) in zip(axes, names):
        pi = np.array(json.loads((root / "models" / f"{f}.json").read_text())["Q"]).argmax(2)
        ax.imshow(pi, cmap=cmap, vmin=0, vmax=1, origin="lower")
        for ns in range(pi.shape[0]):
            for ew in range(pi.shape[1]):
                ax.text(ew, ns, "NS" if pi[ns, ew] else "EW", ha="center", va="center", fontsize=9.5,
                        color=INK, fontweight="bold" if pi[ns, ew] != pi_star[ns, ew] else "normal")
                if pi[ns, ew] != pi_star[ns, ew]:
                    ax.add_patch(plt.Rectangle((ew - .46, ns - .46), .92, .92, fill=False, ec=ORANGE, lw=2))
        ax.set_xticks(np.arange(pi.shape[1]) + .5, minor=True)
        ax.set_yticks(np.arange(pi.shape[0]) + .5, minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(pi.shape[1]))
        ax.set_yticks(range(pi.shape[0]))
        agree = (pi == pi_star).mean()
        ax.set_title(f"{title} (saved model)\n{agree:.0%} of states match optimal" if f != "optimal" else f"{title}\nground truth",
                     fontsize=11)
        ax.set_xlabel("Cars waiting East-West")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Cars waiting North-South")
    fig.text(0.01, 0.01, "Cell = greedy action in that state: EW = East-West green, NS = North-South green. "
             "Outlined = differs from the optimal policy.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(root / "docs" / "figures" / "policies.png", dpi=160)
    plt.close(fig)


def make_all(root: Path):
    root = Path(root)
    (root / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    res = json.loads((root / "results" / "results.json").read_text())
    learning_curves(root, res)
    evaluation(root, res)
    policies(root)
    web = root / "public" / "static" / "figures"          # same images for the website
    web.mkdir(parents=True, exist_ok=True)
    for f in (root / "docs" / "figures").glob("*.png"):
        shutil.copy(f, web / f.name)
    print("figures written to docs/figures/ and public/static/figures/")


if __name__ == "__main__":
    make_all(Path(__file__).resolve().parent)
