"""Summarise the structure test (run src/structure_test.py first).

For each network, beta, b, A cell: difference in BM cooperation
  rewired - original    (does structure beyond degree matter?)
  shuffled - original   (do fixed partners matter?)
with a Welch 95% confidence interval over seeds.

A difference is
  "equivalent"         if the whole CI lies inside +/- MARGIN (no practical difference)
  "clearly different"  if the whole CI lies outside +/- MARGIN
  otherwise undecided (needs more seeds).

Run from project root:
    python src/structure_summary.py          (or --test for the test output)
Writes results/structure_differences.csv, results/structure_verdict.txt,
results/structure_differences_beta*.png, results/structure_coop_by_degree.png
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from networks import RESULTS_DIR

TEST = "--test" in sys.argv
NAME = "structure_test" if TEST else "structure"
MARGIN = 0.01  # cooperation differences smaller than this count as "no practical difference"
COMPARISONS = ["rewired", "shuffled"]
DEGREE_BIN_ORDER = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257+"]


def out_path(suffix):
    return os.path.join(RESULTS_DIR, NAME + suffix)


def load():
    runs = pd.read_csv(out_path("_runs.csv"))
    nodes = pd.read_csv(out_path("_nodes.csv"))
    # a crash between writing node rows and the run row means the run was redone
    runs = runs.drop_duplicates(subset="run_id", keep="last")
    nodes = nodes.drop_duplicates(subset=["run_id", "degree_bin"], keep="last")
    return runs, nodes


def welch_interval(x, y):
    """95% CI for mean(x) - mean(y), unequal variances (Welch)."""
    diff = x.mean() - y.mean()
    var_x = x.var(ddof=1) / len(x)
    var_y = y.var(ddof=1) / len(y)
    se = math.sqrt(var_x + var_y)
    if se == 0:
        return diff, diff, diff
    dof = (var_x + var_y) ** 2 / (var_x ** 2 / (len(x) - 1) + var_y ** 2 / (len(y) - 1))
    half = stats.t.ppf(0.975, dof) * se
    return diff, diff - half, diff + half


def differences(runs):
    rows = []
    for (network, beta, b, aspiration), cell in runs.groupby(["network", "beta", "b", "aspiration"]):
        original = cell[cell["condition"] == "original"]["equilibrium"]
        for condition in COMPARISONS:
            other = cell[cell["condition"] == condition]["equilibrium"]
            if len(original) < 2 or len(other) < 2:
                continue
            diff, low, high = welch_interval(other, original)
            rows.append({
                "network": network, "beta": beta, "b": b, "aspiration": aspiration,
                "comparison": f"{condition} - original",
                "original": original.mean(), "other": other.mean(),
                "diff": diff, "ci_low": low, "ci_high": high,
                "n_original": len(original), "n_other": len(other),
                "equivalent": (low > -MARGIN) and (high < MARGIN),
                "clearly_different": (low > MARGIN) or (high < -MARGIN),
            })
    return pd.DataFrame(rows)


def verdict(diff):
    lines = ["STRUCTURE TEST: does structure beyond degree change BM cooperation?",
             f"equivalent = 95% CI inside +/-{MARGIN}; clearly different = CI outside it", ""]
    for comparison in ["rewired - original", "shuffled - original"]:
        lines.append(f"=== {comparison} ===")
        part = diff[diff["comparison"] == comparison]
        for (network, beta), group in part.groupby(["network", "beta"]):
            lines.append(
                f"  {network:9s} beta={beta:<5g} cells={len(group):2d}  "
                f"mean diff={group['diff'].mean():+.4f}  "
                f"range=[{group['diff'].min():+.4f}, {group['diff'].max():+.4f}]  "
                f"equivalent={int(group['equivalent'].sum())}/{len(group)}  "
                f"clearly different={int(group['clearly_different'].sum())}/{len(group)}")
        lines.append("")

    rewired = diff[diff["comparison"] == "rewired - original"]
    n_equivalent = int(rewired["equivalent"].sum())
    n_different = int(rewired["clearly_different"].sum())
    lines.append("=== DECISION (rewired vs original) ===")
    if n_different == 0 and n_equivalent >= 0.9 * len(rewired):
        lines.append(f"A: structure beyond degree does NOT matter "
                     f"({n_equivalent}/{len(rewired)} cells equivalent, none different).")
    elif n_different == 0:
        lines.append(f"C: too noisy to call ({n_equivalent}/{len(rewired)} equivalent, "
                     f"none clearly different). Add seeds.")
    else:
        lines.append(f"B: structure matters somewhere ({n_different} cells clearly different, "
                     f"{n_equivalent}/{len(rewired)} equivalent). Cells:")
        different = rewired[rewired["clearly_different"]]
        for _, row in different.iterrows():
            lines.append(f"   {row['network']:9s} beta={row['beta']:g} b={row['b']} "
                         f"A={row['aspiration']}  diff={row['diff']:+.4f} "
                         f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
    return "\n".join(lines)


def plot_differences(diff):
    networks = sorted(diff["network"].unique())
    for beta in sorted(diff["beta"].unique()):
        fig, axes = plt.subplots(len(COMPARISONS), len(networks),
                                 figsize=(3.2 * len(networks) + 1, 6.5), squeeze=False)
        image = None
        for row, condition in enumerate(COMPARISONS):
            for col, network in enumerate(networks):
                ax = axes[row, col]
                part = diff[(diff["network"] == network) & (diff["beta"] == beta)
                            & (diff["comparison"] == f"{condition} - original")]
                grid = part.pivot(index="aspiration", columns="b", values="diff")
                grid = grid.sort_index(ascending=False)
                image = ax.imshow(grid.values, cmap="RdBu", vmin=-0.05, vmax=0.05)
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        ax.text(j, i, f"{grid.values[i, j]:+.3f}", ha="center", va="center", fontsize=6)
                ax.set_xticks(range(grid.shape[1]))
                ax.set_xticklabels([f"{x:g}" for x in grid.columns])
                ax.set_yticks(range(grid.shape[0]))
                ax.set_yticklabels([f"{y:g}" for y in grid.index])
                ax.set_xlabel("b")
                ax.set_ylabel("A")
                ax.set_title(f"{network}\n{condition} - original", fontsize=9)
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, label="difference in cooperation")
        fig.suptitle(f"BM structure test, beta = {beta:g}")
        fig.savefig(out_path(f"_differences_beta{beta:g}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_coop_by_degree(nodes):
    # weighted mean over runs: each bin's cooperation weighted by its number of nodes
    nodes = nodes.copy()
    nodes["weighted"] = nodes["coop"] * nodes["n_nodes"]
    totals = nodes.groupby(["network", "condition", "degree_bin"])[["weighted", "n_nodes"]].sum()
    totals = totals.reset_index()
    totals["coop"] = totals["weighted"] / totals["n_nodes"]
    totals.to_csv(out_path("_coop_by_degree.csv"), index=False)

    networks = sorted(totals["network"].unique())
    fig, axes = plt.subplots(1, len(networks), figsize=(3.5 * len(networks), 3.8),
                             sharey=True, squeeze=False)
    for ax, network in zip(axes[0], networks):
        for condition, style in [("original", "o-"), ("rewired", "s--")]:
            part = totals[(totals["network"] == network) & (totals["condition"] == condition)]
            part = part.set_index("degree_bin")
            labels = []
            values = []
            for label in DEGREE_BIN_ORDER:
                if label in part.index:
                    labels.append(label)
                    values.append(part.loc[label, "coop"])
            ax.plot(labels, values, style, label=condition)
        ax.set_title(network)
        ax.set_xlabel("node degree")
        ax.tick_params(axis="x", rotation=60)
    axes[0][0].set_ylabel("BM cooperation (equilibrium window)")
    axes[0][0].legend()
    fig.savefig(out_path("_coop_by_degree.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    runs, nodes = load()
    print(f"{len(runs)} runs, {len(nodes)} node rows")
    diff = differences(runs)
    diff.to_csv(out_path("_differences.csv"), index=False)
    report = verdict(diff)
    with open(out_path("_verdict.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    plot_differences(diff)
    plot_coop_by_degree(nodes)
    print("saved tables and figures in", RESULTS_DIR)


if __name__ == "__main__":
    main()
