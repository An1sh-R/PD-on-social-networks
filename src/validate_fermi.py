"""Check Fermi + payoff schemes against a known result before the sweep.

Ichinose & Sayama (arXiv 1412.2311, Artificial Life 2017): weak PD on randomised
Barabasi-Albert graphs (N = 5000, mean degree 4-16). Found:
  (a) accumulated payoffs give much more cooperation than average payoffs
  (b) accumulated: cooperation falls as mean degree grows
  (c) average: cooperation rises as mean degree grows
  (d) the gap between the two shrinks as mean degree grows
They used linear imitation, one player at a time, start all-D, mutation 0.005.
We use our sweep's set-up instead (Fermi, synchronous, 50% C start, mutation 0.01,
accumulated payoffs with kappa * <k>, see run.py), so only the direction of (a)-(d)
can be compared, not the numbers. Criteria were written before running.

Run from project root: python src/validate_fermi.py   (~5 min)
Output: results/validate_fermi.csv, results/validate_fermi.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from fast_model import neighbour_matrix, run_fermi
from networks import RESULTS_DIR, reconnect_pieces

N = 5000
MEAN_DEGREES = [4, 8, 16]
TEMPTATIONS = [1.1, 1.2, 1.5, 2.0]
KAPPA = 0.1  # for average payoff; accumulated uses KAPPA * <k> (same as run.py)
MUTATION = 0.01
STEPS = 3000
WINDOW = 1000
SEEDS = 10


def randomised_ba(mean_degree, seed):
    """BA graph, then degree-preserving swaps (as Ichinose & Sayama), kept connected."""
    G = nx.barabasi_albert_graph(N, mean_degree // 2, seed=seed)
    nswap = 10 * G.number_of_edges()
    nx.double_edge_swap(G, nswap=nswap, max_tries=100 * nswap, seed=seed)
    reconnect_pieces(G, np.random.default_rng(seed))
    assert nx.is_connected(G)
    return G


def run_all():
    rows = []
    for mean_degree in MEAN_DEGREES:
        G = randomised_ba(mean_degree, seed=1)
        A, degree = neighbour_matrix(G)
        actual_mean_degree = degree.mean()
        print(f"mean degree {mean_degree}: actual {actual_mean_degree:.2f}, max {int(degree.max())}")
        for b in TEMPTATIONS:
            for scheme in ["average", "accumulated"]:
                kappa = KAPPA
                if scheme == "accumulated":
                    kappa = KAPPA * actual_mean_degree
                for seed in range(SEEDS):
                    curve, _ = run_fermi(A, degree, b, kappa, STEPS, payoff_scheme=scheme,
                                         mutation=MUTATION, seed=seed)
                    rows.append({"mean_degree": mean_degree, "b": b, "payoff_scheme": scheme,
                                 "seed": seed, "cooperation": curve[-WINDOW:].mean(),
                                 "drift": curve[-WINDOW // 2:].mean() - curve[-WINDOW:-WINDOW // 2].mean()})
    return pd.DataFrame(rows)


def verdicts(means):
    """The four claims, checked at b = 1.2 (Ichinose & Sayama's main value) and all b."""
    def coop(k, b, scheme):
        return means[(means["mean_degree"] == k) & (means["b"] == b)
                     & (means["payoff_scheme"] == scheme)]["cooperation"].iloc[0]

    lines = []
    gaps_ok = 0
    n_cells = 0
    for k in MEAN_DEGREES:
        for b in TEMPTATIONS:
            n_cells += 1
            if coop(k, b, "accumulated") > coop(k, b, "average"):
                gaps_ok += 1
    lines.append(f"(a) accumulated > average: {gaps_ok}/{n_cells} cells")
    lines.append(f"    at mean degree 4, b=1.2: accumulated {coop(4, 1.2, 'accumulated'):.3f} "
                 f"vs average {coop(4, 1.2, 'average'):.3f}")
    low = coop(4, 1.2, "accumulated")
    high = coop(16, 1.2, "accumulated")
    lines.append(f"(b) accumulated falls with mean degree (b=1.2): {low:.3f} -> {high:.3f}  "
                 f"{'MATCHES' if high < low else 'DOES NOT MATCH'}")
    low = coop(4, 1.2, "average")
    high = coop(16, 1.2, "average")
    lines.append(f"(c) average rises with mean degree (b=1.2): {low:.3f} -> {high:.3f}  "
                 f"{'MATCHES' if high > low else 'DOES NOT MATCH'}")
    gap_low = coop(4, 1.2, "accumulated") - coop(4, 1.2, "average")
    gap_high = coop(16, 1.2, "accumulated") - coop(16, 1.2, "average")
    lines.append(f"(d) gap shrinks with mean degree (b=1.2): {gap_low:.3f} -> {gap_high:.3f}  "
                 f"{'MATCHES' if gap_high < gap_low else 'DOES NOT MATCH'}")
    return lines


def plot(means):
    fig, axes = plt.subplots(1, len(MEAN_DEGREES), figsize=(4.5 * len(MEAN_DEGREES), 4), sharey=True)
    for ax, k in zip(axes, MEAN_DEGREES):
        for scheme, style in [("accumulated", "-o"), ("average", "--s")]:
            part = means[(means["mean_degree"] == k) & (means["payoff_scheme"] == scheme)]
            ax.errorbar(part["b"], part["cooperation"], yerr=part["ci"], fmt=style, label=scheme, capsize=3)
        ax.set_title(f"randomised BA, mean degree {k}")
        ax.set_xlabel("b")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("cooperation (last 1000 of 3000 steps)")
    axes[0].legend()
    fig.suptitle(f"Fermi, kappa={KAPPA} (accumulated: kappa*<k>), mutation={MUTATION}, {SEEDS} seeds, 95% CI")
    fig.savefig(os.path.join(RESULTS_DIR, "validate_fermi.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    runs = run_all()
    keys = ["mean_degree", "b", "payoff_scheme"]
    means = runs.groupby(keys, as_index=False).agg(cooperation=("cooperation", "mean"),
                                                   std=("cooperation", "std"),
                                                   max_abs_drift=("drift", lambda x: x.abs().max()))
    means["ci"] = 2.262 * means["std"] / np.sqrt(SEEDS)  # t(9) 95%
    means.to_csv(os.path.join(RESULTS_DIR, "validate_fermi.csv"), index=False)
    print(means.round(3).to_string(index=False))
    print()
    for line in verdicts(means):
        print(line)
    plot(means)


if __name__ == "__main__":
    main()
