"""Check our BM agent against Ezaki & Masuda (2017), PLOS ONE 12(12): e0189220.

Their setup: ring of N=100 players, each with k neighbours. Donation game
(c=1, b=2 or 6), payoff averaged over neighbours. BM with beta=0.2, A=1.0,
eps=0.05, starting p=0.8. 50 rounds, cooperation averaged over rounds 1-25
and 1000 runs.

Their result we should reproduce:
- k=2, b/c=6: static network -> more cooperation than shuffled network
- k=2, b/c=2: no such difference
- the static-minus-shuffled gap shrinks as k grows

"Shuffled" = same ring, but players get new random positions after every round.

Run from project root: python src/validate_bm.py
Writes results/validate_bm.csv and results/validate_bm.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from model import CooperationModel
from networks import RESULTS_DIR

N = 100
ROUNDS = 50
AVERAGE_OVER = 25  # first 25 rounds, as in the paper
RUNS = 1000
BM_PARAMS = {"beta": 0.2, "aspiration": 1.0, "epsilon": 0.05, "initial_coop": 0.8}


def ring(k):
    # each node linked to k/2 nodes on each side (Watts-Strogatz with no rewiring)
    return nx.watts_strogatz_graph(N, k, 0.0)


def run_once(G, b, shuffled, seed):
    model = CooperationModel(G, "bm", game="donation", b=b, c=1.0, shuffle_seats=shuffled,
                             seed=seed, **BM_PARAMS)
    model.run(ROUNDS)
    return model.datacollector.model_vars["cooperation"]  # one value per round


def mean_curve(G, b, shuffled):
    """Cooperation in each round, averaged over RUNS runs."""
    totals = [0.0] * ROUNDS
    for seed in range(RUNS):
        curve = run_once(G, b, shuffled, seed)
        for t in range(ROUNDS):
            totals[t] += curve[t]
    return [x / RUNS for x in totals]


def main():
    rows = []
    curves_k2 = {}
    for k in [2, 4, 6, 8]:
        G = ring(k)
        for b in [2.0, 6.0]:
            for shuffled in [False, True]:
                curve = mean_curve(G, b, shuffled)
                coop = sum(curve[:AVERAGE_OVER]) / AVERAGE_OVER
                treatment = "shuffled" if shuffled else "static"
                rows.append({"k": k, "b_over_c": b, "treatment": treatment, "cooperation": coop})
                if k == 2:
                    curves_k2[(b, treatment)] = curve
                print(f"k={k} b/c={b:.0f} {treatment:8s} cooperation={coop:.3f}")

    table = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "validate_bm.csv"), index=False)

    # gap = static - shuffled. Paper: positive for large b/c and small k only.
    print("\nstatic minus shuffled (network reciprocity):")
    for k in [2, 4, 6, 8]:
        for b in [2.0, 6.0]:
            part = table[(table["k"] == k) & (table["b_over_c"] == b)]
            static = part[part["treatment"] == "static"]["cooperation"].iloc[0]
            shuffled = part[part["treatment"] == "shuffled"]["cooperation"].iloc[0]
            print(f"  k={k} b/c={b:.0f}: {static - shuffled:+.3f}")

    # plot like the paper's Fig 2: cooperation per round, k=2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, b in zip(axes, [6.0, 2.0]):
        for treatment, color in [("static", "tab:blue"), ("shuffled", "tab:red")]:
            ax.plot(range(1, ROUNDS + 1), curves_k2[(b, treatment)], color=color, label=treatment)
        ax.set_title(f"b/c = {b:.0f}, k = 2")
        ax.set_xlabel("round")
    axes[0].set_ylabel("fraction cooperating")
    axes[0].legend()
    fig.suptitle(f"BM check vs Ezaki & Masuda 2017 Fig 2 ({RUNS} runs)")
    fig.savefig(os.path.join(RESULTS_DIR, "validate_bm.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
