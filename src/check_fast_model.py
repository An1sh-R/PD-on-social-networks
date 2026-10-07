"""Check that fast_model.py (numpy) does the same as model.py (Mesa reference).

Run from project root: python src/check_fast_model.py   (~5-10 min)

1. Exact: same actions -> same payoffs; same p/action/payoff -> same BM update.
2. Statistical: whole runs. Seeds give different random numbers in the two
   versions, so compare the mean cooperation over 20 seeds per cell (Welch t-test).
Output: results/fast_model_check.csv
"""
import csv
import os
import random

import numpy as np
from scipy.stats import ttest_ind

from fast_model import bm_update, neighbour_matrix, run_bm, weak_pd_payoffs
from model import CooperationModel
from networks import RESULTS_DIR, load_graph

GRAPH = "lastfm"  # the pilot graph
B = 1.5  # except the last cell
EPSILON = 0.05
STEPS = 600  # shorter than the sweep: enough to compare the two versions
WINDOW = 200
SEEDS = 20
# (b, beta, a_fraction, shuffled): low, mid, high beta, extremes, shuffled seats.
# Last cell: where the pilot saw repeat runs end at different levels (0.16 / 0.76 / 0.80).
CELLS = [
    (B, 0.1, 0.3, False),
    (B, 1.0, 0.5, False),
    (B, 10.0, 0.7, False),
    (B, 10.0, 0.7, True),
    (B, 100.0, 0.2, False),
    (B, 0.3, 0.5, True),
    (1.9, 10.0, 0.79, False),
]


def check_payoffs(G, A, degree):
    model = CooperationModel(G, "bm", b=B, seed=0)
    rng = random.Random(1)
    coop = np.zeros(len(model.players), dtype=bool)
    for i in range(len(model.players)):
        if rng.random() < 0.5:
            model.players[i].action = "C"
            coop[i] = True
        else:
            model.players[i].action = "D"
    mesa_payoffs = []
    for player in model.players:
        player.compute_payoff()
        mesa_payoffs.append(player.payoff)
    fast_payoffs = weak_pd_payoffs(A, degree, coop, B)
    assert np.allclose(mesa_payoffs, fast_payoffs), "payoffs differ"
    print("exact check: payoffs match on", len(mesa_payoffs), "players")


def check_bm_update(G):
    rng = random.Random(2)
    n_checked = 0
    for beta in [0.01, 0.3, 1.0, 10.0, 100.0]:
        model = CooperationModel(G, "bm", b=B, beta=beta, aspiration=0.7, seed=0)
        n = len(model.players)
        p = np.zeros(n)
        coop = np.zeros(n, dtype=bool)
        payoff = np.zeros(n)
        for i in range(n):
            player = model.players[i]
            player.p = rng.random()
            player.action = rng.choice(["C", "D"])
            player.payoff = rng.uniform(0, B)  # above and below the aspiration
            p[i] = player.p
            coop[i] = player.action == "C"
            payoff[i] = player.payoff
        new_p = bm_update(p, coop, payoff, beta, 0.7)
        mesa_p = []
        for player in model.players:
            player.update()
            mesa_p.append(player.p)
        assert np.allclose(mesa_p, new_p), f"BM update differs at beta={beta}"
        n_checked += n
    print("exact check: BM update matches on", n_checked, "player updates")


def compare_runs(G, A, degree):
    rows = []
    record_from = STEPS - WINDOW
    for b, beta, a_fraction, shuffled in CELLS:
        aspiration = round(a_fraction * b, 6)
        mesa_values = []
        fast_values = []
        for seed in range(SEEDS):
            model = CooperationModel(G, "bm", b=b, beta=beta, aspiration=aspiration,
                                     epsilon=EPSILON, shuffle_seats=shuffled, seed=seed)
            model.run(STEPS)
            curve = model.datacollector.model_vars["cooperation"]
            mesa_values.append(np.mean(curve[record_from:]))

            curve, p, nodes = run_bm(A, degree, b, beta, aspiration, EPSILON, STEPS,
                                     record_from=record_from, shuffle_seats=shuffled,
                                     seed=seed)
            fast_values.append(curve[record_from:].mean())

        p_value = ttest_ind(mesa_values, fast_values, equal_var=False).pvalue
        if np.isnan(p_value):
            p_value = 1.0  # both versions gave identical constant values
        row = {"b": b, "beta": beta, "a_fraction": a_fraction, "shuffled": shuffled,
               "mesa_mean": round(float(np.mean(mesa_values)), 4),
               "fast_mean": round(float(np.mean(fast_values)), 4),
               "mesa_std": round(float(np.std(mesa_values)), 4),
               "fast_std": round(float(np.std(fast_values)), 4),
               "difference": round(float(np.mean(fast_values) - np.mean(mesa_values)), 4),
               "p_value": round(float(p_value), 3),
               "agree": bool(p_value > 0.01)}
        print(row)
        rows.append(row)
    return rows


def main():
    G = load_graph(GRAPH)
    A, degree = neighbour_matrix(G)
    check_payoffs(G, A, degree)
    check_bm_update(G)

    print(f"statistical check: {GRAPH}, {STEPS} steps, {SEEDS} seeds per cell")
    rows = compare_runs(G, A, degree)

    path = os.path.join(RESULTS_DIR, "fast_model_check.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_agree = 0
    for row in rows:
        if row["agree"]:
            n_agree += 1
    print(f"{n_agree}/{len(rows)} cells agree (p > 0.01). Saved {path}")


if __name__ == "__main__":
    main()
