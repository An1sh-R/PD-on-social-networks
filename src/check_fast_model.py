"""Check that fast_model.py (numpy) does the same as model.py (Mesa reference).

Run from project root: python src/check_fast_model.py   (~35 min)

1. Exact: same actions -> same payoffs (average and accumulated); same
   p/action/payoff -> same BM update (both schemes); same chosen neighbour ->
   same Fermi copy probability; random neighbours are real, uniform neighbours.
   Mean-degree normalisation (run.py): on a regular graph, accumulated payoffs with
   beta / k (BM) or kappa * k (Fermi) give exactly the averaged-payoff run.
2. Statistical: whole runs (BM average, BM accumulated, Fermi both schemes, with
   and without mutation).
   Seeds give different random numbers in the two versions, so compare the mean
   cooperation over 20 seeds per cell (Welch t-test).
Output: results/fast_model_check.csv
"""
import csv
import os
import random

import networkx as nx
import numpy as np
from scipy.special import expit
from scipy.stats import chisquare, ttest_ind

from fast_model import (assortment, bm_update, fermi_copy_probability, neighbour_matrix,
                        random_neighbours, run_bm, run_fermi, weak_pd_payoffs)
from model import CooperationModel
from networks import RESULTS_DIR, load_graph

GRAPH = "lastfm"  # the pilot graph
B = 1.5  # except where a cell says otherwise
EPSILON = 0.05
STEPS = 600  # shorter than the sweep: enough to compare the two versions
WINDOW = 200
SEEDS = 20
# BM (b, beta, a_fraction, shuffled, payoff_scheme): low, mid, high beta, extremes,
# shuffled seats. (1.9, 10, 0.79): where the pilot saw repeats end at 0.16 / 0.76 / 0.80.
BM_CELLS = [
    (B, 0.1, 0.3, False, "average"),
    (B, 1.0, 0.5, False, "average"),
    (B, 10.0, 0.7, False, "average"),
    (B, 10.0, 0.7, True, "average"),
    (B, 100.0, 0.2, False, "average"),
    (B, 0.3, 0.5, True, "average"),
    (1.9, 10.0, 0.79, False, "average"),
    (B, 0.1, 0.3, False, "accumulated"),
    (B, 1.0, 0.3, False, "accumulated"),
    (B, 1.0, 0.3, True, "accumulated"),
]
# Fermi (b, kappa, shuffled, payoff_scheme). Chosen where cooperation is not stuck at 0
# (Fermi with average payoff dies out at almost every kappa on LastFM; with shuffled
# seats it always dies out, so that cell is about how fast: kappa=10 decays slowest).
# (b, kappa, shuffled, payoff_scheme, mutation)
FERMI_CELLS = [
    (1.05, 0.01, False, "average", 0.0),
    (1.2, 0.1, False, "accumulated", 0.0),
    (1.5, 1.0, False, "accumulated", 0.0),
    (1.05, 10.0, True, "average", 0.0),
    (1.05, 0.1, False, "average", 0.01),
    (1.5, 0.1, False, "accumulated", 0.01),
]


def check_payoffs(G, A, degree, payoff_scheme):
    model = CooperationModel(G, "bm", b=B, payoff_scheme=payoff_scheme, seed=0)
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
    fast_payoffs = weak_pd_payoffs(A, degree, coop, B, payoff_scheme)
    assert np.allclose(mesa_payoffs, fast_payoffs), f"{payoff_scheme} payoffs differ"
    print(f"exact check: {payoff_scheme} payoffs match on", len(mesa_payoffs), "players")


def check_bm_update(G, degree, payoff_scheme):
    rng = random.Random(2)
    n_checked = 0
    if payoff_scheme == "accumulated":
        aspiration = 0.7 * degree  # aspiration of each seat (see run_bm)
    else:
        aspiration = 0.7
    for beta in [0.01, 0.3, 1.0, 10.0, 100.0]:
        model = CooperationModel(G, "bm", b=B, beta=beta, aspiration=0.7,
                                 payoff_scheme=payoff_scheme, seed=0)
        n = len(model.players)
        p = np.zeros(n)
        coop = np.zeros(n, dtype=bool)
        payoff = np.zeros(n)
        for i in range(n):
            player = model.players[i]
            player.p = rng.random()
            player.action = rng.choice(["C", "D"])
            # above and below the aspiration (accumulated: scaled by degree)
            if payoff_scheme == "accumulated":
                player.payoff = rng.uniform(0, B) * len(player.neighbors)
            else:
                player.payoff = rng.uniform(0, B)
            p[i] = player.p
            coop[i] = player.action == "C"
            payoff[i] = player.payoff
        new_p = bm_update(p, coop, payoff, beta, aspiration)
        mesa_p = []
        for player in model.players:
            player.update()
            mesa_p.append(player.p)
        assert np.allclose(mesa_p, new_p), f"{payoff_scheme} BM update differs at beta={beta}"
        n_checked += n
    print(f"exact check: {payoff_scheme} BM update matches on", n_checked, "player updates")


def check_fermi_update(G, A, degree):
    """Same payoffs + same chosen neighbour -> same copy probability as FermiAgent.update.
    (Mesa draws its own neighbour, so we use its formula on our chosen one.)"""
    rng = np.random.default_rng(3)
    n = A.shape[0]
    payoff = rng.uniform(0, B * 20, n)
    other = random_neighbours(A, degree, rng)
    for kappa in [0.01, 0.1, 1.0]:
        model = CooperationModel(G, "fermi", kappa=kappa, seed=0)
        fast_probability = fermi_copy_probability(payoff, other, kappa)
        for i in range(n):
            mesa_probability = expit((payoff[other[i]] - payoff[i]) / model.kappa)
            assert np.isclose(mesa_probability, fast_probability[i]), "copy probability differs"
    print("exact check: Fermi copy probability matches on", 3 * n, "players")


def check_random_neighbours(A, degree):
    """Chosen neighbour is always a real neighbour, and each neighbour equally likely."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        other = random_neighbours(A, degree, rng)
        for i in range(A.shape[0]):
            assert A[i, other[i]] == 1, "chose a node that is not a neighbour"
    # the highest-degree node: count how often each of its neighbours is picked
    hub = int(np.argmax(degree))
    counts = {}
    for _ in range(20000):
        picked = int(random_neighbours(A, degree, rng)[hub])
        counts[picked] = counts.get(picked, 0) + 1
    observed = []
    for neighbour in A.indices[A.indptr[hub]:A.indptr[hub + 1]]:
        observed.append(counts.get(int(neighbour), 0))
    p_value = chisquare(observed).pvalue
    assert p_value > 0.001, f"neighbour choice not uniform (p={p_value:.4f})"
    print(f"exact check: random neighbours are neighbours; uniform at hub "
          f"(degree {int(degree[hub])}, chi-square p={p_value:.2f})")


def check_normalisation_on_regular_graph():
    """run.py divides beta by <k> (BM) and multiplies kappa by <k> (Fermi) for
    accumulated payoffs. On a regular graph every k = <k>, so that must give exactly
    the averaged-payoff run (same seed -> same random numbers -> same curve)."""
    k = 6
    G = nx.random_regular_graph(k, 500, seed=1)
    A, degree = neighbour_matrix(G)
    for beta in [0.1, 1.0, 10.0]:
        average, _, _, _ = run_bm(A, degree, B, beta, 0.6, EPSILON, 300, seed=5)
        accumulated, _, _, _ = run_bm(A, degree, B, beta / k, 0.6, EPSILON, 300,
                                   payoff_scheme="accumulated", seed=5)
        assert np.array_equal(average, accumulated), f"BM normalisation off at beta={beta}"
    for kappa in [0.01, 0.1, 1.0]:
        average, _, _ = run_fermi(A, degree, 1.05, kappa, 300, mutation=0.01, seed=5)
        accumulated, _, _ = run_fermi(A, degree, 1.05, kappa * k, 300, payoff_scheme="accumulated",
                                   mutation=0.01, seed=5)
        assert np.array_equal(average, accumulated), f"Fermi normalisation off at kappa={kappa}"
    print("exact check: mean-degree normalisation gives the averaged run on a regular graph")


def check_assortment(G, A, degree):
    """assortment() against a plain loop over every edge, both directions."""
    rng = np.random.default_rng(6)
    for share_coop in [0.1, 0.5, 0.9]:
        coop = rng.random(A.shape[0]) < share_coop
        c_neighbours_of_c, edges_of_c = 0, 0
        c_neighbours_of_d, edges_of_d = 0, 0
        for u, v in G.edges():
            for me, other in [(u, v), (v, u)]:
                if coop[me]:
                    edges_of_c += 1
                    c_neighbours_of_c += coop[other]
                else:
                    edges_of_d += 1
                    c_neighbours_of_d += coop[other]
        by_loop = c_neighbours_of_c / edges_of_c - c_neighbours_of_d / edges_of_d
        assert np.isclose(by_loop, assortment(A, degree, coop)), "assortment differs"
    assert np.isnan(assortment(A, degree, np.ones(A.shape[0], dtype=bool)))
    print("exact check: assortment matches a plain loop over all edges")


def compare(label, mesa_values, fast_values, settings):
    p_value = ttest_ind(mesa_values, fast_values, equal_var=False).pvalue
    if np.isnan(p_value):
        p_value = 1.0  # both versions gave identical constant values
    row = {"agent": label}
    row.update(settings)
    row.update({"mesa_mean": round(float(np.mean(mesa_values)), 4),
                "fast_mean": round(float(np.mean(fast_values)), 4),
                "mesa_std": round(float(np.std(mesa_values)), 4),
                "fast_std": round(float(np.std(fast_values)), 4),
                "difference": round(float(np.mean(fast_values) - np.mean(mesa_values)), 4),
                "p_value": round(float(p_value), 3),
                "agree": bool(p_value > 0.01)})
    print(row)
    return row


def compare_bm_runs(G, A, degree):
    rows = []
    record_from = STEPS - WINDOW
    for b, beta, a_fraction, shuffled, scheme in BM_CELLS:
        aspiration = round(a_fraction * b, 6)
        mesa_values = []
        fast_values = []
        for seed in range(SEEDS):
            model = CooperationModel(G, "bm", b=b, beta=beta, aspiration=aspiration,
                                     epsilon=EPSILON, shuffle_seats=shuffled,
                                     payoff_scheme=scheme, seed=seed)
            model.run(STEPS)
            curve = model.datacollector.model_vars["cooperation"]
            mesa_values.append(np.mean(curve[record_from:]))

            curve, p, nodes, _ = run_bm(A, degree, b, beta, aspiration, EPSILON, STEPS,
                                     record_from=record_from, shuffle_seats=shuffled,
                                     payoff_scheme=scheme, seed=seed)
            fast_values.append(curve[record_from:].mean())
        settings = {"payoff_scheme": scheme, "b": b, "beta": beta, "a_fraction": a_fraction,
                    "kappa": "-", "mutation": "-", "shuffled": shuffled}
        rows.append(compare("bm", mesa_values, fast_values, settings))
    return rows


def compare_fermi_runs(G, A, degree):
    """Fermi has no mutation: runs end in all-C or all-D sooner or later. So compare the
    mean over the whole run, which includes the way there, not only the late window."""
    rows = []
    for b, kappa, shuffled, scheme, mutation in FERMI_CELLS:
        mesa_values = []
        fast_values = []
        for seed in range(SEEDS):
            model = CooperationModel(G, "fermi", b=b, kappa=kappa, payoff_scheme=scheme,
                                     shuffle_seats=shuffled, mutation=mutation, seed=seed)
            model.run(STEPS)
            curve = model.datacollector.model_vars["cooperation"]
            mesa_values.append(np.mean(curve))

            curve, nodes, _ = run_fermi(A, degree, b, kappa, STEPS, payoff_scheme=scheme,
                                     shuffle_seats=shuffled, mutation=mutation, seed=seed)
            fast_values.append(curve.mean())
        settings = {"payoff_scheme": scheme, "b": b, "beta": "-", "a_fraction": "-",
                    "kappa": kappa, "mutation": mutation, "shuffled": shuffled}
        rows.append(compare("fermi", mesa_values, fast_values, settings))
    return rows


def main():
    G = load_graph(GRAPH)
    A, degree = neighbour_matrix(G)
    for scheme in ["average", "accumulated"]:
        check_payoffs(G, A, degree, scheme)
        check_bm_update(G, degree, scheme)
    check_fermi_update(G, A, degree)
    check_random_neighbours(A, degree)
    check_normalisation_on_regular_graph()
    check_assortment(G, A, degree)

    print(f"statistical check: {GRAPH}, {STEPS} steps, {SEEDS} seeds per cell")
    rows = compare_fermi_runs(G, A, degree) + compare_bm_runs(G, A, degree)

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
