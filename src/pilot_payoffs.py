"""Pilot for the payoff-scheme sweep: are 1500 steps (window = last 500) enough?

New in that sweep: Fermi (kappa grid, with and without mutation) and BM with
accumulated payoffs (low-degree players learn at beta * k / <k>, so they may be slow).
Runs 4000 steps and compares the sweep window (steps 1000-1500) with the end (3500-4000).

Run from project root: python src/pilot_payoffs.py   (~10 min)
Output: results/pilot_payoffs.csv
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fast_model import neighbour_matrix, run_bm, run_fermi
from networks import RESULTS_DIR, load_graph

GRAPHS = ["facebook", "lastfm", "github"]  # densest, pilot graph, biggest hubs
STEPS = 4000
SWEEP_WINDOW = (1000, 1500)
END_WINDOW = (3500, 4000)
SEEDS = 3
EPSILON = 0.05
KAPPAS = [0.01, 0.1, 1.0, 10.0]
MUTATIONS = [0.0, 0.01]
FERMI_TEMPTATIONS = [1.02, 1.3]
BM_BETAS = [0.1, 1.0, 10.0]
BM_A_FRACTIONS = [0.3, 0.6]
BM_B = 1.5


def make_jobs():
    jobs = []
    for graph in GRAPHS:
        for seed in range(SEEDS):
            for scheme in ["average", "accumulated"]:
                for kappa in KAPPAS:
                    for mutation in MUTATIONS:
                        for b in FERMI_TEMPTATIONS:
                            jobs.append({"graph": graph, "agent": "fermi", "payoff_scheme": scheme,
                                         "kappa": kappa, "mutation": mutation, "b": b,
                                         "beta": None, "a_fraction": None, "seed": seed})
            for beta in BM_BETAS:
                for a_fraction in BM_A_FRACTIONS:
                    jobs.append({"graph": graph, "agent": "bm", "payoff_scheme": "accumulated",
                                 "kappa": None, "mutation": None, "b": BM_B, "beta": beta,
                                 "a_fraction": a_fraction, "seed": seed})
    return jobs


cached_name = None
cached_matrix = None


def run_one(job):
    global cached_name, cached_matrix
    if job["graph"] != cached_name:
        cached_matrix = neighbour_matrix(load_graph(job["graph"]))
        cached_name = job["graph"]
    A, degree = cached_matrix
    mean_degree = degree.mean()
    start = time.time()
    if job["agent"] == "fermi":
        kappa = job["kappa"]
        if job["payoff_scheme"] == "accumulated":
            kappa = kappa * mean_degree  # same normalisation as run.py
        curve, _ = run_fermi(A, degree, job["b"], kappa, STEPS, payoff_scheme=job["payoff_scheme"],
                             mutation=job["mutation"], seed=job["seed"])
    else:
        aspiration = job["a_fraction"] * job["b"]
        curve, _, _ = run_bm(A, degree, job["b"], job["beta"] / mean_degree, aspiration, EPSILON,
                             STEPS, payoff_scheme="accumulated", seed=job["seed"])
    row = dict(job)
    row["sweep_window"] = curve[SWEEP_WINDOW[0]:SWEEP_WINDOW[1]].mean()
    row["end_window"] = curve[END_WINDOW[0]:END_WINDOW[1]].mean()
    row["change"] = row["end_window"] - row["sweep_window"]
    # first step where everyone plays the same (only possible without mutation)
    absorbed = np.where((curve == 0) | (curve == 1))[0]
    row["absorbed_at"] = int(absorbed[0]) if len(absorbed) > 0 else -1
    row["seconds"] = round(time.time() - start, 1)
    return row


def main():
    jobs = make_jobs()
    print(len(jobs), "pilot runs")
    rows = []
    with Pool(8) as pool:
        for row in pool.imap_unordered(run_one, jobs):
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "pilot_payoffs.csv"), index=False)

    pd.set_option("display.width", 200)
    keys_fermi = ["agent", "payoff_scheme", "mutation", "kappa"]
    fermi = table[table["agent"] == "fermi"]
    print("\nFermi: change from sweep window (1000-1500) to end (3500-4000), max |change| over graphs, b, seeds")
    summary = fermi.groupby(keys_fermi).agg(
        mean_sweep_window=("sweep_window", "mean"), max_abs_change=("change", lambda x: x.abs().max()),
        share_absorbed_by_1500=("absorbed_at", lambda x: ((x >= 0) & (x < 1500)).mean()),
        share_absorbed_ever=("absorbed_at", lambda x: (x >= 0).mean()))
    print(summary.round(3).to_string())
    bm = table[table["agent"] == "bm"]
    print("\nBM accumulated: change from sweep window to end")
    summary = bm.groupby(["graph", "beta", "a_fraction"]).agg(
        mean_sweep_window=("sweep_window", "mean"), max_abs_change=("change", lambda x: x.abs().max()))
    print(summary.round(3).to_string())
    print("\nseconds per 4000-step run, by graph:", table.groupby("graph")["seconds"].mean().round(1).to_dict())


if __name__ == "__main__":
    main()
