"""Main parameter sweep. One row per run in results/sweep_runs.csv.

Run from project root:
    python src/run.py           full sweep (long, can be stopped and restarted)
    python src/run.py --test    tiny grid, 2 small graphs, few steps (checks the code)

Safe to stop any time: every finished run is written straight to the CSV, and a
restart skips runs already in it.

Arms (see ResearchQ/design.md):
- bm                 Bush-Mosteller, full beta x aspiration grid
- fermi_average      Fermi imitation, average payoff (fair comparison with BM)
- fermi_accumulated  Fermi imitation, total payoff (classic setup)

Aspiration is given as a fraction of b (A = a * b): the pilot showed the
interesting band moves with b.
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from model import CooperationModel
from networks import REAL_FILES, RESULTS_DIR, all_names, load_graph

TEST = "--test" in sys.argv

# ---- grid ------------------------------------------------------------------
# Decided: epsilon 0.05, b = 1.1-2.0 (5 values). TODO: grid density / budget.
if TEST:
    # 8 graphs: enough for analyze.py to test its regression and real/degseq/er steps
    GRAPHS = ["lastfm", "lastfm_er", "lastfm_ba", "lastfm_ws", "lastfm_degseq",
              "twitch", "twitch_er", "twitch_degseq"]
    BETAS = [0.1, 10.0]
    A_FRACTIONS = [0.3, 0.6]
    TEMPTATIONS = [1.5]
    REPS = 2
    STEPS = 60
    WINDOW = 20
    OUT_NAME = "sweep_test"
else:
    GRAPHS = all_names()  # 5 real + 20 synthetic
    BETAS = [0.01, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 100.0]
    A_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    TEMPTATIONS = [1.1, 1.3, 1.5, 1.7, 2.0]
    REPS = 5
    STEPS = 1000
    WINDOW = 400  # final cooperation = mean over the last 400 steps (pilot: flat by ~600)
    OUT_NAME = "sweep"

EPSILON = 0.05  # as Ezaki & Masuda 2017; 0.2 squeezed cooperation into [0.1, 0.9]
KAPPA = 0.1
# 8 processes on this 10-core laptop give ~3x speed-up over 1 (not 8x: cores share memory).
N_PROCESSES = 8

RUNS_PATH = os.path.join(RESULTS_DIR, OUT_NAME + "_runs.csv")
NODES_DIR = os.path.join(RESULTS_DIR, OUT_NAME + "_nodes")  # per-node cooperation

COLUMNS = ["run_id", "graph", "family", "graph_type", "agent", "b", "beta",
           "a_fraction", "aspiration", "epsilon", "kappa", "rep", "steps",
           "cooperation", "cooperation_std", "mean_p", "seconds"]


# ---- jobs ------------------------------------------------------------------

def make_run_id(job):
    return (f"{job['graph']}|{job['agent']}|b={job['b']}|beta={job['beta']}"
            f"|a={job['a_fraction']}|rep={job['rep']}")


def make_jobs():
    """Every run in the sweep, grouped by graph (workers then reuse the loaded graph)."""
    jobs = []
    for graph in GRAPHS:
        for b in TEMPTATIONS:
            for rep in range(REPS):
                for beta in BETAS:
                    for a_fraction in A_FRACTIONS:
                        jobs.append({"graph": graph, "agent": "bm", "b": b, "beta": beta,
                                     "a_fraction": a_fraction, "rep": rep})
                # Fermi does not use beta or aspiration
                for agent in ["fermi_average", "fermi_accumulated"]:
                    jobs.append({"graph": graph, "agent": agent, "b": b, "beta": "-",
                                 "a_fraction": "-", "rep": rep})
    for job in jobs:
        job["run_id"] = make_run_id(job)
    return jobs


def remove_cut_off_line():
    """If the sweep was killed while writing, the last line has no newline. Drop it,
    otherwise the next row would be glued onto it."""
    if not os.path.exists(RUNS_PATH):
        return
    with open(RUNS_PATH, newline="") as f:
        text = f.read()
    if text == "" or text.endswith("\n"):
        return
    last_newline = text.rfind("\n")
    with open(RUNS_PATH, "w", newline="") as f:
        f.write(text[:last_newline + 1])
    print("removed a cut-off last line from", RUNS_PATH)


def finished_run_ids():
    """run_ids already in the output CSV (so a restart skips them)."""
    done = set()
    if not os.path.exists(RUNS_PATH):
        return done
    with open(RUNS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["run_id"])
    return done


# ---- one run ---------------------------------------------------------------

# Each worker keeps only the last graph it loaded: holding all 25 graphs in
# every worker would use several GB of memory.
cached_name = None
cached_graph = None


def get_graph(name):
    global cached_name, cached_graph
    if name != cached_name:
        cached_graph = load_graph(name)
        cached_name = name
    return cached_graph


def graph_type(name):
    if name in REAL_FILES:
        return "real"
    return name.rsplit("_", 1)[1]


def run_one(job):
    start = time.time()
    G = get_graph(job["graph"])
    record_from = STEPS - WINDOW

    if job["agent"] == "bm":
        aspiration = round(job["a_fraction"] * job["b"], 6)  # avoid 0.44999999999999996
        model = CooperationModel(G, "bm", b=job["b"], beta=job["beta"],
                                 aspiration=aspiration, epsilon=EPSILON,
                                 record_from=record_from, seed=job["rep"])
    else:
        aspiration = "-"
        scheme = job["agent"].replace("fermi_", "")
        model = CooperationModel(G, "fermi", b=job["b"], payoff_scheme=scheme,
                                 kappa=KAPPA, record_from=record_from, seed=job["rep"])
    model.run(STEPS)

    curve = model.datacollector.model_vars["cooperation"]
    window = np.array(curve[record_from:])
    if job["agent"] == "bm":
        p_values = []
        for player in model.players:
            p_values.append(player.p)
        mean_p = float(np.mean(p_values))
    else:
        mean_p = "-"

    row = {
        "run_id": job["run_id"],
        "graph": job["graph"],
        "family": job["graph"].split("_")[0],
        "graph_type": graph_type(job["graph"]),
        "agent": job["agent"],
        "b": job["b"],
        "beta": job["beta"],
        "a_fraction": job["a_fraction"],
        "aspiration": aspiration,
        "epsilon": EPSILON if job["agent"] == "bm" else "-",
        "kappa": KAPPA if job["agent"] != "bm" else "-",
        "rep": job["rep"],
        "steps": STEPS,
        "cooperation": float(window.mean()),
        "cooperation_std": float(window.std()),
        "mean_p": mean_p,
        "seconds": round(time.time() - start, 2),
    }

    # Per-node cooperation: real graphs, first rep only (keeps disk use small).
    node_values = None
    if job["graph"] in REAL_FILES and job["rep"] == 0:
        node_values = np.array(model.node_cooperation(), dtype=np.float32)
    return row, node_values


def node_file_name(run_id):
    # run_id has | and = which are awkward in file names
    return run_id.replace("|", "__").replace("=", "-") + ".npy"


# ---- main ------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(NODES_DIR, exist_ok=True)

    remove_cut_off_line()
    all_jobs = make_jobs()
    done = finished_run_ids()
    todo = []
    for job in all_jobs:
        if job["run_id"] not in done:
            todo.append(job)
    print(f"{len(all_jobs)} runs in sweep, {len(done)} already done, {len(todo)} to go")
    if not todo:
        return

    new_file = not os.path.exists(RUNS_PATH)
    start = time.time()
    with open(RUNS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        with Pool(N_PROCESSES) as pool:
            for i, (row, node_values) in enumerate(pool.imap_unordered(run_one, todo)):
                writer.writerow(row)
                f.flush()  # written to disk now, so a crash loses nothing
                if node_values is not None:
                    np.save(os.path.join(NODES_DIR, node_file_name(row["run_id"])), node_values)
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    minutes = (time.time() - start) / 60
                    left = minutes / (i + 1) * (len(todo) - i - 1)
                    print(f"  {i + 1}/{len(todo)} done, {minutes:.1f} min so far, ~{left:.0f} min left")


if __name__ == "__main__":
    main()
