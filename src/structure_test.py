"""Structure test (mentor's design): does network structure beyond the degree
sequence change BM cooperation?

Three conditions per real network:
  original  the real graph
  rewired   degree-preserving shuffle of the real graph: every node keeps its degree,
            clustering and communities are destroyed. 5 independent shuffles;
            seed s uses shuffle (s % 5), so 20 seeds = 4 per shuffle.
  shuffled  the real graph, but players get new random seats after every round:
            same seats (same degrees), no fixed partners.

Run from project root:
    python src/structure_test.py          full run (resumable, safe to stop)
    python src/structure_test.py --test   tiny version on LastFM
Then: python src/structure_summary.py

Writes results/structure_runs.csv (one row per run) and
results/structure_nodes.csv (cooperation by degree bin, original + rewired only).
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from model import CooperationModel
from networks import REAL_FILES, RESULTS_DIR, load_graph, rewired_names

TEST = "--test" in sys.argv

# ---- settings (mentor's file, except: our 5 networks, step count from our pilot) ----
CONDITIONS = ["original", "rewired", "shuffled"]
EPSILON = 0.05
INITIAL_COOP = 0.8
if TEST:
    NETWORKS = ["lastfm"]
    B_VALUES = [1.1, 2.0]
    ASPIRATIONS = [0.3, 0.7]
    BETAS = [0.5]
    SEEDS = list(range(5))
    STEPS = 60
    WINDOW = 20
    OUT_NAME = "structure_test"
else:
    NETWORKS = list(REAL_FILES)
    B_VALUES = [1.1, 1.3, 1.5, 1.7, 2.0]
    ASPIRATIONS = [0.1, 0.3, 0.5, 0.7, 0.9]  # absolute A, as in mentor's file
    BETAS = [0.5]
    SEEDS = list(range(20))
    STEPS = 600  # TODO: set from the eps=0.05 re-pilot
    WINDOW = 200  # equilibrium = mean cooperation over the last 200 steps
    OUT_NAME = "structure"

# Memory is tight on this laptop (~2 GB free): a GitHub worker holds a big graph.
# Lower these if you get MemoryError.
N_PROCESSES = 8
N_BUILD_PROCESSES = 3  # building rewired graphs needs more memory per process
RUNS_PATH = os.path.join(RESULTS_DIR, OUT_NAME + "_runs.csv")
NODES_PATH = os.path.join(RESULTS_DIR, OUT_NAME + "_nodes.csv")
RUN_COLUMNS = ["run_id", "network", "condition", "graph", "b", "aspiration", "beta",
               "seed", "steps", "equilibrium", "seconds"]
NODE_COLUMNS = ["run_id", "network", "condition", "b", "aspiration", "beta", "seed",
                "degree_bin", "n_nodes", "coop"]

# degree bins for cooperation-vs-degree (as in mentor's file)
DEGREE_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64),
               (65, 128), (129, 256), (257, 10**9)]


def degree_bin_label(degree):
    for low, high in DEGREE_BINS:
        if low <= degree <= high:
            if low == high:
                return str(low)
            if high == 10**9:
                return f"{low}+"
            return f"{low}-{high}"
    raise ValueError(degree)


# ---- jobs ------------------------------------------------------------------

def graph_name(network, condition, seed):
    if condition == "rewired":
        return rewired_names(network)[seed % len(rewired_names(network))]
    return network  # original and shuffled both use the real graph


def make_jobs():
    jobs = []
    for network in NETWORKS:
        for condition in CONDITIONS:
            for beta in BETAS:
                for b in B_VALUES:
                    for aspiration in ASPIRATIONS:
                        for seed in SEEDS:
                            job = {"network": network, "condition": condition, "beta": beta,
                                   "b": b, "aspiration": aspiration, "seed": seed,
                                   "graph": graph_name(network, condition, seed)}
                            job["run_id"] = (f"{network}|{condition}|beta={beta}|b={b}"
                                             f"|A={aspiration}|seed={seed}")
                            jobs.append(job)
    return jobs


def remove_cut_off_line(path):
    """A run killed while writing leaves a last line with no newline. Drop it."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        text = f.read()
    if text == "" or text.endswith("\n"):
        return
    with open(path, "w", newline="") as f:
        f.write(text[:text.rfind("\n") + 1])


def finished_run_ids():
    done = set()
    if not os.path.exists(RUNS_PATH):
        return done
    with open(RUNS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["run_id"])
    return done


# ---- one run ---------------------------------------------------------------

cached_name = None  # each worker keeps only its last graph (memory)
cached_graph = None


def get_graph(name):
    global cached_name, cached_graph
    if name != cached_name:
        cached_graph = load_graph(name)
        cached_name = name
    return cached_graph


def run_one(job):
    start = time.time()
    G = get_graph(job["graph"])
    record_from = STEPS - WINDOW
    model = CooperationModel(G, "bm", b=job["b"], beta=job["beta"],
                             aspiration=job["aspiration"], epsilon=EPSILON,
                             initial_coop=INITIAL_COOP, record_from=record_from,
                             shuffle_seats=(job["condition"] == "shuffled"),
                             seed=job["seed"])
    model.run(STEPS)
    curve = model.datacollector.model_vars["cooperation"]
    equilibrium = float(np.mean(curve[record_from:]))

    run_row = {"run_id": job["run_id"], "network": job["network"],
               "condition": job["condition"], "graph": job["graph"], "b": job["b"],
               "aspiration": job["aspiration"], "beta": job["beta"], "seed": job["seed"],
               "steps": STEPS, "equilibrium": equilibrium,
               "seconds": round(time.time() - start, 2)}

    # Cooperation by degree bin. Not for "shuffled": players there have no fixed degree.
    node_rows = []
    if job["condition"] != "shuffled":
        node_coop = model.node_cooperation()
        totals = {}  # bin label -> [number of nodes, summed cooperation]
        for node in range(G.number_of_nodes()):
            label = degree_bin_label(G.degree(node))
            if label not in totals:
                totals[label] = [0, 0.0]
            totals[label][0] += 1
            totals[label][1] += node_coop[node]
        for label, (n_nodes, coop_sum) in totals.items():
            node_rows.append({"run_id": job["run_id"], "network": job["network"],
                              "condition": job["condition"], "b": job["b"],
                              "aspiration": job["aspiration"], "beta": job["beta"],
                              "seed": job["seed"], "degree_bin": label,
                              "n_nodes": n_nodes, "coop": coop_sum / n_nodes})
    return run_row, node_rows


# ---- main ------------------------------------------------------------------

def build_graph(name):
    load_graph(name)  # builds and caches it in data/synthetic/ if missing
    return name


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build rewired graphs first, one process per graph, so workers never
    # try to create the same file at the same time.
    to_build = []
    for network in NETWORKS:
        to_build.extend(rewired_names(network))
    print(f"making sure {len(to_build)} rewired graphs exist (first time: GitHub ~5 min each)")
    with Pool(N_BUILD_PROCESSES) as pool:
        for name in pool.imap_unordered(build_graph, to_build):
            print("  ready:", name)

    remove_cut_off_line(RUNS_PATH)
    remove_cut_off_line(NODES_PATH)
    all_jobs = make_jobs()
    done = finished_run_ids()
    todo = []
    for job in all_jobs:
        if job["run_id"] not in done:
            todo.append(job)
    # sort by graph so each worker mostly reuses its loaded graph
    todo.sort(key=lambda job: job["graph"])
    print(f"{len(all_jobs)} runs, {len(done)} already done, {len(todo)} to go")
    if not todo:
        return

    runs_new = not os.path.exists(RUNS_PATH)
    nodes_new = not os.path.exists(NODES_PATH)
    start = time.time()
    with open(RUNS_PATH, "a", newline="") as runs_file, open(NODES_PATH, "a", newline="") as nodes_file:
        runs_writer = csv.DictWriter(runs_file, fieldnames=RUN_COLUMNS)
        nodes_writer = csv.DictWriter(nodes_file, fieldnames=NODE_COLUMNS)
        if runs_new:
            runs_writer.writeheader()
        if nodes_new:
            nodes_writer.writeheader()
        with Pool(N_PROCESSES) as pool:
            for i, (run_row, node_rows) in enumerate(pool.imap_unordered(run_one, todo, chunksize=4)):
                # node rows first: the run row marks the run as done. If we crash
                # in between, the run is redone and summary drops duplicate node rows.
                nodes_writer.writerows(node_rows)
                nodes_file.flush()
                runs_writer.writerow(run_row)
                runs_file.flush()
                if (i + 1) % 50 == 0 or i + 1 == len(todo):
                    minutes = (time.time() - start) / 60
                    left = minutes / (i + 1) * (len(todo) - i - 1)
                    print(f"  {i + 1}/{len(todo)} done, {minutes:.1f} min, ~{left:.0f} min left")


if __name__ == "__main__":
    main()
