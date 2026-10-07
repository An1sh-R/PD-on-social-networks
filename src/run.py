"""Main parameter sweep. One row per run in results/sweep_runs.csv.

Run from project root:
    python src/run.py           full sweep (long, can be stopped and restarted)
    python src/run.py --test    tiny grid, 2 small graphs, few steps (checks the code)

Safe to stop any time: every finished run is written straight to the CSV, and a
restart skips runs already in it.

Graphs: 5 real + 20 synthetic (er, ba, ws, degseq) + 5 "shuffled".
"<real>_shuffled" = the real graph, but players get new random seats after every
round (mentor's idea): same network, no fixed partners.

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
from networks import REAL_FILES, RESULTS_DIR, all_names, load_graph, rewired_names

TEST = "--test" in sys.argv

# ---- grid ------------------------------------------------------------------
# Decided: epsilon 0.05, b = 1.1-2.0, full beta x a grid, 5 reps (20 in high-beta band).
if TEST:
    # 8 graphs: enough for analyze.py to test its regression and real/degseq/er steps
    GRAPHS = ["lastfm", "lastfm_er", "lastfm_ba", "lastfm_ws", "lastfm_degseq",
              "lastfm_shuffled", "twitch", "twitch_er", "twitch_degseq", "twitch_shuffled"]
    BETAS = [0.1, 10.0]
    A_FRACTIONS = [0.3, 0.6]
    TEMPTATIONS = [1.5]
    REPS = 2
    BAND_REPS = 4
    STEPS = 60
    WINDOW = 20
    OUT_NAME = "sweep_test"
else:
    # Decided 2026-10-07: real, degseq, shuffled only (no ER/BA/WS). Small networks
    # first, so their results are complete before the slow GitHub/Deezer runs.
    NETWORK_ORDER = ["facebook", "lastfm", "twitch", "deezer", "github"]
    GRAPHS = []
    for real_name in NETWORK_ORDER:
        GRAPHS.append(real_name)
        GRAPHS.append(real_name + "_degseq")
        GRAPHS.append(real_name + "_shuffled")
    # kept on purpose: beta=0.01 still drifts at 1000 steps, a=0.9 is the dead zone (flag in analysis)
    BETAS = [0.01, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 100.0]
    A_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    TEMPTATIONS = [1.1, 1.3, 1.5, 1.7, 2.0]
    REPS = 5
    BAND_REPS = 20  # high-beta band: repeat runs disagree there (pilot), see reps_for()
    STEPS = 1500
    WINDOW = 500  # final cooperation = mean over the last 500 steps (eps=0.05 pilot: most flat by ~600)
    OUT_NAME = "sweep"

RUN_FERMI = False  # BM only for now; Fermi baseline later (set True, rerun: done runs are skipped)
EPSILON = 0.05  # as Ezaki & Masuda 2017; 0.2 squeezed cooperation into [0.1, 0.9]
KAPPA = 0.1
# 8 processes on this 10-core laptop give ~3x speed-up over 1 (not 8x: cores share memory).
N_PROCESSES = 8
N_BUILD_PROCESSES = 3  # building rewired graphs needs more memory per process

RUNS_PATH = os.path.join(RESULTS_DIR, OUT_NAME + "_runs.csv")
NODES_DIR = os.path.join(RESULTS_DIR, OUT_NAME + "_nodes")  # per-node cooperation

COLUMNS = ["run_id", "graph", "family", "graph_type", "agent", "b", "beta",
           "a_fraction", "aspiration", "epsilon", "kappa", "rep", "source_graph", "steps",
           "cooperation", "cooperation_std", "drift", "mean_p", "seconds"]


# ---- jobs ------------------------------------------------------------------

def make_run_id(job):
    return (f"{job['graph']}|{job['agent']}|b={job['b']}|beta={job['beta']}"
            f"|a={job['a_fraction']}|rep={job['rep']}")


def reps_for(beta, a_fraction):
    """More repeats where the eps=0.05 pilot showed repeat runs ending at different
    cooperation levels (beta >= 3, a = 0.6-0.8). Fixed before the sweep, not chosen
    after seeing results."""
    if beta >= 3.0 and 0.6 <= a_fraction <= 0.8:
        return BAND_REPS
    return REPS


def source_graph(graph, rep):
    """The graph file a run actually loads.
    degseq: rep r uses rewiring r % 5, so the result does not hang on one random graph.
    shuffled: the real graph (players are re-seated during the run instead)."""
    if graph.endswith("_degseq"):
        real_name = graph[:-len("_degseq")]
        names = rewired_names(real_name)
        return names[rep % len(names)]
    if graph.endswith("_shuffled"):
        return graph[:-len("_shuffled")]
    return graph


def make_jobs():
    """Every run in the sweep, grouped by graph (workers then reuse the loaded graph)."""
    jobs = []
    for graph in GRAPHS:
        for b in TEMPTATIONS:
            for beta in BETAS:
                for a_fraction in A_FRACTIONS:
                    for rep in range(reps_for(beta, a_fraction)):
                        jobs.append({"graph": graph, "agent": "bm", "b": b, "beta": beta,
                                     "a_fraction": a_fraction, "rep": rep})
            if not RUN_FERMI:
                continue
            # Fermi does not use beta or aspiration
            for rep in range(REPS):
                for agent in ["fermi_average", "fermi_accumulated"]:
                    jobs.append({"graph": graph, "agent": agent, "b": b, "beta": "-",
                                 "a_fraction": "-", "rep": rep})
    for job in jobs:
        job["run_id"] = make_run_id(job)
        job["source_graph"] = source_graph(job["graph"], job["rep"])
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
    G = get_graph(job["source_graph"])
    record_from = STEPS - WINDOW
    shuffle = job["graph"].endswith("_shuffled")

    if job["agent"] == "bm":
        aspiration = round(job["a_fraction"] * job["b"], 6)  # avoid 0.44999999999999996
        model = CooperationModel(G, "bm", b=job["b"], beta=job["beta"],
                                 aspiration=aspiration, epsilon=EPSILON,
                                 record_from=record_from, shuffle_seats=shuffle,
                                 seed=job["rep"])
    else:
        aspiration = "-"
        scheme = job["agent"].replace("fermi_", "")
        model = CooperationModel(G, "fermi", b=job["b"], payoff_scheme=scheme,
                                 kappa=KAPPA, record_from=record_from,
                                 shuffle_seats=shuffle, seed=job["rep"])
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
        "source_graph": job["source_graph"],
        "steps": STEPS,
        "cooperation": float(window.mean()),
        "cooperation_std": float(window.std()),
        # second half of the window minus first half: ~0 if the run had settled,
        # big if it was still moving (tells slow convergence from bistability)
        "drift": float(window[WINDOW // 2:].mean() - window[:WINDOW // 2].mean()),
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

def build_graph(name):
    load_graph(name)  # builds and caches it in data/synthetic/ if missing
    return name


def build_missing_graphs(jobs):
    """Build every graph file the sweep needs before the runs start, so parallel
    workers never try to create the same file at the same time."""
    needed = []
    for job in jobs:
        if job["source_graph"] not in needed:
            needed.append(job["source_graph"])
    print(f"making sure {len(needed)} graph files exist (first time: a GitHub rewiring takes ~5 min)")
    with Pool(N_BUILD_PROCESSES) as pool:
        for name in pool.imap_unordered(build_graph, needed):
            pass


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(NODES_DIR, exist_ok=True)

    remove_cut_off_line()
    all_jobs = make_jobs()
    build_missing_graphs(all_jobs)
    done = finished_run_ids()
    todo = []
    for job in all_jobs:
        if job["run_id"] not in done:
            todo.append(job)
    # keep GRAPHS order (small networks first); inside a graph, same file next to
    # each other so workers mostly reuse their loaded graph
    graph_position = {}
    for i, graph in enumerate(GRAPHS):
        graph_position[graph] = i
    todo.sort(key=lambda job: (graph_position[job["graph"]], job["source_graph"]))
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
