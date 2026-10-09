"""Parameter sweep, version 2: BM and Fermi, average and accumulated payoffs.
One row per run in results/sweep_v2_runs.csv.

Run from project root:
    python src/run.py                  full sweep (long, can be stopped and restarted)
    python src/run.py --test           tiny grid, 2 small families, few steps (checks the code)
    python src/run.py --processes 4    fewer worker processes, e.g. while using the laptop
                                       (default 12; any number, can change between restarts)

Safe to stop any time (Ctrl+C): every finished run is written straight to the CSV,
and a restart skips runs already in it. Only the runs in progress are lost.

Graphs: for each of the 5 real networks: real, degseq, shuffled, er, ba, ws.
"<real>_shuffled" = the real graph, but players get new random seats after every
round (mentor's idea): same network, no fixed partners.

Arms (agent x payoff_scheme, see ResearchQ/design.md):
- bm average          Bush-Mosteller, full beta x aspiration grid (main sweep)
- bm accumulated      payoff summed over neighbours, aspiration A * k of the seat
- fermi average       Fermi imitation, kappa grid, mutation 0.01 (3000 steps)
- fermi accumulated   same, summed payoff

Accumulated payoffs are scaled by the graph's mean degree <k> (decided 2026-10-08):
BM runs with beta / <k>, Fermi with kappa * <k>. Then a player of average degree
has the same sensitivity in both schemes; only the spread across degrees differs
(hubs more, leaves less). Without it, "accumulated - average" would mostly measure
<k> (k times higher sensitivity). beta and kappa columns hold the grid values,
beta_used and kappa_used what the model got.

Aspiration is given as a fraction of b (A = a * b): the pilot showed the
interesting band moves with b.

All runs use the numpy models (fast_model.py), checked against Mesa by check_fast_model.py.
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from fast_model import neighbour_matrix, run_bm, run_fermi
from networks import REAL_FILES, RESULTS_DIR, load_graph, rewired_names

TEST = "--test" in sys.argv

# ---- grid ------------------------------------------------------------------
if TEST:
    FAMILIES = ["lastfm", "twitch"]
    BM_AVERAGE_BETAS = [0.1, 10.0]
    BM_AVERAGE_A = [0.3, 0.7]  # beta=10, a=0.7, b=2 oscillates: tests that flag
    BM_ACCUMULATED_BETAS = [0.1, 10.0]
    BM_ACCUMULATED_A = [0.3]
    BM_TEMPTATIONS = [2.0]
    KAPPAS = [0.1, 10.0]
    MUTATIONS = [0.01]
    FERMI_TEMPTATIONS = [1.05]
    REPS = 2
    STEPS = 300
    WINDOW = 100
    FERMI_STEPS = 400
    FERMI_WINDOW = 150
    OUT_NAME = "sweep_v2_test"
else:
    # small networks first, so their results are complete before GitHub/Deezer
    FAMILIES = ["facebook", "lastfm", "twitch", "deezer", "github"]
    # BM average: same grid as the main sweep. All rerun (2026-10-09): the main sweep
    # has no assortment column and its degseq graphs were not connected.
    # beta=0.01 still drifts, a=0.9 is the dead zone: kept, flagged in the analysis.
    BM_AVERAGE_BETAS = [0.01, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 100.0]
    BM_AVERAGE_A = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # BM accumulated: drop beta=0.01 (too slow), a=0 (nobody ever unhappy), a=0.9 (dead zone)
    BM_ACCUMULATED_BETAS = [0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 100.0]
    BM_ACCUMULATED_A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    BM_TEMPTATIONS = [1.1, 1.3, 1.5, 1.7, 2.0]
    KAPPAS = [0.01, 0.1, 1.0, 10.0]
    # Mutation 0.01: no getting stuck in all-C / all-D (like BM's epsilon). Classic
    # Fermi (mutation 0) dropped 2026-10-08: pilot_payoffs.py showed it has not
    # settled even at 4000 steps (seeds split 0.28 / 1.0 / 1.0), so a fixed window
    # would only catch the transient.
    MUTATIONS = [0.01]
    # Fermi with average payoff sits at the mutation floor (~0.01-0.05) for b >= 1.02
    # at kappa <= 0.1; b = 1.005, 1.01 lift it off (LastFM 0.08-0.11, BA 0.39-0.44)
    FERMI_TEMPTATIONS = [1.005, 1.01, 1.02, 1.05, 1.1, 1.3, 1.5, 1.7, 2.0]
    REPS = 10
    STEPS = 1500  # BM: settled by then (pilot.py, pilot_payoffs.py)
    WINDOW = 500  # final cooperation = mean over the last 500 steps
    # Fermi with accumulated payoff fluctuates slowly (pilot_payoffs.py: seeds differ
    # ~0.02, GitHub still drifting ~+0.015 after 1500 steps): longer run, longer window
    FERMI_STEPS = 3000
    FERMI_WINDOW = 1000
    OUT_NAME = "sweep_v2"

GRAPHS = []
for family in FAMILIES:
    for kind in ["", "_degseq", "_shuffled", "_er", "_ba", "_ws"]:
        GRAPHS.append(family + kind)

EPSILON = 0.05  # BM slip = flip the intended action, as Ezaki & Masuda 2017
DEFAULT_PROCESSES = 12  # laptop otherwise idle; fewer with --processes when using it


def processes_from_command_line():
    """Number after --processes, or DEFAULT_PROCESSES. Read in main() only: worker
    processes import this file too and must not stop on a bad argument."""
    if "--processes" not in sys.argv:
        return DEFAULT_PROCESSES
    position = sys.argv.index("--processes")
    if position + 1 >= len(sys.argv) or not sys.argv[position + 1].isdigit():
        sys.exit("--processes needs a whole number, e.g. --processes 4")
    n_processes = int(sys.argv[position + 1])
    if n_processes < 1:
        sys.exit("--processes must be at least 1")
    return n_processes

RUNS_PATH = os.path.join(RESULTS_DIR, OUT_NAME + "_runs.csv")
NODES_DIR = os.path.join(RESULTS_DIR, OUT_NAME + "_nodes")  # per-node cooperation

COLUMNS = ["run_id", "graph", "family", "graph_type", "agent", "payoff_scheme", "b", "beta",
           "a_fraction", "aspiration", "epsilon", "kappa", "mutation", "mean_degree",
           "beta_used", "kappa_used", "rep", "source_graph", "steps", "window", "cooperation",
           "cooperation_std", "drift", "even_odd_gap", "mean_p", "assortment", "absorbed_at",
           "seconds"]


# ---- jobs ------------------------------------------------------------------

def make_run_id(job):
    return (f"{job['graph']}|{job['agent']}|{job['payoff_scheme']}|b={job['b']}"
            f"|beta={job['beta']}|a={job['a_fraction']}|kappa={job['kappa']}"
            f"|mu={job['mutation']}|rep={job['rep']}")


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


def bm_jobs(graph, scheme, betas, a_fractions):
    jobs = []
    for b in BM_TEMPTATIONS:
        for beta in betas:
            for a_fraction in a_fractions:
                for rep in range(REPS):
                    jobs.append({"graph": graph, "agent": "bm", "payoff_scheme": scheme, "b": b,
                                 "beta": beta, "a_fraction": a_fraction, "kappa": "-",
                                 "mutation": "-", "rep": rep})
    return jobs


def fermi_jobs(graph):
    jobs = []
    for scheme in ["average", "accumulated"]:
        for b in FERMI_TEMPTATIONS:
            for kappa in KAPPAS:
                for mutation in MUTATIONS:
                    for rep in range(REPS):
                        jobs.append({"graph": graph, "agent": "fermi", "payoff_scheme": scheme,
                                     "b": b, "beta": "-", "a_fraction": "-", "kappa": kappa,
                                     "mutation": mutation, "rep": rep})
    return jobs


def make_jobs():
    """Every run in the sweep, grouped by graph (workers then reuse the loaded graph)."""
    jobs = []
    for graph in GRAPHS:
        jobs.extend(bm_jobs(graph, "average", BM_AVERAGE_BETAS, BM_AVERAGE_A))
        jobs.extend(bm_jobs(graph, "accumulated", BM_ACCUMULATED_BETAS, BM_ACCUMULATED_A))
        jobs.extend(fermi_jobs(graph))
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

# Each worker keeps only the last graph it loaded: holding all graphs in
# every worker would use several GB of memory.
cached_name = None
cached_matrix = None  # (A, degree)


def get_matrix(name):
    global cached_name, cached_matrix
    if name != cached_name:
        cached_matrix = neighbour_matrix(load_graph(name))
        cached_name = name
    return cached_matrix


def graph_type(name):
    if name in REAL_FILES:
        return "real"
    return name.rsplit("_", 1)[1]


def keeps_node_values(job):
    """Per-node cooperation: real and degseq graphs, first rep only (disk use)."""
    return job["rep"] == 0 and graph_type(job["graph"]) in ("real", "degseq")


def run_one(job):
    start = time.time()
    A, degree = get_matrix(job["source_graph"])
    mean_degree = float(degree.mean())
    steps = STEPS
    window_length = WINDOW
    if job["agent"] == "fermi":
        steps = FERMI_STEPS
        window_length = FERMI_WINDOW
    record_from = steps - window_length
    shuffle = job["graph"].endswith("_shuffled")
    scheme = job["payoff_scheme"]
    beta_used = "-"
    kappa_used = "-"

    if job["agent"] == "bm":
        aspiration = round(job["a_fraction"] * job["b"], 6)  # avoid 0.44999999999999996
        beta_used = job["beta"]
        if scheme == "accumulated":
            beta_used = job["beta"] / mean_degree  # mean-degree normalisation (docstring)
        curve, p, node_cooperation, mean_assortment = run_bm(A, degree, job["b"], beta_used, aspiration,
                                            EPSILON, steps, record_from=record_from,
                                            shuffle_seats=shuffle, payoff_scheme=scheme,
                                            seed=job["rep"])
        mean_p = float(p.mean())
    else:
        aspiration = "-"
        kappa_used = job["kappa"]
        if scheme == "accumulated":
            kappa_used = job["kappa"] * mean_degree  # mean-degree normalisation (docstring)
        curve, node_cooperation, mean_assortment = run_fermi(A, degree, job["b"], kappa_used, steps,
                                            payoff_scheme=scheme, record_from=record_from,
                                            shuffle_seats=shuffle, mutation=job["mutation"],
                                            seed=job["rep"])
        mean_p = "-"

    window = curve[record_from:]
    # first round where everyone plays the same (Fermi without mutation can get stuck)
    everyone_same = np.where((curve == 0) | (curve == 1))[0]
    absorbed_at = -1
    if len(everyone_same) > 0:
        absorbed_at = int(everyone_same[0])

    row = {
        "run_id": job["run_id"],
        "graph": job["graph"],
        "family": job["graph"].split("_")[0],
        "graph_type": graph_type(job["graph"]),
        "agent": job["agent"],
        "payoff_scheme": scheme,
        "b": job["b"],
        "beta": job["beta"],
        "a_fraction": job["a_fraction"],
        "aspiration": aspiration,
        "epsilon": EPSILON if job["agent"] == "bm" else "-",
        "kappa": job["kappa"],
        "mutation": job["mutation"],
        "mean_degree": round(mean_degree, 4),
        "beta_used": beta_used,
        "kappa_used": kappa_used,
        "rep": job["rep"],
        "source_graph": job["source_graph"],
        "steps": steps,
        "window": window_length,
        "cooperation": float(window.mean()),
        "cooperation_std": float(window.std()),
        # second half of the window minus first half: ~0 if the run had settled
        "drift": float(window[window_length // 2:].mean() - window[:window_length // 2].mean()),
        # every-other-round gap: big if the whole population flips C <-> D each round
        "even_odd_gap": float(abs(window[0::2].mean() - window[1::2].mean())),
        "mean_p": mean_p,
        # do cooperators sit next to cooperators? (~0 on shuffled seats: a check)
        "assortment": mean_assortment,
        "absorbed_at": absorbed_at,
        "seconds": round(time.time() - start, 2),
    }

    node_values = None
    if keeps_node_values(job):
        node_values = np.array(node_cooperation, dtype=np.float32)
    return row, node_values


def node_file_name(run_id):
    # run_id has | and = which are awkward in file names
    return run_id.replace("|", "__").replace("=", "-") + ".npy"


def save_node_file(run_id, node_values):
    """Save to a temporary file, then rename: a crash never leaves a half-written file."""
    path = os.path.join(NODES_DIR, node_file_name(run_id))
    temp_path = path[:-len(".npy")] + "_tmp.npy"  # np.save adds .npy if missing
    np.save(temp_path, node_values)
    os.replace(temp_path, path)


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
    print(f"making sure {len(needed)} graph files exist")
    with Pool(3) as pool:  # building rewired graphs needs more memory per process
        for name in pool.imap_unordered(build_graph, needed):
            pass


def main():
    n_processes = processes_from_command_line()
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
    print(f"{len(all_jobs)} runs in sweep, {len(done)} already done, {len(todo)} to go, "
          f"{n_processes} processes")
    if not todo:
        return

    new_file = not os.path.exists(RUNS_PATH)
    start = time.time()
    with open(RUNS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        with Pool(n_processes) as pool:
            for i, (row, node_values) in enumerate(pool.imap_unordered(run_one, todo)):
                # node file first: the CSV row marks the run as done. A crash in between
                # just means the run is redone (and its node file overwritten).
                if node_values is not None:
                    save_node_file(row["run_id"], node_values)
                writer.writerow(row)
                f.flush()  # hand the row to Windows right away
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    os.fsync(f.fileno())  # make sure it is on disk (power cut loses < 20 runs)
                    minutes = (time.time() - start) / 60
                    left = minutes / (i + 1) * (len(todo) - i - 1)
                    print(f"  {i + 1}/{len(todo)} done, {minutes:.1f} min so far, ~{left:.0f} min left")


if __name__ == "__main__":
    main()
