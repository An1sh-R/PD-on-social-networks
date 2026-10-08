"""Load real networks, build synthetic equivalents, compute topology metrics.

Run `python src/networks.py` from the project root to build all graphs and
write results/topology_table.csv.
"""
import os

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA_DIR = os.path.join(ROOT, "data")
SYNTH_DIR = os.path.join(DATA_DIR, "synthetic")
RESULTS_DIR = os.path.join(ROOT, "results")

REAL_FILES = {
    "facebook": "facebook_combined.txt",
    "deezer": "deezer_edges.csv",
    "github": "github_edges.csv",
    "lastfm": "lastfm_edges.csv",
    "twitch": "twitch_ENGB_edges.csv",
}
SYNTH_TYPES = ["er", "ba", "ws", "degseq"]

SEED = 42
# Rewiring probabilities tried for Watts-Strogatz. We keep the one whose
# clustering is closest to the real graph.
WS_P_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]


def clean(G):
    """Drop self-loops, keep largest connected component, relabel nodes 0..n-1."""
    G = nx.Graph(G)  # also merges any parallel edges
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    biggest = max(nx.connected_components(G), key=len)
    G = G.subgraph(biggest).copy()
    return nx.convert_node_labels_to_integers(G)


def load_real(name):
    """Read an edge list. Handles comma or space separators and a header line."""
    path = os.path.join(DATA_DIR, REAL_FILES[name])
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if not parts[0].isdigit():  # header line
                continue
            G.add_edge(int(parts[0]), int(parts[1]))
    return clean(G)


def mean_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()


def make_er(n, avg_k):
    p = avg_k / (n - 1)
    return clean(nx.fast_gnp_random_graph(n, p, seed=SEED))


def make_ba(n, avg_k):
    m = max(1, round(avg_k / 2))
    return clean(nx.barabasi_albert_graph(n, m, seed=SEED))


def make_ws(n, avg_k, target_clustering):
    k = max(2, 2 * round(avg_k / 2))  # WS needs an even k
    best_graph = None
    best_gap = None
    best_p = None
    for p in WS_P_GRID:
        G = nx.watts_strogatz_graph(n, k, p, seed=SEED)
        gap = abs(nx.average_clustering(G) - target_clustering)
        if best_gap is None or gap < best_gap:
            best_graph, best_gap, best_p = G, gap, p
    print(f"  WS: k={k}, chose p={best_p}")
    return clean(best_graph)


def reconnect_pieces(G, rng):
    """Join every small piece to the largest one without changing any degree.

    For a piece edge (u, v) and a main-component edge (x, y): replace them with
    (u, x) and (v, y). Degrees stay the same. (x, y) is never a bridge, so the main
    component stays connected, and both halves of the piece hang on to it.
    Fixed 2026-10-08: isolated pairs left by the swaps cooperated ~0.84 at high beta
    and pushed degseq cooperation up by ~0.004 (LastFM, Deezer).
    """
    pieces = sorted(nx.connected_components(G), key=len, reverse=True)
    for piece in pieces[1:]:
        main = max(nx.connected_components(G), key=len)
        main_graph = G.subgraph(main)
        bridges = set()
        for x, y in nx.bridges(main_graph):
            bridges.add((x, y))
            bridges.add((y, x))
        main_edges = []
        for edge in main_graph.edges():
            if edge not in bridges:
                main_edges.append(edge)
        piece_edges = list(G.subgraph(piece).edges())
        u, v = piece_edges[rng.integers(len(piece_edges))]
        x, y = main_edges[rng.integers(len(main_edges))]
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        G.add_edge(u, x)
        G.add_edge(v, y)


def make_degseq(G_real, seed=SEED):
    """Shuffle a copy of the real graph, keeping every node's degree exactly, connected.

    Done once to generate a fixed graph. (nx.configuration_model was tried first,
    but it loses hub edges on heavy-tailed graphs.)
    Swaps can cut off a few tiny pieces (mostly two degree-1 nodes linked to each
    other). Dropping them would break the degree sequence, so they are joined back
    to the main component instead (reconnect_pieces).
    """
    G = G_real.copy()
    nswap = 10 * G.number_of_edges()
    nx.double_edge_swap(G, nswap=nswap, max_tries=100 * nswap, seed=seed)
    reconnect_pieces(G, np.random.default_rng(seed))
    real_degrees = sorted(d for _, d in G_real.degree())
    new_degrees = sorted(d for _, d in G.degree())
    assert new_degrees == real_degrees, "degree sequence changed"
    assert nx.is_connected(G), "rewired graph not connected"
    assert nx.number_of_selfloops(G) == 0, "self-loop"
    return G


# Structure test (structure_test.py) uses 5 independent degree-preserving
# shuffles per real graph, named e.g. "github_rewired0" .. "github_rewired4".
N_REWIRINGS = 5


def rewired_names(real_name):
    names = []
    for r in range(N_REWIRINGS):
        names.append(f"{real_name}_rewired{r}")
    return names


def make_synthetic(real_name, G_real, kind):
    n = G_real.number_of_nodes()
    avg_k = mean_degree(G_real)
    if kind == "er":
        return make_er(n, avg_k)
    if kind == "ba":
        return make_ba(n, avg_k)
    if kind == "ws":
        return make_ws(n, avg_k, nx.average_clustering(G_real))
    if kind.startswith("rewired"):
        r = int(kind[len("rewired"):])
        return make_degseq(G_real, seed=1000 + r)  # different seed per rewiring
    return make_degseq(G_real)


def all_names():
    """Real graph names first, then their synthetic equivalents."""
    names = list(REAL_FILES)
    for real in REAL_FILES:
        for kind in SYNTH_TYPES:
            names.append(f"{real}_{kind}")
    return names


def synthetic_path(name):
    return os.path.join(SYNTH_DIR, name + ".edgelist")


def load_graph(name):
    """Return graph by name, e.g. 'github' or 'github_ba'. Builds and caches synthetic ones."""
    if name in REAL_FILES:
        return load_real(name)
    path = synthetic_path(name)
    if not os.path.exists(path):
        real_name, kind = name.rsplit("_", 1)
        print(f"building {name}")
        G = make_synthetic(real_name, load_real(real_name), kind)
        os.makedirs(SYNTH_DIR, exist_ok=True)
        # write to a temporary file, then rename: if the program is stopped while
        # writing, no half-written graph file is left behind to be loaded later
        temp_path = path + ".tmp"
        nx.write_edgelist(G, temp_path, data=False)
        os.replace(temp_path, path)
    return nx.read_edgelist(path, nodetype=int)


def check_graph(G):
    """Sanity checks every graph must pass before simulation."""
    assert nx.number_of_selfloops(G) == 0, "self-loops present"
    assert nx.is_connected(G), "graph not connected"
    assert nx.number_of_isolates(G) == 0, "node with no neighbours"
    assert sorted(G.nodes()) == list(range(G.number_of_nodes())), "node labels not 0..n-1"


def topology_metrics(G, name):
    """One row of the topology table."""
    degrees = np.array([d for _, d in G.degree()])
    communities = nx.community.louvain_communities(G, seed=SEED)
    return {
        "name": name,
        "N": G.number_of_nodes(),
        "E": G.number_of_edges(),
        "mean_degree": degrees.mean(),
        "max_degree": degrees.max(),
        "degree_cv": degrees.std() / degrees.mean(),
        "clustering": nx.average_clustering(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
        "modularity": nx.community.modularity(G, communities),
    }


def build_table():
    """One row per graph: real, ER/BA/WS/degseq, and the 5 rewirings the sweep uses
    for "<real>_degseq" (analyze.py averages those 5)."""
    names = all_names()
    for real in REAL_FILES:
        names = names + rewired_names(real)
    rows = []
    for name in names:
        G = load_graph(name)
        check_graph(G)
        rows.append(topology_metrics(G, name))
        print(f"done {name}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = build_table()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "topology_table.csv"), index=False)
    print(table.round(3).to_string(index=False))
