"""Draw every real network next to its synthetic equivalents.

Run `python src/plot_graphs.py` from the project root (after src/networks.py).
Writes one figure per real network to graphs/.

Each figure:
- top row: node-link drawing of a 300-node sample of each graph
- bottom: degree distribution of the FULL graphs, all 5 overlaid
"""
import os

import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from networks import REAL_FILES, RESULTS_DIR, ROOT, SEED, SYNTH_TYPES, load_graph

GRAPHS_DIR = os.path.join(ROOT, "graphs")
SAMPLE_SIZE = 300

LABELS = {
    "real": "Real",
    "er": "Erdos-Renyi",
    "ba": "Barabasi-Albert",
    "ws": "Watts-Strogatz",
    "degseq": "Degree-preserving shuffle",
}
LINE_COLORS = {
    "real": "black",
    "er": "tab:blue",
    "ba": "tab:red",
    "ws": "tab:green",
    "degseq": "tab:orange",
}


def snowball_sample(G, size):
    """First `size` nodes reached by a breadth-first walk from a random start node."""
    rng = np.random.default_rng(SEED)
    start = int(rng.choice(list(G.nodes())))
    nodes = [start]
    for _, v in nx.bfs_edges(G, start):
        nodes.append(v)
        if len(nodes) == size:
            break
    return G.subgraph(nodes)


def draw_sample(ax, G, title, vmin, vmax):
    """Draw a sample of G. Node colour and size show the node's degree in the FULL graph."""
    sample = snowball_sample(G, SAMPLE_SIZE)
    pos = nx.spring_layout(sample, seed=SEED)
    log_degrees = [np.log10(G.degree(n)) for n in sample.nodes()]
    sizes = [8 + 14 * d for d in log_degrees]

    nx.draw_networkx_edges(sample, pos, ax=ax, alpha=0.2, width=0.4)
    nx.draw_networkx_nodes(
        sample, pos, ax=ax, node_size=sizes, node_color=log_degrees,
        cmap="viridis", vmin=vmin, vmax=vmax,
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def degree_ccdf(G):
    """Fraction of nodes with degree >= d, for each degree d in the graph."""
    degrees = np.sort([d for _, d in G.degree()])
    fraction = (len(degrees) - np.arange(len(degrees))) / len(degrees)
    return degrees, fraction


def make_title(label, name, G, table):
    row = table.loc[name]
    return (f"{label}\nN={G.number_of_nodes():,}  E={G.number_of_edges():,}\n"
            f"clustering={row['clustering']:.3f}  deg CV={row['degree_cv']:.2f}")


def plot_family(real_name, table):
    """One figure: real graph + its 4 synthetic equivalents."""
    kinds = ["real"] + SYNTH_TYPES
    graph_names = [real_name] + [f"{real_name}_{k}" for k in SYNTH_TYPES]
    graphs = [load_graph(n) for n in graph_names]

    # shared colour scale so colours mean the same thing in every panel
    max_degree = max(max(d for _, d in G.degree()) for G in graphs)
    vmin, vmax = 0.0, np.log10(max_degree)

    fig = plt.figure(figsize=(18, 9))
    grid = fig.add_gridspec(2, 5, height_ratios=[3, 2])

    for i, (kind, name, G) in enumerate(zip(kinds, graph_names, graphs)):
        ax = fig.add_subplot(grid[0, i])
        draw_sample(ax, G, make_title(LABELS[kind], name, G, table), vmin, vmax)

    ax = fig.add_subplot(grid[1, :])
    for kind, G in zip(kinds, graphs):
        degrees, fraction = degree_ccdf(G)
        width = 2.5 if kind == "real" else 1.5
        ax.step(degrees, fraction, where="post", color=LINE_COLORS[kind],
                linewidth=width, label=LABELS[kind])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree d")
    ax.set_ylabel("fraction of nodes with degree >= d")
    ax.legend()

    fig.suptitle(f"{real_name}: real vs synthetic equivalents "
                 f"(drawings = {SAMPLE_SIZE}-node BFS sample, colour = log10 degree in full graph)",
                 fontsize=12)
    path = os.path.join(GRAPHS_DIR, f"{real_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    table_path = os.path.join(RESULTS_DIR, "topology_table.csv")
    if not os.path.exists(table_path):
        raise SystemExit("results/topology_table.csv missing. Run src/networks.py first.")
    table = pd.read_csv(table_path).set_index("name")

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    for real_name in REAL_FILES:
        plot_family(real_name, table)
