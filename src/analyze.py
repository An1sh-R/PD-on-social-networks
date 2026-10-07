"""Analyse the main sweep (run src/run.py first). Answers: how does the topology
sensitivity of BM change with beta, and which network features drive it?

1. Sensitivity   spread (std) of cooperation across all graphs, per condition
2. Steps         real - degseq   (structure beyond the degree sequence)
                 degseq - er     (degree heterogeneity)
3. Regression    cooperation ~ standardised graph features, across graphs
4. Node level    per-node cooperation vs degree and local clustering (real graphs)

BM is plotted against beta (averaged over aspiration), one line per b.
Fermi has no beta: its values are drawn as grey horizontal lines.

Run from project root:  python src/analyze.py   (or --test for the test sweep)
Writes results/<name>_*.csv and results/<name>_*.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from networks import REAL_FILES, RESULTS_DIR, load_graph

TEST = "--test" in sys.argv
NAME = "sweep_test" if TEST else "sweep"
FEATURES = ["mean_degree", "degree_cv", "clustering", "assortativity"]
FERMI_ARMS = ["fermi_average", "fermi_accumulated"]
FERMI_STYLES = {"fermi_average": ":", "fermi_accumulated": "--"}


def out_path(suffix):
    return os.path.join(RESULTS_DIR, NAME + suffix)


def load_condition_means():
    """Mean cooperation over reps, one row per (graph, agent, b, beta, a_fraction)."""
    # beta and a_fraction are "-" for Fermi, so read them as text
    runs = pd.read_csv(out_path("_runs.csv"), dtype={"beta": str, "a_fraction": str})
    keys = ["graph", "family", "graph_type", "agent", "b", "beta", "a_fraction"]
    means = runs.groupby(keys, as_index=False)["cooperation"].mean()
    return runs, means


def bm_only(table):
    """BM rows with beta and a_fraction as numbers."""
    part = table[table["agent"] == "bm"].copy()
    part["beta"] = part["beta"].astype(float)
    part["a_fraction"] = part["a_fraction"].astype(float)
    return part


def plot_vs_beta(ax, table, value, ylabel):
    """BM: value vs beta (mean over aspiration), one line per b. Fermi: grey lines."""
    bm = bm_only(table)
    for b in sorted(bm["b"].unique()):
        line = bm[bm["b"] == b].groupby("beta")[value].mean()
        ax.plot(line.index, line.values, marker="o", markersize=3, label=f"BM b={b:g}")
    for arm in FERMI_ARMS:
        fermi = table[table["agent"] == arm]
        if len(fermi) > 0:
            ax.axhline(fermi[value].mean(), color="grey", linestyle=FERMI_STYLES[arm],
                       label=arm.replace("_", " ") + " (mean over b)")
    ax.set_xscale("log")
    ax.set_xlabel("beta")
    ax.set_ylabel(ylabel)


# ---- 1. sensitivity ----------------------------------------------------------

def sensitivity(means):
    keys = ["agent", "b", "beta", "a_fraction"]
    table = means.groupby(keys, as_index=False)["cooperation"].agg(["std", "mean", "count"])
    table = table.rename(columns={"std": "sensitivity", "mean": "mean_coop", "count": "n_graphs"})
    table.to_csv(out_path("_sensitivity.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_vs_beta(ax, table, "sensitivity", "std of cooperation across graphs")
    ax.set_title("Topology sensitivity")
    ax.legend(fontsize=7)
    fig.savefig(out_path("_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # heatmap beta x a for BM, one panel per b
    bm = bm_only(table)
    temptations = sorted(bm["b"].unique())
    fig, axes = plt.subplots(1, len(temptations), figsize=(4 * len(temptations), 4), squeeze=False)
    image = None
    for ax, b in zip(axes[0], temptations):
        grid = bm[bm["b"] == b].pivot(index="a_fraction", columns="beta", values="sensitivity")
        image = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis",
                          vmin=0, vmax=bm["sensitivity"].max())
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{x:g}" for x in grid.columns], rotation=45)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{y:g}" for y in grid.index])
        ax.set_xlabel("beta")
        ax.set_title(f"b = {b:g}")
    axes[0][0].set_ylabel("aspiration a = A/b")
    fig.colorbar(image, ax=axes[0].tolist(), label="sensitivity")
    fig.savefig(out_path("_sensitivity_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# ---- 2. real -> degseq -> er steps ------------------------------------------

def steps(means):
    rows = []
    keys = ["family", "agent", "b", "beta", "a_fraction"]
    for key, group in means.groupby(keys):
        by_type = {}
        for _, row in group.iterrows():
            by_type[row["graph_type"]] = row["cooperation"]
        if "real" in by_type and "degseq" in by_type and "er" in by_type:
            row = dict(zip(keys, key))
            row["structure_effect"] = by_type["real"] - by_type["degseq"]
            row["degree_effect"] = by_type["degseq"] - by_type["er"]
            rows.append(row)
    if not rows:
        print("steps: no family has real + degseq + er, skipped")
        return None
    table = pd.DataFrame(rows)
    table.to_csv(out_path("_steps.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    plot_vs_beta(axes[0], table, "structure_effect", "cooperation difference (mean over families)")
    axes[0].set_title("real - degseq: structure beyond degree")
    plot_vs_beta(axes[1], table, "degree_effect", "")
    axes[1].set_title("degseq - ER: degree heterogeneity")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5)
    axes[1].legend(fontsize=7)
    fig.savefig(out_path("_steps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# ---- 3. regression across graphs ---------------------------------------------

def regression(means):
    topology = pd.read_csv(os.path.join(RESULTS_DIR, "topology_table.csv"))
    topology = topology.rename(columns={"name": "graph"})
    data = means.merge(topology[["graph"] + FEATURES], on="graph")
    n_graphs = data["graph"].nunique()
    if n_graphs < len(FEATURES) + 2:
        print(f"regression: only {n_graphs} graphs, need {len(FEATURES) + 2}+, skipped")
        return None

    rows = []
    for key, group in data.groupby(["agent", "b", "beta", "a_fraction"]):
        X = group[FEATURES].copy()
        for feature in FEATURES:
            # standardise so coefficients are comparable: effect of +1 std of the feature
            X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()
        X = sm.add_constant(X)
        fit = sm.OLS(group["cooperation"], X).fit()
        intervals = fit.conf_int()
        for feature in FEATURES:
            rows.append({"agent": key[0], "b": key[1], "beta": key[2], "a_fraction": key[3],
                         "feature": feature, "coef": fit.params[feature],
                         "ci_low": intervals.loc[feature, 0], "ci_high": intervals.loc[feature, 1],
                         "r_squared": fit.rsquared, "n_graphs": len(group)})
    table = pd.DataFrame(rows)
    table.to_csv(out_path("_regression.csv"), index=False)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4.2 * len(FEATURES), 4), sharey=True)
    for ax, feature in zip(axes, FEATURES):
        plot_vs_beta(ax, table[table["feature"] == feature], "coef", "")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(feature)
    axes[0].set_ylabel("effect of +1 std on cooperation")
    axes[-1].legend(fontsize=7)
    fig.savefig(out_path("_regression.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# ---- 4. node level -------------------------------------------------------------

def node_file_name(run_id):
    # must match run.py
    return run_id.replace("|", "__").replace("=", "-") + ".npy"


def node_level(runs):
    nodes_dir = out_path("_nodes")
    rows = []
    for graph in REAL_FILES:
        part = runs[(runs["graph"] == graph) & (runs["rep"] == 0)]
        if len(part) == 0:
            continue
        G = load_graph(graph)
        degrees = []
        for node in range(G.number_of_nodes()):
            degrees.append(G.degree(node))
        local_clustering_dict = nx.clustering(G)
        local_clustering = []
        for node in range(G.number_of_nodes()):
            local_clustering.append(local_clustering_dict[node])
        for _, run in part.iterrows():
            path = os.path.join(nodes_dir, node_file_name(run["run_id"]))
            if not os.path.exists(path):
                continue
            coop = np.load(path)
            # Spearman: rank correlation, robust to heavy-tailed degrees
            rows.append({"graph": graph, "agent": run["agent"], "b": run["b"],
                         "beta": run["beta"], "a_fraction": run["a_fraction"],
                         "corr_degree": stats.spearmanr(coop, degrees).statistic,
                         "corr_clustering": stats.spearmanr(coop, local_clustering).statistic})
    if not rows:
        print("node level: no node files found, skipped")
        return None
    table = pd.DataFrame(rows)
    table.to_csv(out_path("_node_level.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    plot_vs_beta(axes[0], table, "corr_degree", "Spearman correlation (mean over real graphs)")
    axes[0].set_title("node cooperation vs degree")
    plot_vs_beta(axes[1], table, "corr_clustering", "")
    axes[1].set_title("node cooperation vs local clustering")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5)
    axes[1].legend(fontsize=7)
    fig.savefig(out_path("_node_level.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


def main():
    runs, means = load_condition_means()
    print(f"{len(runs)} runs, {means['graph'].nunique()} graphs, {len(means)} condition means")
    sensitivity(means)
    steps(means)
    regression(means)
    node_level(runs)
    print("saved tables and figures in", RESULTS_DIR)


if __name__ == "__main__":
    main()
