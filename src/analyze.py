"""Analyse the main sweep (run src/run.py first). Answers: how does the topology
sensitivity of BM change with beta, and which network features drive it?

1. Sensitivity   spread (std) of cooperation across all topologies, per condition
                 ("shuffled" runs are left out: same topology as the real graph)
2. Steps         real - shuffled (fixed partners: mentor's re-seating test)
                 real - degseq   (structure beyond the degree sequence)
                 degseq - er     (degree heterogeneity)
3. Regression    cooperation ~ standardised graph features, across topologies
4. Node level    per-node cooperation vs degree and local clustering (real graphs)
0. Diagnostics   is each cell settled? (drift) Do repeats agree? (spread between reps)
                 Does it oscillate? (even_odd_gap: population flips C <-> D every round)
                 Flags known problem spots: beta = 0.01 (slow), a = 0.9 (dead zone).

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

DRIFT_LIMIT = 0.02  # |drift| above this: run was still moving at the end
SPREAD_LIMIT = 0.05  # std between repeats above this: repeats end at different levels
OSCILLATION_LIMIT = 0.05  # |even - odd round cooperation| above this: period-2 oscillation
SLOW_BETA = 0.01  # pilot: not settled by 1000 steps
DEAD_ZONE_A = 0.9  # pilot: a >= 0.9 -> everyone unhappy, cooperation ~0.5, nothing happens


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
    # dead zone left out of the average over aspiration: it would only flatten the lines
    bm = bm[bm["a_fraction"] < DEAD_ZONE_A]
    for b in sorted(bm["b"].unique()):
        line = bm[bm["b"] == b].groupby("beta")[value].mean()
        ax.plot(line.index, line.values, marker="o", markersize=3, label=f"BM b={b:g}")
    if len(bm) > 0 and bm["beta"].min() <= SLOW_BETA:
        ax.axvspan(SLOW_BETA * 0.7, SLOW_BETA * 1.4, color="orange", alpha=0.15,
                   label=f"beta={SLOW_BETA:g}: may not be settled")
    for arm in FERMI_ARMS:
        fermi = table[table["agent"] == arm]
        if len(fermi) > 0:
            ax.axhline(fermi[value].mean(), color="grey", linestyle=FERMI_STYLES[arm],
                       label=arm.replace("_", " ") + " (mean over b)")
    ax.set_xscale("log")
    ax.set_xlabel("beta")
    ax.set_ylabel(ylabel)


# ---- 0. diagnostics ----------------------------------------------------------

def diagnostics(runs):
    """Per cell (graph, b, beta, a): settled? repeats agree? oscillating?
    Uses the drift and even_odd_gap columns."""
    bm = bm_only(runs)
    if "drift" not in bm.columns:
        print("diagnostics: no drift column (sweep made before it was added), skipped")
        return None
    bm["abs_drift"] = bm["drift"].abs()
    keys = ["graph", "graph_type", "b", "beta", "a_fraction"]
    cells = bm.groupby(keys, as_index=False).agg(
        mean_abs_drift=("abs_drift", "mean"), rep_spread=("cooperation", "std"),
        cooperation=("cooperation", "mean"), n_reps=("rep", "count"))
    # older sweeps have no even_odd_gap column
    has_gap = "even_odd_gap" in bm.columns
    if has_gap:
        gaps = bm.groupby(keys, as_index=False)["even_odd_gap"].mean()
        cells["even_odd_gap"] = gaps["even_odd_gap"].values
        # separate flag, not a status: an oscillating cell can still be settled (its mean is fine)
        cells["oscillating"] = cells["even_odd_gap"] > OSCILLATION_LIMIT

    statuses = []
    for _, cell in cells.iterrows():
        if cell["mean_abs_drift"] > DRIFT_LIMIT:
            statuses.append("still moving")  # slow convergence: longer runs needed
        elif cell["rep_spread"] > SPREAD_LIMIT:
            statuses.append("settled, repeats differ")  # repeats end at different levels
        else:
            statuses.append("ok")
    cells["status"] = statuses
    cells.to_csv(out_path("_diagnostics.csv"), index=False)

    print()
    print("DIAGNOSTICS (BM cells)")
    print(cells["status"].value_counts().to_string())
    slow = cells[cells["beta"] <= SLOW_BETA]
    other = cells[cells["beta"] > SLOW_BETA]
    if len(slow) > 0 and len(other) > 0:
        print(f"still moving: beta={SLOW_BETA:g} {(slow['status'] == 'still moving').mean():.0%} "
              f"of cells, other betas {(other['status'] == 'still moving').mean():.0%}")
    if has_gap:
        oscillating = cells[cells["oscillating"]]
        print(f"oscillating (even-odd gap > {OSCILLATION_LIMIT}): {len(oscillating)} of {len(cells)} cells"
              f" ({cells['oscillating'].mean():.0%})")
        if len(oscillating) > 0:
            print(f"  at beta {oscillating['beta'].min():g}-{oscillating['beta'].max():g}, "
                  f"a {oscillating['a_fraction'].min():g}-{oscillating['a_fraction'].max():g}, "
                  f"b {oscillating['b'].min():g}-{oscillating['b'].max():g}; "
                  f"mean cooperation there {oscillating['cooperation'].mean():.3f}")
    dead = cells[cells["a_fraction"] >= DEAD_ZONE_A]
    if len(dead) > 0:
        print(f"dead zone a>={DEAD_ZONE_A:g}: mean cooperation {dead['cooperation'].mean():.3f} "
              f"(range {dead['cooperation'].min():.3f}-{dead['cooperation'].max():.3f}); "
              f"left out of beta plots")

    # heatmaps beta x a, averaged over graphs: drift, rep spread, (oscillation)
    rows = [("mean_abs_drift", "mean |drift|", 2 * DRIFT_LIMIT),
            ("rep_spread", "std between repeats", 2 * SPREAD_LIMIT)]
    if has_gap:
        rows.append(("even_odd_gap", "even-odd gap", 2 * OSCILLATION_LIMIT))
    temptations = sorted(cells["b"].unique())
    fig, axes = plt.subplots(len(rows), len(temptations),
                             figsize=(4 * len(temptations), 3.75 * len(rows)), squeeze=False)
    fig.subplots_adjust(hspace=0.5)  # room for x labels between rows
    for r, (column, label, vmax) in enumerate(rows):
        image = None
        for c, b in enumerate(temptations):
            ax = axes[r][c]
            part = cells[cells["b"] == b].groupby(["a_fraction", "beta"], as_index=False)[column].mean()
            grid = part.pivot(index="a_fraction", columns="beta", values=column)
            image = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(grid.columns)))
            ax.set_xticklabels([f"{x:g}" for x in grid.columns], rotation=45)
            ax.set_yticks(range(len(grid.index)))
            ax.set_yticklabels([f"{y:g}" for y in grid.index])
            ax.set_xlabel("beta")
            ax.set_ylabel("a = A/b")
            ax.set_title(f"{label}, b = {b:g}", fontsize=9)
        fig.colorbar(image, ax=axes[r].tolist(), shrink=0.8)
    fig.suptitle(f"Diagnostics (mean over graphs). Red = still moving (> {DRIFT_LIMIT}), "
                 f"repeats disagree (> {SPREAD_LIMIT}), oscillating (> {OSCILLATION_LIMIT})")
    fig.savefig(out_path("_diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cells


# ---- 1. sensitivity ----------------------------------------------------------

def sensitivity(means):
    keys = ["agent", "b", "beta", "a_fraction"]
    topologies = means[means["graph_type"] != "shuffled"]
    table = topologies.groupby(keys, as_index=False)["cooperation"].agg(["std", "mean", "count"])
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
        # each comparison is computed whenever its own two graphs are in the sweep
        row = dict(zip(keys, key))
        row["partner_effect"] = np.nan
        row["structure_effect"] = np.nan
        row["degree_effect"] = np.nan
        if "real" in by_type and "shuffled" in by_type:
            row["partner_effect"] = by_type["real"] - by_type["shuffled"]
        if "real" in by_type and "degseq" in by_type:
            row["structure_effect"] = by_type["real"] - by_type["degseq"]
        if "degseq" in by_type and "er" in by_type:
            row["degree_effect"] = by_type["degseq"] - by_type["er"]
        has_any = False
        for name in ["partner_effect", "structure_effect", "degree_effect"]:
            if not np.isnan(row[name]):
                has_any = True
        if has_any:
            rows.append(row)
    if not rows:
        print("steps: no comparable graph pairs in the sweep, skipped")
        return None
    table = pd.DataFrame(rows)
    table.to_csv(out_path("_steps.csv"), index=False)

    # one panel per comparison that has data (e.g. no ER in the main sweep -> no degree panel)
    panels = []
    for column, title in [("partner_effect", "real - shuffled: fixed partners"),
                          ("structure_effect", "real - degseq: structure beyond degree"),
                          ("degree_effect", "degseq - ER: degree heterogeneity")]:
        if table[column].notna().any():
            panels.append((column, title))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), sharey=True, squeeze=False)
    for ax, (column, title) in zip(axes[0], panels):
        plot_vs_beta(ax, table.dropna(subset=[column]), column, "")
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.5)
    axes[0][0].set_ylabel("cooperation difference (mean over families)")
    axes[0][-1].legend(fontsize=7)
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
    diagnostics(runs)
    sensitivity(means)
    steps(means)
    regression(means)
    node_level(runs)
    print("saved tables and figures in", RESULTS_DIR)


if __name__ == "__main__":
    main()
