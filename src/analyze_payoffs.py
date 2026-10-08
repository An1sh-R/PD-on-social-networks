"""Average vs accumulated payoffs, BM and Fermi (run src/run.py first).
Question: how much does summing payoffs (instead of averaging) change cooperation,
for each learner, and which network features predict that change?

Accumulated payoffs are scaled by the graph's mean degree (see run.py), so the
difference measures only the spread of sensitivity across degrees (hubs more,
leaves less), not a higher overall sensitivity.

0. Fermi diagnostics   absorbed (all-C / all-D) before the window? still moving?
1. Payoff effect       accumulated - average per graph, paired by seed, 95% CI over reps.
                       BM vs beta (per regime), Fermi vs 1/kappa (per mutation rate).
2. Predictors          across the 25 topologies (shuffled left out): Spearman correlation
                       of the effect with each feature, and a regression on all of them
                       (descriptive: features are correlated, n = 25).

Run from project root:  python src/analyze_payoffs.py   (or --test)
Writes results/<name>_payoff_*.csv/png and results/<name>_fermi_diagnostics.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from analyze import FEATURES, SHOWN_REGIMES, regime, t_interval, topology_features
from networks import RESULTS_DIR

TEST = "--test" in sys.argv
NAME = "sweep_v2_test" if TEST else "sweep_v2"
TYPE_COLOURS = {"real": "black", "degseq": "tab:blue", "er": "tab:green", "ba": "tab:red",
                "ws": "tab:purple", "shuffled": "tab:grey"}


def out_path(suffix):
    return os.path.join(RESULTS_DIR, NAME + suffix)


def load_runs():
    runs = pd.read_csv(out_path("_runs.csv"), dtype=str)
    for column in ["b", "cooperation", "drift", "rep", "steps", "window"]:
        runs[column] = runs[column].astype(float)
    bm = runs[runs["agent"] == "bm"].copy()
    bm["beta"] = bm["beta"].astype(float)
    bm["a_fraction"] = bm["a_fraction"].astype(float)
    regimes = []
    for _, row in bm.iterrows():
        regimes.append(regime(row["a_fraction"], row["b"]))
    bm["regime"] = regimes
    fermi = runs[runs["agent"] == "fermi"].copy()
    fermi["kappa"] = fermi["kappa"].astype(float)
    fermi["mutation"] = fermi["mutation"].astype(float)
    fermi["absorbed_at"] = fermi["absorbed_at"].astype(float)
    return bm, fermi


# ---- 0. Fermi diagnostics -----------------------------------------------------

def fermi_diagnostics(fermi):
    """Share of runs stuck in all-C / all-D before the window starts, and drift."""
    fermi = fermi.copy()
    window_start = fermi["steps"] - fermi["window"]
    fermi["absorbed_before_window"] = (fermi["absorbed_at"] >= 0) & (fermi["absorbed_at"] < window_start)
    fermi["abs_drift"] = fermi["drift"].abs()
    table = fermi.groupby(["payoff_scheme", "mutation", "kappa", "graph_type"], as_index=False).agg(
        cooperation=("cooperation", "mean"), absorbed_before_window=("absorbed_before_window", "mean"),
        mean_abs_drift=("abs_drift", "mean"), max_abs_drift=("abs_drift", "max"))
    table.to_csv(out_path("_fermi_diagnostics.csv"), index=False)
    print()
    print("FERMI DIAGNOSTICS (mean over graphs of each type and b)")
    summary = fermi.groupby(["payoff_scheme", "mutation", "kappa"]).agg(
        cooperation=("cooperation", "mean"), absorbed_before_window=("absorbed_before_window", "mean"),
        max_abs_drift=("abs_drift", "max"))
    print(summary.round(3).to_string())
    return table


# ---- 1. payoff effect -----------------------------------------------------------

def paired_effect(runs, setting_keys):
    """accumulated - average per graph and setting, paired by rep (same seed)."""
    keys = ["graph", "family", "graph_type"] + setting_keys + ["rep"]
    wide = runs.pivot_table(index=keys, columns="payoff_scheme", values="cooperation").reset_index()
    if "accumulated" not in wide.columns or "average" not in wide.columns:
        return None
    wide = wide.dropna(subset=["accumulated", "average"])  # BM accumulated grid is smaller
    wide["effect"] = wide["accumulated"] - wide["average"]
    return wide


def summarise(wide, group_keys):
    """Per graph and group: mean effect over the cells in the group, CI over reps."""
    per_rep = wide.groupby(["graph", "family", "graph_type"] + group_keys + ["rep"],
                           as_index=False)[["effect", "average", "accumulated"]].mean()
    rows = []
    for key, group in per_rep.groupby(["graph", "family", "graph_type"] + group_keys):
        mean, half = t_interval(group["effect"])
        row = dict(zip(["graph", "family", "graph_type"] + group_keys, key))
        row.update({"effect": mean, "ci": half, "average": group["average"].mean(),
                    "accumulated": group["accumulated"].mean(), "n_reps": len(group)})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_effect(ax, table, x_column, x_label):
    """One thin line per graph, coloured by graph type."""
    labelled = set()
    for graph, line in table.groupby("graph"):
        kind = line["graph_type"].iloc[0]
        line = line.sort_values(x_column)
        label = None
        if kind not in labelled:
            label = kind
            labelled.add(kind)
        ax.plot(line[x_column], line["effect"], color=TYPE_COLOURS.get(kind, "orange"),
                linewidth=1.5 if kind == "real" else 0.8, marker="o", markersize=2, label=label)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(x_label)


def payoff_effect(bm, fermi):
    tables = {}
    bm_wide = paired_effect(bm, ["b", "beta", "a_fraction", "regime"])
    if bm_wide is not None:
        tables["bm"] = summarise(bm_wide, ["regime", "beta"])
    fermi_wide = paired_effect(fermi, ["b", "kappa", "mutation"])
    if fermi_wide is not None:
        table = summarise(fermi_wide, ["mutation", "kappa"])
        table["selection"] = 1 / table["kappa"]
        tables["fermi"] = table
    for learner, table in tables.items():
        table.to_csv(out_path(f"_payoff_effect_{learner}.csv"), index=False)

    panels = []
    if "bm" in tables:
        for shown in SHOWN_REGIMES:
            part = tables["bm"][tables["bm"]["regime"] == shown]
            panels.append((part, "beta", "beta", f"BM, regime {shown} (mean over b, a)"))
    if "fermi" in tables:
        for mutation in sorted(tables["fermi"]["mutation"].unique()):
            part = tables["fermi"][tables["fermi"]["mutation"] == mutation]
            panels.append((part, "selection", "1 / kappa", f"Fermi, mutation {mutation:g} (mean over b)"))
    if not panels:
        print("payoff effect: no graph has both payoff schemes, skipped")
        return tables
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.3), squeeze=False)
    for ax, (part, x_column, x_label, title) in zip(axes[0], panels):
        plot_effect(ax, part, x_column, x_label)
        ax.set_title(title, fontsize=9)
    axes[0][0].set_ylabel("cooperation: accumulated - average")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Effect of summing payoffs (scaled by mean degree), one line per graph")
    fig.savefig(out_path("_payoff_effect.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tables


# ---- 2. predictors -------------------------------------------------------------

def predictors(tables):
    """Across topologies: which features go with a bigger payoff effect?"""
    features = topology_features()
    rows = []
    for learner, table in tables.items():
        topologies = table[table["graph_type"] != "shuffled"].merge(features, on="graph")
        if learner == "bm":
            group_keys = ["regime", "beta"]
        else:
            group_keys = ["mutation", "kappa"]
        for key, group in topologies.groupby(group_keys):
            n_graphs = len(group)
            if n_graphs < len(FEATURES) + 3:
                continue
            base = {"learner": learner, "setting": f"{group_keys[0]}={key[0]}",
                    "x": key[1], "n_graphs": n_graphs}
            for feature in FEATURES:
                rho = stats.spearmanr(group[feature], group["effect"]).statistic
                row = dict(base)
                row.update({"method": "spearman", "feature": feature, "value": rho,
                            "ci_low": np.nan, "ci_high": np.nan})
                rows.append(row)
            X = group[FEATURES].copy()
            for feature in FEATURES:
                X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()
            fit = sm.OLS(group["effect"], sm.add_constant(X)).fit()
            intervals = fit.conf_int()
            for feature in FEATURES:
                row = dict(base)
                row.update({"method": "regression", "feature": feature, "value": fit.params[feature],
                            "ci_low": intervals.loc[feature, 0], "ci_high": intervals.loc[feature, 1],
                            "r_squared": fit.rsquared})
                rows.append(row)
    if not rows:
        print("predictors: too few graphs, skipped")
        return None
    table = pd.DataFrame(rows)
    table.to_csv(out_path("_payoff_predictors.csv"), index=False)

    settings = table[["learner", "setting"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(2, len(settings), figsize=(4.5 * len(settings), 7.5), squeeze=False)
    for c, (learner, setting) in enumerate(settings):
        for r, method in enumerate(["spearman", "regression"]):
            ax = axes[r][c]
            part = table[(table["learner"] == learner) & (table["setting"] == setting)
                         & (table["method"] == method)]
            for feature in FEATURES:
                line = part[part["feature"] == feature].sort_values("x")
                x = line["x"]
                if learner == "fermi":
                    x = 1 / x
                ax.plot(x, line["value"], marker="o", markersize=3, label=feature)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("beta" if learner == "bm" else "1 / kappa")
            if method == "spearman":
                ax.set_ylabel("Spearman rho with payoff effect")
            else:
                ax.set_ylabel("coef (+1 std of feature)")
            ax.set_title(f"{learner}, {setting}: {method}", fontsize=9)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"What predicts accumulated - average? ({table['n_graphs'].max()} topologies)")
    fig.savefig(out_path("_payoff_predictors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


def main():
    bm, fermi = load_runs()
    print(f"{len(bm)} BM runs, {len(fermi)} Fermi runs")
    if len(fermi) > 0:
        fermi_diagnostics(fermi)
    tables = payoff_effect(bm, fermi)
    predictors(tables)
    print("saved tables and figures in", RESULTS_DIR)


if __name__ == "__main__":
    main()
