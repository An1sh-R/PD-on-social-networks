"""Average vs accumulated payoffs, BM and Fermi (run src/run.py first).
Question: how much does summing payoffs (instead of averaging) change cooperation,
for each learner, and which network features predict that change?
Hypotheses written down before the run: ResearchQ/question.md (H6-H8).

Accumulated payoffs are scaled by the graph's mean degree (see run.py), so the
difference measures only the spread of sensitivity across degrees (hubs more,
leaves less), not a higher overall sensitivity. For BM this is exact: accumulated
BM = average BM with beta_i = beta * k_i / <k>. So the effect is 0 on a regular
graph and grows with degree spread BY CONSTRUCTION: "degree CV predicts it" is
built in, not a finding. The informative test is 2 (real vs degseq: same degrees).

0. Fermi diagnostics   drift, and how many cells sit at the mutation floor (cooperation
                       < 0.05: mutants weeded out every round, nothing to compare).
1. Payoff effect       accumulated - average per graph, paired by seed. BM vs beta (per
                       regime), Fermi vs 1/kappa. 95% CI over reps (degseq: over its 5
                       rewirings; ER/BA/WS: one graph, CI = run noise only).
2. Structure test      payoff effect on real minus on degseq, per family: does anything
                       beyond the degree sequence change the payoff effect?
3. Predictors          across the 25 topologies (shuffled left out), within families:
                       regression on degree CV, clustering, assortativity + one dummy per
                       family; Spearman on family-centred values. Descriptive.

Run from project root:  python src/analyze_payoffs.py   (or --test)
Writes results/<name>_payoff_*.csv/png and results/<name>_fermi_diagnostics.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from analyze import (REGRESSION_FEATURES, SHOWN_REGIMES, fit_within_family, independent_unit,
                     regime, t_interval, topology_features)
from networks import RESULTS_DIR

TEST = "--test" in sys.argv
NAME = "sweep_v2_test" if TEST else "sweep_v2"
FLOOR = 0.05  # Fermi cooperation below this: at the mutation floor (mutation 0.01)
TYPE_COLOURS = {"real": "black", "degseq": "tab:blue", "er": "tab:green", "ba": "tab:red",
                "ws": "tab:purple", "shuffled": "tab:grey"}


def out_path(suffix):
    return os.path.join(RESULTS_DIR, NAME + suffix)


def load_runs():
    runs = pd.read_csv(out_path("_runs.csv"), dtype=str)
    for column in ["b", "cooperation", "drift", "rep", "steps", "window"]:
        runs[column] = runs[column].astype(float)
    runs["rep"] = runs["rep"].astype(int)
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
    return bm, fermi


# ---- 0. Fermi diagnostics -----------------------------------------------------

def fermi_diagnostics(fermi):
    """Drift, and share of cells (graph, b, kappa) at the mutation floor.
    (absorbed_at is not used: with mutation 0.01 on thousands of players an all-C or
    all-D round never happens.)"""
    fermi = fermi.copy()
    fermi["abs_drift"] = fermi["drift"].abs()
    cells = fermi.groupby(["payoff_scheme", "kappa", "b", "graph", "graph_type"], as_index=False).agg(
        cooperation=("cooperation", "mean"), mean_abs_drift=("abs_drift", "mean"))
    cells["at_floor"] = cells["cooperation"] < FLOOR
    cells.to_csv(out_path("_fermi_diagnostics.csv"), index=False)
    print()
    print(f"FERMI DIAGNOSTICS (cells = graph x b; at floor = cooperation < {FLOOR})")
    summary = cells.groupby(["payoff_scheme", "kappa"]).agg(
        cooperation=("cooperation", "mean"), share_at_floor=("at_floor", "mean"),
        max_abs_drift=("mean_abs_drift", "max"))
    print(summary.round(3).to_string())
    return cells


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
    """Per graph and group: mean effect over the cells in the group, CI over
    independent units (reps; rewirings for degseq)."""
    per_rep = wide.groupby(["graph", "family", "graph_type"] + group_keys + ["rep"],
                           as_index=False)[["effect", "average", "accumulated"]].mean()
    units = []
    for _, row in per_rep.iterrows():
        units.append(independent_unit([row["graph_type"]], row["rep"]))
    per_rep["unit"] = units
    per_unit = per_rep.groupby(["graph", "family", "graph_type"] + group_keys + ["unit"],
                               as_index=False)[["effect", "average", "accumulated"]].mean()
    rows = []
    for key, group in per_unit.groupby(["graph", "family", "graph_type"] + group_keys):
        mean, half = t_interval(group["effect"])
        row = dict(zip(["graph", "family", "graph_type"] + group_keys, key))
        row.update({"effect": mean, "ci": half, "average": group["average"].mean(),
                    "accumulated": group["accumulated"].mean(), "n_units": len(group)})
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
    """Returns per-graph summary tables and the paired per-rep tables (for 2.)."""
    tables = {}
    wides = {}
    bm_wide = paired_effect(bm, ["b", "beta", "a_fraction", "regime"])
    if bm_wide is not None:
        wides["bm"] = bm_wide
        tables["bm"] = summarise(bm_wide, ["regime", "beta"])
    fermi_wide = paired_effect(fermi, ["b", "kappa"])
    if fermi_wide is not None:
        wides["fermi"] = fermi_wide
        table = summarise(fermi_wide, ["kappa"])
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
        panels.append((tables["fermi"], "selection", "1 / kappa", "Fermi (mean over b)"))
    if not panels:
        print("payoff effect: no graph has both payoff schemes, skipped")
        return tables, wides
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.3), squeeze=False)
    for ax, (part, x_column, x_label, title) in zip(axes[0], panels):
        plot_effect(ax, part, x_column, x_label)
        ax.set_title(title, fontsize=9)
    axes[0][0].set_ylabel("cooperation: accumulated - average")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Effect of summing payoffs (scaled by mean degree), one line per graph")
    fig.savefig(out_path("_payoff_effect.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tables, wides


# ---- 2. structure test: real vs degseq ---------------------------------------------

def structure_test(wides):
    """Payoff effect on real minus on degseq (same degrees), per family, paired by rep,
    CI over the 5 rewirings. The one comparison where degree spread cannot explain
    the result."""
    rows = []
    for learner, wide in wides.items():
        if learner == "bm":
            cell_keys = ["regime", "b", "beta", "a_fraction"]
            group_keys = ["regime", "beta"]
        else:
            cell_keys = ["b", "kappa"]
            group_keys = ["kappa"]
        pairs = wide[wide["graph_type"].isin(["real", "degseq"])].pivot_table(
            index=["family"] + cell_keys + ["rep"], columns="graph_type", values="effect").reset_index()
        if "real" not in pairs.columns or "degseq" not in pairs.columns:
            continue
        pairs["difference"] = pairs["real"] - pairs["degseq"]
        per_rep = pairs.groupby(["family"] + group_keys + ["rep"], as_index=False)["difference"].mean()
        units = []
        for rep in per_rep["rep"]:
            units.append(independent_unit(["degseq"], rep))
        per_rep["unit"] = units
        per_unit = per_rep.groupby(["family"] + group_keys + ["unit"], as_index=False)["difference"].mean()
        for key, group in per_unit.groupby(["family"] + group_keys):
            mean, half = t_interval(group["difference"])
            row = {"learner": learner}
            row.update(dict(zip(["family"] + group_keys, key)))
            row.update({"real_minus_degseq": mean, "ci": half, "n_units": len(group)})
            rows.append(row)
    if not rows:
        print("structure test: no real/degseq pairs, skipped")
        return None
    table = pd.DataFrame(rows)
    table.to_csv(out_path("_payoff_structure_test.csv"), index=False)

    panels = []
    if "bm" in set(table["learner"]):
        for shown in SHOWN_REGIMES:
            part = table[(table["learner"] == "bm") & (table["regime"] == shown)]
            panels.append((part, "beta", f"BM, regime {shown}"))
    if "fermi" in set(table["learner"]):
        part = table[table["learner"] == "fermi"].copy()
        part["selection"] = 1 / part["kappa"]
        panels.append((part, "selection", "Fermi"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.3), squeeze=False)
    for ax, (part, x_column, title) in zip(axes[0], panels):
        for family in sorted(part["family"].unique()):
            line = part[part["family"] == family].sort_values(x_column)
            ax.errorbar(line[x_column], line["real_minus_degseq"], yerr=line["ci"], marker="o",
                        markersize=3, capsize=2, label=family)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("beta" if x_column == "beta" else "1 / kappa")
        ax.set_title(title, fontsize=9)
    axes[0][0].set_ylabel("payoff effect: real - degseq")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Does structure beyond degree change the payoff effect? 95% CI over 5 rewirings")
    fig.savefig(out_path("_payoff_structure_test.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


# ---- 3. predictors -------------------------------------------------------------

def predictors(tables):
    """Across topologies, within families: which features go with a bigger payoff effect?"""
    features = topology_features()
    rows = []
    for learner, table in tables.items():
        topologies = table[table["graph_type"] != "shuffled"].merge(features, on="graph")
        if learner == "bm":
            group_keys = ["regime", "beta"]
        else:
            group_keys = ["kappa"]
        for key, group in topologies.groupby(group_keys):
            if not isinstance(key, tuple):
                key = (key,)
            n_graphs = len(group)
            n_families = group["family"].nunique()
            if n_graphs < len(REGRESSION_FEATURES) + n_families + 2:
                continue
            base = {"learner": learner, "setting": ", ".join(f"{k}={v}" for k, v in zip(group_keys[:-1], key[:-1])),
                    "x": key[-1], "n_graphs": n_graphs}
            # Spearman on family-centred values: compares graphs within a family only
            centred = group.copy()
            for column in REGRESSION_FEATURES + ["effect"]:
                centred[column] = group[column] - group.groupby("family")[column].transform("mean")
            for feature in REGRESSION_FEATURES:
                row = dict(base)
                row.update({"method": "spearman", "feature": feature,
                            "value": stats.spearmanr(centred[feature], centred["effect"]).statistic})
                rows.append(row)
            result, r_squared = fit_within_family(group, "effect")
            for feature in REGRESSION_FEATURES:
                coef, low, high = result[feature]
                row = dict(base)
                row.update({"method": "regression", "feature": feature, "value": coef,
                            "ci_low": low, "ci_high": high, "r_squared": r_squared})
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
            for feature in REGRESSION_FEATURES:
                line = part[part["feature"] == feature].sort_values("x")
                x = line["x"]
                if learner == "fermi":
                    x = 1 / x
                ax.plot(x, line["value"], marker="o", markersize=3, label=feature)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_xlabel("beta" if learner == "bm" else "1 / kappa")
            if method == "spearman":
                ax.set_ylabel("Spearman rho, within family")
            else:
                ax.set_ylabel("coef (+1 std), family dummies")
            ax.set_title(f"{learner} {setting}: {method}", fontsize=9)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"What predicts accumulated - average, within families? ({table['n_graphs'].max()} "
                 f"topologies). Degree CV is built in for BM (see docstring)")
    fig.savefig(out_path("_payoff_predictors.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table


def main():
    bm, fermi = load_runs()
    print(f"{len(bm)} BM runs, {len(fermi)} Fermi runs")
    if len(fermi) > 0:
        fermi_diagnostics(fermi)
    tables, wides = payoff_effect(bm, fermi)
    structure_test(wides)
    predictors(tables)
    print("saved tables and figures in", RESULTS_DIR)


if __name__ == "__main__":
    main()
