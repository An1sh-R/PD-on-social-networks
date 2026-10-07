"""Pilot scan for BM on one network (LastFM): coarse beta x A x b grid.

Answers three questions before the big sweep:
1. Where does cooperation change? (so the dense grid covers the interesting part)
2. How many steps until cooperation plateaus? (run length, averaging window)
3. How much do repeat runs differ? (number of replicates)

Run from project root: python src/pilot.py
Writes results/pilot_curves.csv (cooperation every 10 steps, every run)
and results/pilot_*.png
"""
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import CooperationModel
from networks import RESULTS_DIR, load_graph

NETWORK = "lastfm"
BETAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
ASPIRATIONS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TEMPTATIONS = [1.1, 1.5, 1.9]
REPS = 3
STEPS = 1000
SAVE_EVERY = 10
FINAL_WINDOW = 200  # last 200 steps = "final" cooperation
PLATEAU_TOLERANCE = 0.02
N_PROCESSES = 8
EPSILON = 0.05  # first pilot used 0.2 (model default); its files keep the _eps0.2 tag

TAG = f"eps{EPSILON}"
CURVES_PATH = os.path.join(RESULTS_DIR, f"pilot_curves_{TAG}.csv")


def run_one(settings):
    """One BM run. Returns rows of (settings, step, cooperation) every SAVE_EVERY steps."""
    beta, aspiration, b, rep = settings
    G = load_graph(NETWORK)
    model = CooperationModel(G, "bm", b=b, beta=beta, aspiration=aspiration,
                             epsilon=EPSILON, seed=rep)
    model.run(STEPS)
    curve = model.datacollector.model_vars["cooperation"]
    rows = []
    for step in range(0, STEPS, SAVE_EVERY):
        rows.append({"beta": beta, "aspiration": aspiration, "b": b, "rep": rep,
                     "step": step, "cooperation": curve[step]})
    return rows


def run_all():
    jobs = []
    for beta in BETAS:
        for aspiration in ASPIRATIONS:
            for b in TEMPTATIONS:
                for rep in range(REPS):
                    jobs.append((beta, aspiration, b, rep))
    print(f"{len(jobs)} runs of {STEPS} steps on {NETWORK}")

    all_rows = []
    with Pool(N_PROCESSES) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, jobs)):
            all_rows.extend(rows)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} done")
    curves = pd.DataFrame(all_rows)
    curves.to_csv(CURVES_PATH, index=False)
    return curves


def settle_step(steps, values, final):
    """First saved step after which cooperation stays within tolerance of its final value."""
    settled_at = steps[-1]
    for i in range(len(values) - 1, -1, -1):
        if abs(values[i] - final) > PLATEAU_TOLERANCE:
            break
        settled_at = steps[i]
    return settled_at


def summarise(curves):
    """One row per run: final cooperation and when it settled."""
    rows = []
    groups = curves.groupby(["beta", "aspiration", "b", "rep"])
    for (beta, aspiration, b, rep), run in groups:
        run = run.sort_values("step")
        steps = list(run["step"])
        values = list(run["cooperation"])
        final_part = run[run["step"] >= STEPS - FINAL_WINDOW]["cooperation"]
        final = final_part.mean()
        rows.append({"beta": beta, "aspiration": aspiration, "b": b, "rep": rep,
                     "final": final, "settle_step": settle_step(steps, values, final)})
    return pd.DataFrame(rows)


def heatmap(ax, table, value_column, title, vmin, vmax, cmap):
    grid = table.pivot(index="aspiration", columns="beta", values=value_column)
    image = ax.imshow(grid.values, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{x:g}" for x in grid.columns], rotation=45)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{y:g}" for y in grid.index])
    ax.set_xlabel("beta")
    ax.set_ylabel("aspiration A")
    ax.set_title(title, fontsize=10)
    return image


def plot_summary(summary):
    # average over reps, also spread between reps
    mean_table = summary.groupby(["beta", "aspiration", "b"], as_index=False).agg(
        final=("final", "mean"), spread=("final", "std"), settle=("settle_step", "max"))

    rows = [("final", "final cooperation (mean of reps)", 0, 1, "viridis"),
            ("settle", "steps to plateau (max of reps)", 0, STEPS, "magma"),
            ("spread", "std of final coop between reps", 0, 0.05, "Reds")]
    fig, axes = plt.subplots(len(rows), len(TEMPTATIONS), figsize=(15, 13))
    for r, (column, label, vmin, vmax, cmap) in enumerate(rows):
        for c, b in enumerate(TEMPTATIONS):
            part = mean_table[mean_table["b"] == b]
            image = heatmap(axes[r, c], part, column, f"{label}\nb = {b}", vmin, vmax, cmap)
        fig.colorbar(image, ax=axes[r, :].tolist(), shrink=0.8)
    fig.suptitle(f"BM pilot on {NETWORK}, eps={EPSILON}: {REPS} reps x {STEPS} steps", fontsize=13)
    fig.savefig(os.path.join(RESULTS_DIR, f"pilot_heatmaps_{TAG}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return mean_table


def plot_example_curves(curves):
    """Cooperation over time for A=0.5, b=1.5, every beta (rep 0)."""
    part = curves[(curves["aspiration"] == 0.5) & (curves["b"] == 1.5) & (curves["rep"] == 0)]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(BETAS)))
    for beta, color in zip(BETAS, colors):
        run = part[part["beta"] == beta].sort_values("step")
        ax.plot(run["step"], run["cooperation"], color=color, label=f"beta={beta:g}")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction cooperating")
    ax.set_title(f"BM on {NETWORK}, A=0.5, b=1.5")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(os.path.join(RESULTS_DIR, f"pilot_curves_{TAG}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(CURVES_PATH):
        print("using saved", CURVES_PATH, "(delete it to rerun)")
        curves = pd.read_csv(CURVES_PATH)
    else:
        curves = run_all()
    summary = summarise(curves)
    mean_table = plot_summary(summary)
    plot_example_curves(curves)

    print("\nslowest to settle (steps):")
    print(mean_table.sort_values("settle", ascending=False).head(8).round(3).to_string(index=False))
    print("\nlargest spread between reps:")
    print(mean_table.sort_values("spread", ascending=False).head(5).round(3).to_string(index=False))
    print(f"\nfinal cooperation range: {mean_table['final'].min():.3f} - {mean_table['final'].max():.3f}")
