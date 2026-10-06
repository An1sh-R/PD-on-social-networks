# Design decisions

Companion to `question.md`. Decided 2026-10-06.

## Scope

- Static networks. No rewiring, no coevolution. Synthetic graphs generated once, then fixed.
- Game: **weak PD**. R=1, T=b, S=P=0. b = temptation.
- Simulation: mesa 3.5.1 + networkx. Code rules in `../CLAUDE.md`.
- Python 3.12+ required (mesa ≥3.4). Default `python` here = 3.11 → use `py -3.14 -m venv .venv`. Versions pinned in `requirements.txt`.

## Data

Location: `data/` in project. Edge lists only, flat. Synthetic graphs cached in `data/synthetic/`. Spare Twitch regions (DE, ES, FR, PTBR, RU) stay in `internship/snaps/`; raw zips too.

| File | Source | N | E | ⟨k⟩ | Deg CV | Clust | Assort |
|---|---|---|---|---|---|---|---|
| facebook_combined.txt | SNAP ego-Facebook | 4,039 | 88k | 43.7 | 1.2 | 0.61 | +0.06 |
| deezer_edges.csv | Deezer Europe | 28,281 | 93k | 6.6 | 1.2 | 0.14 | +0.10 |
| github_edges.csv | GitHub (musae) | 37,700 | 289k | 15.3 | 5.3 | 0.17 | −0.08 |
| lastfm_edges.csv | LastFM Asia | 7,624 | 28k | 7.3 | 1.6 | 0.22 | +0.02 |
| twitch_ENGB_edges.csv | Twitch ENGB | 7,126 | 35k | 9.9 | 2.2 | 0.13 | −0.12 |

- Main sweep uses these 5. Other Twitch regions (DE, ES, FR, PTBR, RU) in `snaps/`, add later if compute allows. ⟨k⟩ 17–34, assort −0.12 to −0.23.
- All 5 already single connected component. Still drop self-loops on load, check.
- Facebook: use SNAP `facebook_combined.txt`, not the ego tarball. Rebuilding from tarball gave 3,963 nodes ≠ 4,039.
- Formats: CSVs have 1 header row, comma. Facebook = space, no header.
- Caution: clustering range narrow (0.13–0.22 except Facebook). Hard to separate clustering effect.

## Synthetic equivalents

Per real graph. Same N, largest component only.

| Type | Match | Isolates |
|---|---|---|
| ER | p = ⟨k⟩/(N−1) | random baseline |
| BA | m = round(⟨k⟩/2) | heavy tail, low clustering |
| WS | k = round(⟨k⟩); grid over rewiring p, pick clustering closest to real | clustering + short paths, no heavy tail |
| Degseq | copy of real graph, `nx.double_edge_swap` (10×E swaps). Exact real degrees | degree sequence alone |

- 5 real + 20 synthetic = 25 graphs. Built by `networks.py`, cached in `data/synthetic/`, fixed seed 42.
- Catch: ER/BA/WS differ from real in many features at once. Gap real vs synthetic can't name the cause alone.
- Fix: degseq. real vs degseq = clustering/mixing/communities. degseq vs ER = degree heterogeneity.
- Degseq replaced `nx.configuration_model`: it collapses hub multi-edges, GitHub kmax 9458 → 5923, CV 5.27 → 4.09. Swaps keep degrees exact. One-time generation, networks stay static during simulation.
- Degseq/ER/BA/WS take largest component → lose 0–1% nodes (deezer, lastfm most).
- BA ⟨k⟩ off (m integer): deezer 6.0 vs 6.56, lastfm 8.0 vs 7.29, github 16.0 vs 15.3. WS ⟨k⟩ rounded to even. Use actual ⟨k⟩ from table, not target.
- Degseq kills clustering mostly but not fully on hub graphs: GitHub 0.168 → 0.092 (hubs make triangles by degree alone).
- Output: `results/topology_table.csv` (N, E, ⟨k⟩, kmax, CV, clustering, assortativity, Louvain modularity).
- Drawings: `src/plot_graphs.py` → `graphs/<name>.png`, one per real graph. Top: 300-node BFS sample of real + 4 synthetic (colour/size = degree in full graph). Bottom: full-graph degree CCDF, all 5 overlaid. Samples only look like local neighbourhoods, not whole graph. Degseq line hides under real line (identical degrees, by design).

## Agents

Both: play weak PD with all neighbours each round. Synchronous. Mesa 3 has no `SimultaneousActivation` → plain 2-phase loop (play all, then update all).

### Fermi

- Pick 1 random neighbour j. Copy with prob `expit((π_j − π_i)/κ)`. κ = 0.1.
- `math.exp` overflows on accumulated payoffs (Facebook, Δ/κ up to ~15,000). Use `scipy.special.expit`.
- Payoff scheme flag: accumulated (classic) or averaged.

### Bush–Mosteller (Ezaki et al. 2016)

- State: p = cooperation propensity.
- Action: C with prob (1−ε)p + ε/2. ε = 0.2 default (Ezaki).
- Payoff r = **average** over neighbours, ∈ [0, b].
- Stimulus s = tanh(β(r − A)).
- Update:

| Action | s ≥ 0 | s < 0 |
|---|---|---|
| C | p + (1−p)s | p + p·s |
| D | p − p·s | p − (1−p)s |

- p stays in [0,1], no clipping needed.
- **BM averaged payoff only.** A needs fixed scale; accumulated payoff scales with degree.
- Comparison arms: BM-averaged, Fermi-averaged (fair), Fermi-accumulated (classic).
- A: in [0, b]. Start 0.2 / 0.5 / 0.8 (TBD after pilot).

## Measuring topology effect

Three layers.

1. **Sensitivity.** Spread (std, range) of final cooperation across 25 graphs, per (agent, β, b). Headline plot: sensitivity vs β, BM vs Fermi.
2. **Attribution across graphs.** Per condition, regress cooperation on 3–4 standardised features: ⟨k⟩, degree CV, clustering, assortativity (maybe modularity). n=25, correlated features → effect sizes + CIs, not p-values. Compare BM vs Fermi coefficients. Plus controlled steps: real vs degseq vs ER.
3. **Node level.** Each node's late-window cooperation vs degree, local clustering, k-core. Thousands of points per graph. Shows hub/cluster effects, BM vs Fermi.

## Outcome measure

- Mean cooperation over late window. BM: fraction of C actions (also log mean p).
- Window start from plateau check on pilot runs, not fixed step.

## Parameters (provisional)

| Param | Values | Note |
|---|---|---|
| b | 1.1–2.0 | weak PD temptation |
| β | TBD, wide log-ish range | key sweep |
| A | 0.2, 0.5, 0.8 | BM only |
| ε | 0.2 | BM, fixed |
| κ | 0.1 | Fermi, fixed |
| init | 50% C | Fermi; BM p random or 0.5 TBD |
| steps, reps | TBD | from pilot |

## Compute

- GitHub: ~578k neighbour visits/step. Pure-Python mesa maybe 1–2 min per 200-step run.
- Timing pilot before any sweep. Fallbacks: fewer reps on big graphs, simple adjacency-list payoff function (no clever code).

## Still open

- β range. Pilot.
- Steps, reps, plateau window. Pilot.
- Replication check for BM: Ezaki & Masuda 2017 rings (get params from paper). Fermi: sanity on lattice.
- Add other Twitch regions? After timing pilot.
- Weak-PD note: weak-selection theory gives no cooperation advantage in weak PD for b>1 → all signal at larger β.
