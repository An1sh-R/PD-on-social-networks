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
- **Fixed 2026-10-07 (code review):** degseq/rewired graphs used to go through `clean()` (largest component only). Swaps cut off a few tiny pieces (mostly two degree-1 nodes linked together), so nodes were dropped: LastFM 48–91 per rewiring, Twitch 12–26, Deezer 189, GitHub 40 → degrees not exact. Now every node kept (a few tiny separate pieces, no isolated nodes), `make_degseq` asserts the degree sequence equals the real one. All degseq/rewired files rebuilt.
- Graph files written to `.tmp` then renamed: stopping during a build leaves no half-written graph.
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

### Implementation notes (`src/model.py`)

- Classes: `FermiAgent`, `BMAgent` (both subclass `Player`), `CooperationModel`. Checks: `python src/test_model.py` (13 checks).
- Update phase reads only frozen actions + payoffs, so in-place sequential updates = synchronous. No `next_strategy` buffer.
- Fermi copies neighbour's *action* (= its strategy). BM initial p = `initial_coop` (0.5) for every node (as Ezaki et al. 2016). Note: mentor's `structure_test.py` and `validate_bm.py` use 0.8 (as Ezaki & Masuda 2017).
- Cooperation recorded = fraction of C **actions played** that round, for both agents (comparable).
- `record_from` = round index (0-based) from which per-node cooperation is counted. Own counter, because mesa's `steps` is already +1 inside `step()`.
- Mesa 3.5: use `rng=` not `seed=` (deprecated). `model.random` still stdlib Random seeded from it.
- First smoke run, b=1.2, LastFM: Fermi-averaged → cooperation collapses to 0.00; Fermi-accumulated → 0.67; BM 0.45. Matches the known hub effect of accumulated payoffs, so payoff scheme is a real confound, keep it crossed.

### BM validation (`src/validate_bm.py`) — PASSED (qualitative)

Ezaki & Masuda 2017 setup: ring N=100, donation game c=1, β=0.2, A=1.0, ε=0.05, p0=0.8, 50 rounds, avg rounds 1–25, 1000 runs. Model has `game="donation"` option for this only.

Re-run 2026-10-07 with the corrected ε (flip). Old ε rule (random pick) in brackets.

| k | static − shuffled, b/c=2 | static − shuffled, b/c=6 |
|---|---|---|
| 2 | +0.007 (+0.008) | **+0.084** (+0.127) |
| 4 | +0.003 (+0.003) | +0.042 (+0.062) |
| 6 | +0.002 (+0.002) | +0.020 (+0.015) |
| 8 | +0.001 (+0.002) | +0.015 (+0.005) |

- All 3 paper claims hold under both ε rules: reciprocity at b/c=6 k=2; none at b/c=2; gap shrinks with k.
- Paper gives no numbers in text → shape match only, not exact values. Figure: `results/validate_bm.png`.
- Note for main study: BM *is* structure-sensitive here, at β=0.2 (not tiny). Fits H2.

### Bush–Mosteller (Ezaki et al. 2016)

- State: p = cooperation propensity.
- Action: C with prob (1−ε)p + ε(1−p) = intended action flipped with prob ε (Ezaki & Masuda 2017: "submit the action opposite to the intention … with probability ε"). Main sweep ε = 0.05.
- **Fixed 2026-10-07 (code review):** code had (1−ε)p + ε/2 (random pick with prob ε = flip with prob ε/2), so "ε=0.05" was really a 0.025 flip rate. Both pilots (`pilot_*_eps0.2.*`, `pilot_*_eps0.05.*`) and the first `validate_bm` ran with the old rule; main sweep uses the new one.
- Payoff r = **average** over neighbours, ∈ [0, b].
- Stimulus s = tanh(β(r − A)).
- Update:

| Action | s ≥ 0 | s < 0 |
|---|---|---|
| C | p + (1−p)s | p + p·s |
| D | p − p·s | p − (1−p)s |

- p stays in [0,1], no clipping needed.
- ~~BM averaged payoff only.~~ Main sweep: BM averaged. **BM accumulated added 2026-10-08** (see "Payoff scheme comparison"): aspiration of seat = A·k.
- Comparison arms: BM-averaged, BM-accumulated, Fermi-averaged (fair), Fermi-accumulated (classic).
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

**Main focus = BM.** Main run: large, dense β × A grid for BM. Fermi = baseline only, coarser grid.

| Param | Values | Note |
|---|---|---|
| b | 1.1–2.0 | weak PD temptation |
| β | large dense grid, log-spaced (e.g. 0.01 → 100) | key sweep, BM |
| A | large dense grid across payoff range [0, b] | key sweep, BM. Avg payoff in weak PD ∈ [0, b] |
| ε | ~~0.2~~ 0.05 (flip) | BM, fixed |
| κ | 0.1 | Fermi, fixed |
| init | 50% C (Fermi), p = 0.5 (BM) | |
| steps, reps | TBD | from pilot |

Grid size set after pilot (cost ≈ graphs × β × A × b × reps).

## Pilot results (`src/pilot.py`, LastFM, BM, 9 β × 9 A × 3 b × 3 reps × 1000 steps)

Figures: `results/pilot_heatmaps.png`, `results/pilot_curves.png`. Data: `results/pilot_curves.csv`.

- Final cooperation 0.19–0.69. ε=0.2 caps it to [0.1, 0.9] by construction.
- β: flat for β ≤ 0.3 (weak-learning regime, all alike), transition β ≈ 1–10, saturates β ≥ 30.
- A: dead zone A ≳ 0.8b → everyone always unhappy → coop ≈ 0.5, nothing happens. Interesting A ∈ [0, ~0.8b]. Interesting band shifts with b → parametrise A as fraction of b.
- Plateau: most runs flat by 300–600 steps (β = 0.01 slowest, ~600).
- Ridge: β ≥ 10, A ≈ 0.65b. Reps diverge (e.g. 0.41 vs 0.54) and drift for hundreds of steps. Std between reps up to 0.08; elsewhere ≤ 0.01. **Probably the same sampling artifact as below** (every-10th-step curves + period-2 oscillation); not re-checked at ε=0.2.
- "Steps to plateau" heatmap overstates (tolerance 0.02 vs noise ~0.01–0.02). Block means used instead.

## Shuffled seats in main sweep (decided 2026-10-07)

- Mentor's idea folded into `run.py`: graph type `shuffled` = real graph, players re-seated every round (no fixed partners). 30 graphs total: 5 real + 20 synthetic + 5 shuffled. All arms (BM + both Fermi).
- Wide β × A grid kept. Runs 1500 steps, final = mean of last 500.
- analyze.py: shuffled excluded from sensitivity + regression (not a new topology); new step real − shuffled = fixed-partner effect, next to real − degseq and degseq − ER.
- `structure_test.py` / `structure_summary.py` = mentor's standalone version (β fixed 0.5). Superseded by the main sweep; kept for reference.

## Fixes after code review (2026-10-07)

- degseq in main sweep: rep r loads rewiring r % 5 (`<real>_rewired0..4`, seeds 1000+r). Results no longer hang on one random graph. No extra runs. Row column `source_graph` says which file was used. (`topology_table.csv` still lists the seed-42 degseq; rewirings have identical degrees, similar clustering. The regression uses that row for "<real>_degseq": fine for degree features, approximate for clustering/assortativity.)
- Reps: ~~5 default, 20 in the high-β band~~ → **10 everywhere** (decided 2026-10-07). Band reps dropped: the "diverging seeds" there were a period-2 oscillation (see re-pilot section), not different end states. 12 β × 10 a × 5 b × 10 reps = 6000 runs per graph, 90,000 total, ≈ 7 h with `fast_model.py`.
- analyze.py steps(): each comparison computed whenever its own pair exists (was: nothing without ER).
- Timing: single-core measurements on this laptop vary 2–4× (e.g. 3 small networks × 3 types: 0.056 vs 0.093 s/step on different runs). Quote ranges, or benchmark actual parallel sweep jobs.
- **Decided:** GRAPHS = real, degseq, shuffled for all 5 networks (15 graphs, no ER/BA/WS). Order: facebook, lastfm, twitch, deezer, github, so small networks finish first. β=0.01 and a=0.9 kept (flag in analysis: β=0.01 not settled at 1000 steps; a=0.9 dead zone). 4350 runs per graph, 65,250 total.
- `drift` column per run = mean(last 250) − mean(first 250) of the 500-step window. ~0 = settled. Lets the sweep itself tell slow convergence from bistability.
- analyze.py diagnostics: per cell status = "still moving" (mean |drift| > 0.02), "settled, repeats differ" (rep std > 0.05, bistability candidate) or "ok". Prints share of moving cells at β=0.01 vs rest, dead-zone mean. Beta plots leave out a ≥ 0.9 and shade β=0.01. Output `sweep_diagnostics.csv/.png`.
- Consequence: no degseq − ER step; regression has only 10 topologies (real + degseq × 5) for 4 features → weak, treat as descriptive.

## Cost of main sweep (measured 2026-10-07)

- Memory: < 0.2 GB Python objects per worker, even GitHub. 8 workers OK.
- One step on all 30 graphs: ~1.4 s (20 graphs without BA/WS: ~1.06 s). Slowest: github_shuffled 0.27 s, github_ws 0.14 s, deezer_shuffled 0.13 s. Shuffled 2–5× slower than static.
- One (β, a, b, rep) setting, 1500 steps, 8 processes (~3× speed-up): ~12 min (30 graphs), ~9 min (20 graphs).

| Grid | Settings | 30 graphs | 20 graphs |
|---|---|---|---|
| 12 β × 10 a × 5 b × 5 reps (as written) | 3000 | ~600 h | ~450 h |
| 11 β × 9 a × 5 b × 5 reps (drop β=0.01, a=0.9) | 2475 | ~495 h | ~370 h |
| 11 × 9 × 5 × 3 reps | 1485 | ~300 h | ~225 h |
| 11 × 9 × 3 b × 3 reps | 891 | ~180 h | ~135 h |
| 8 β × 7 a × 3 b × 3 reps | 504 | ~100 h | ~75 h |

Fermi arms negligible (2 runs per b × rep vs 99 BM). 1000 steps instead of 1500 → ×0.67.

## Re-pilot, ε = 0.05 (`results/pilot_*_eps0.05.*`, same grid)

- Range now 0.06–0.82 (was 0.19–0.69). ε=0.05 gives real room.
- β ≤ 0.3: still flat across β (weak-learning regime), but slow: β=0.01, A=0 still drifting at step 1000 (0.22 and falling). A=0 → every payoff satisfies → slow drift.
- Dead zone A ≳ 0.8b remains (coop ≈ 0.5).
- ~~Bistability~~ **WRONG, corrected 2026-10-07**: at high β, A ≈ 0.65–0.8b the pilot saw reps at 0.16 / 0.76 / 0.80. Real cause = **period-2 oscillation**: whole population flips C↔D every round (seed 0 at steps 990–999: 0.167, 0.808, 0.163, 0.812 …). Pilot saved every 10th step = even steps only → saw one phase; seeds differ only in phase. Mean over all steps ≈ 0.487 for every seed. Main sweep averages every step → fine.
- Oscillation map (numpy, LastFM, 3 seeds, last 500 of 1000 steps, |even − odd| mean): none at b=1.1 or β ≤ 2 (bar few b=2 cells); present for b ≥ 1.5, β ≥ 3, a ≈ 0.6–0.9 (A > 1 = max cooperator payoff), gap up to 0.3–0.58. Means there ≈ 0.49.
- Likely synchronous-update effect (everyone reacts to the same round at once, high β → near-full flips). Mention in paper.
- Recorded per run: `even_odd_gap` = |mean coop on even rounds − odd rounds| over the window. `analyze.py` flags cells with gap > 0.05 as oscillating (separate from status: their mean is still valid) and draws a heatmap row for it.
- Elsewhere: plateau by 300–600 steps, rep spread ≤ 0.01.
- Implication for structure test: β=0.5 sits in the flat regime → likely "no structure effect" there. Interesting regime is high β.

## Compute

- Measured (model.py, plain loops, one core): GitHub 0.14 s/step (~30 s per 200-step run), LastFM 0.01 s/step (~2 s per run). Much faster than the 1–2 min feared. Model build < 1 s.
- Parallel: 8 processes ≈ 3× one process (95 vs 33 runs/min on LastFM), not 8×. Cores share memory.
- Main sweep cost, all 25 graphs, 1000 steps: ≈ 5 min per (β, a, b, rep) setting. 400 settings ≈ 35 h, 1200 ≈ 100 h, 1920 ≈ 160 h. Cut options: 800 steps, 3 reps off the ridge, fewer graphs.
- `src/run.py`: arms bm / fermi_average / fermi_accumulated. A = a·b. One CSV row per run (`results/sweep_runs.csv`), written immediately; restart skips done runs, drops cut-off last line. Per-node cooperation saved for real graphs, rep 0 only (`results/sweep_nodes/*.npy`, gitignored). `--test` = tiny grid.
- Run budget still needs a pilot: rough 25 graphs × 3 arms × β × b × A × reps. Decide sizes after pilot. Multiprocessing if needed.

## Numpy engine for the sweep (decided 2026-10-07)

- `src/fast_model.py`: same BM rules as `model.py`, but all players in arrays. Payoffs = one sparse matrix product (`A @ coop`), BM update = `np.where` over the four cases, shuffled seats = permute p (and per-player counts) each round. networkx still builds every graph; only gives A once per graph.
- Measured single core, 200 steps: LastFM 0.94 → 0.045 s (21×), Facebook 1.11 → 0.11 s (10×), GitHub 6.6 → 0.49 s (13×).
- Parallel, real sweep jobs, 8 processes, 1500 steps (s per run, wall): Facebook/LastFM/Twitch 0.06–0.10, Deezer 0.31–0.43, GitHub 0.77–0.94. **Full sweep (15 graphs, 65,250 runs) ≈ 5 h** (Mesa estimate was 155–350 h). With 10 reps everywhere (90,000 runs) ≈ 7 h.
- Caveat: runs left in background / laptop idle got 10–300× slower (Windows throttling or sleep). Run the sweep in an open terminal, plugged in, sleep off.
- `model.py` (Mesa) stays the readable reference + used by tests, validation, pilot, Fermi.
- `src/check_fast_model.py` → `results/fast_model_check.csv`. Exact: payoffs (7624 players) and BM update (38120 updates, 5 β) identical to Mesa. Statistical: 7 cells × 20 seeds × 600 steps on LastFM (incl. shuffled, β up to 100): all agree, |diff| ≤ 0.0005. Re-run with corrected ε: 7/7 agree, |diff| ≤ 0.0002.
- Same seed ≠ same run in the two versions (random numbers drawn differently).

## Main sweep run (2026-10-08)

- 90,000 BM runs (15 graphs × 6000), 12 processes, 339 min (+ ~10 min building rewirings). No errors, no restarts, 0 missing values. Output `results/sweep_runs.csv`, log `results/sweep.log`, analysis `results/sweep_*`.
- Diagnostics: 8992/9000 cells ok, 8 still moving (all β=0.01); 1062 cells (12%) oscillate (β ≥ 2); dead zone a=0.9 mean coop 0.478.

## Payoff scheme comparison (decided 2026-10-08)

Question: averaged vs accumulated payoff, BM and Fermi. Which topology features predict the difference?

- Graphs: all 25 (5 real + ER/BA/WS/degseq) + shuffled. ER/BA/WS back in: wider range of degree heterogeneity, needed to say which feature predicts anything (5 degree sequences alone is too few).
- Fermi: sweep κ (no single κ). Reason: accumulated payoff gaps ≈ ⟨k⟩ × averaged gaps, so one κ = different selection strength per scheme and per graph.
- BM accumulated: payoff = sum over neighbours, aspiration of seat i = A·k_i (A = a·b as before). Equivalent to averaged payoff with β_i = β·k_i: accumulation = degree-scaled learning sensitivity. Hubs at β·k ≫ 10 behave as win-stay-lose-shift even at small β.
- Literature behind the BM choice:
  - Ezaki & Masuda 2017 (BM on rings): payoff "averaged over the k neighbors". Supports averaged as main arm.
  - Zhou, Wu, Du & Wang (arXiv 1810.00387, SI "Accumulated payoffs"): accumulated payoffs with fixed per-individual aspirations (not strategy-dependent) keep the network-independent weak-selection condition. Degree-scaled A_i is such a personalised aspiration → still inside the theory.
  - Pal & Sengupta 2021 (bioRxiv 10.1101/2021.09.16.460462, BM, summed payoffs): initial aspiration ∝ k_i. Precedent for degree-scaled aspiration.
  - No paper found with BM + accumulated payoffs + one shared fixed A on heterogeneous networks. Shared A would make hubs always satisfied, leaves never → outcome fixed by degree. Rejected.
- Code: `fast_model.py` has `run_fermi` (numpy) and `run_bm(payoff_scheme=...)`. Mesa `BMAgent` takes payoff_scheme too. `check_fast_model.py` checks both.
- First look (numpy, LastFM, 600 steps): Fermi-averaged → cooperation ≈ 0 for κ ≥ 0.1 at b ≥ 1.05 (0.01–0.09 at κ = 0.01). Fermi-accumulated 0.4–0.97. Floor effect: Fermi-averaged sensitivity will be ~0 by construction. Fermi has no mutation → absorbing states, unlike BM (ε). Consider a mutation rate before the sweep.

## Still open

- β range. Pilot.
- Steps, reps, plateau window. Pilot.
- Replication check for BM: Ezaki & Masuda 2017 rings (get params from paper). Fermi: sanity on lattice.
- Add other Twitch regions? After timing pilot.
- Weak-PD note: weak-selection theory gives no cooperation advantage in weak PD for b>1 → all signal at larger β.
