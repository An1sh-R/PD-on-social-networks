# Follow-up idea: heterogeneous learning sensitivity (parked 2026-10-09)

Status: **parked**. Sweep v2 (`question.md`) runs first. Come back after its analysis.

## Question

Does tying learning sensitivity β to degree change cooperation among aspiration-based (BM) learners on real social networks, and does the cooperation-maximising placement match that of imitation learners under the same game, networks and form of heterogeneity?

## How we got here

- Main BM sweep (2026-10-08, report 2026-10-09): nothing headline-worthy.
  - Structure beyond degree small: real − degseq ≤ 0.009 at β ≤ 1, up to 0.024 at β ≥ 10 (A ≤ 1). "Topology barely matters for RL" already reported (Cimini & Sánchez 2014; Amaral et al. 2016).
  - Fixed partners: +0.01 to +0.03 at low β, +0.03 to +0.06 at high β. Real, small.
  - **Degree is where the action is:** node cooperation vs degree, Spearman −0.5 to −0.77 (β ≤ 3). LastFM b = 1.5, a = 0.3: top-1% hubs 0.17 vs degree-1 nodes 0.45 (β = 0.1); 0.26 vs 0.58 (β = 100). Facebook hubs 0.24 → 0.51 from β = 0.1 to 100.
- β changes cooperation a lot (our sweep; Ezaki et al. 2016).
- BM accumulated payoffs with aspiration A·k = averaged payoffs with β_i ∝ k_i. So the payoff-scheme question is already a question about how β is spread over degree. Make it explicit.

## Literature check (2026-10-09)

Citation lists of Meng 2024 (43), Ezaki 2016 (70), Ezaki & Masuda 2017 (15), Zhou 2021 (63) screened by title (abstracts where relevant); 5 OpenAlex searches; ~20 web searches. Paywalled full texts not read; very new papers may be missing.

**Imitation side: done, answer = slow hubs help.**
- Wu, Zhao & Hu 2026, Chaos Solitons Fractals 210:118666 (doi 10.1016/j.chaos.2026.118666). Imitation on BA; degree-scaled learning and teaching activity. Lower learning activity at hubs promotes cooperation.
- Jiang, Li, Chen & Zhang 2026, J Math Biol 93:42 (doi 10.1007/s00285-026-02453-8). Degree-dependent inertia, weak selection, donation game. Slow hubs with low-degree neighbours help.
- Meng, Cornelius, Liu & Li 2024, Nat Commun (doi 10.1038/s41467-024-47380-8). Update rate ∝ 1/k → heterogeneous networks beat homogeneous.
- Wang 2026, arXiv 2604.19436. Degree-tied weight on personal information, weak selection, GitHub data.
- Chen et al. 2008, arXiv 0812.3518 (imitation rationality ∝ k^β); Wu et al. 2018 IJMPC (imitation capability ∝ degree).

**RL / aspiration side: nothing found with β tied to degree, nor BM on empirical networks.** Closest: heterogeneous aspiration *levels* (Zhou 2018/2021, weak selection); heterogeneous learning rates without networks (PNAS 2025, congestion games); per-player rationality co-evolving with aspiration (PLOS ONE 2023, lattice PGG); Q-learning rates on activity-driven networks (IJMPC 2023).

## Hypotheses

- HB1: heterogeneity alone (same mean β, random placement) changes cooperation little.
- HB2: placement by degree changes cooperation clearly more than random placement (≥ 0.03).
- HB3: best placement for BM may differ from imitation (slow hubs). Hint only: hubs cooperate more at high β (uniform-β node data). Not tested.

## Design (pilot first)

1. **Mixing:** two types (sensitive β_high, insensitive β_low), random placement, vs uniform β with the same mean. Node level: own β and neighbours' β vs node cooperation.
2. **Placement:** same β values assigned by degree, β_i = β·(k_i/⟨k⟩)^γ normalised to the same mean, γ ∈ {−1, −0.5, 0, 0.5, 1}, vs random. Optional: by clustering or k-core. γ = 1 = BM accumulated (exact).
3. **Optional:** search for the best assignment on LastFM (~0.3 s per run), compare with the degree rule.

- Matched imitation arm in our own code: Fermi with κ_i tied to degree the same way, plus update probability ∝ k^γ (mirrors Wu 2026 / Meng 2024). Same game, same networks.
- Hold mean fixed: geometric mean of β (grid is log-spaced) as main choice, arithmetic mean as check (tanh saturates, so it matters).
- Shuffled seats: β travels with the player → breaks the β-degree link (extra control).
- Pilot: LastFM + Facebook, steps 1–2, β ∈ {0.3, 2, 10}, a ∈ {0.2, 0.3, 0.5}, b ∈ {1.3, 1.7}, 5 reps. A few hours.
- **Go/no-go:** best − worst γ ≥ 0.03 in both networks, best γ stable across b, degree placement beats random. Else drop, or fold into the payoff-scheme paper (γ = 0 vs 1).

## Risks

- Incremental ("Wu 2026 with BM"). Realistic venues: CSF, Physica A, PLOS ONE, R Soc Open Sci.
- β mixes speed and sensitivity; tanh saturation → effect may live only in a middle β band.
- Effect may be small (structure effects so far ≤ 0.02).
- Area is active (3 imitation papers 2024–26): if it works, preprint fast.
