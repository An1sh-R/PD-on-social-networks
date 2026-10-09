# Research Question

## Question

Weak PD on real social networks. How does cooperation depend on network structure for Bush–Mosteller (BM) reinforcement learners vs Fermi imitators, with averaged vs accumulated payoffs? Which structural features matter, and how does that change with learning strength (β for BM, 1/κ for Fermi)?

"Topology sensitivity" = spread of late-window cooperation across a family's graph types (real, degseq, ER, BA, WS), at fixed game and learning params, mean over the 5 families. Plus paired contrasts (below).

## Why this question

- Cooperation on networks is mostly studied with imitation (Fermi) on synthetic graphs. Real networks less (Fu et al. 2007, Lozano et al. 2008, Cameron & Cintrón-Arias 2020).
- BM = reinforcement learning: players learn from their own payoff, nobody copies anyone. Different mechanism from imitation, so structure may matter differently.
- Ezaki et al. 2016 (lattice): β changes BM cooperation. Ezaki & Masuda 2017 (rings): fixed partners help BM cooperation, effect depends on degree k and b/c.
- Payoff scheme (averaged vs accumulated) changes Fermi results strongly (Santos & Pacheco 2005, Ichinose & Sayama 2017). Little known for BM.
- Gap: no paper found with BM vs Fermi on empirical networks, across learning strength and both payoff schemes, separating structural features. (Targeted search only, not a systematic review: 15 citing papers of Ezaki & Masuda 2017 checked, Semantic Scholar.)

## Hypotheses

(H1–H5 reworded 2026-10-09, before sweep v2: framing is purely empirical.)

- H1 (BM, small β): in the slow-learning regime (β ≤ ~0.3, cooperation flat in β) the topology effect is small and comes mostly from degrees: averaged-payoff noise ∝ 1/k, and each player's p settles to a level set by its own and its neighbours' degrees. Clustering and communities should matter little. Test: real − degseq ≈ 0, degseq − ER ≠ 0.
- H2 (BM, growing β): topology effect grows with β. Main sweep (2026-10-08) saw the fixed-partner effect grow (A ≤ 1: 0.015 at β = 0.1 → 0.047 at β = 100); v2 checks it on all graph types.
- H3 (Fermi): topology-sensitive where copying follows payoff gaps closely (κ ≤ 1). At κ = 10 imitation is near-neutral (copy probability ≈ 0.45–0.55), so every graph sits near 0.5: small spread there by construction, not a finding. BM vs Fermi compared relative to each learner's own shuffled-seats baseline, not as raw spread on β vs 1/κ (no common scale).
- H4 (which structure): reported as graph-type contrasts, not single features. With these graphs, clustering, modularity and assortativity move together (real and WS vs the rest), so they cannot be separated:
  - degseq − ER = degree spread
  - WS − ER = clustering + locality, at near-equal degrees
  - real − degseq = everything beyond the degree sequence (clustering, communities, degree mixing)
  - real − shuffled = fixed partners (see H5)
- H5 (mentor's shuffled seats): fixed partners matter more for Fermi than BM. Re-seating every round should wipe out Fermi cooperation (imitation needs stable clusters) but hurt BM less, mainly at low β. At high β, BM partner effect may grow (fits Ezaki & Masuda 2017 network reciprocity). Caveat: shuffling also removes fixed degree position.

### Payoff comparison: averaged vs accumulated (written 2026-10-09, BEFORE sweep v2)

Accumulated payoffs scaled by ⟨k⟩ (design.md). For BM this is exactly averaged BM with β_i = β·k_i/⟨k⟩: the effect is 0 on a regular graph and grows with degree spread **by construction**. So "degree CV predicts the BM payoff effect" is built in, not a result. Only real vs degseq (same degrees) can show more.

- H6 (Fermi): accumulated > averaged on every static graph at κ ≤ 1; gap biggest on heterogeneous graphs (BA, GitHub, Twitch), smallest on near-regular ones (WS, ER). Shuffled seats: both near the mutation floor. Basis: Santos & Pacheco 2005, Ichinose & Sayama 2017, our `validate_fermi.py`. Ichinose & Sayama's "gap shrinks with ⟨k⟩" is NOT predicted here: our ⟨k⟩ scaling removes the overall sensitivity part.
- H7 (BM): |accumulated − averaged| smaller than for Fermi at every graph, because BM players do not copy successful hubs, so hub success cannot spread. Sign not predicted (hubs learn faster, leaves slower; cooperation vs β is flat for β ≤ 1 and rises for β 2–10).
- H8 (structure beyond degree): BM: payoff effect on real ≈ on degseq (|real − degseq| small next to the effect itself). Fermi: no direction predicted (clustering and assortativity are known to change how hub strategies spread, but the literature disagrees on direction).
- Fermi averaged at the mutation floor (cooperation < 0.05, mostly κ ≤ 0.1, b ≥ 1.02): no topology comparison possible there; reported only where off the floor.

### Reporting rules (set before sweep v2)

- Per family and per regime, never pooled over families (audit 2026-10-08: opposite signs cancel).
- Smallest effect worth reporting: |difference| ≥ 0.01 cooperation. Many CIs are computed (families × regimes × β × contrasts): read patterns across β, not single CIs.
- Dead zone (a ≥ 0.8) and β = 0.01 (not settled) flagged, not used for claims.

## Design sketch

(Up to date version: `design.md`.)

- Networks: Facebook, LastFM, Twitch ENGB, Deezer, GitHub (no self-loops, largest component).
- Per real network: degseq (5 connected degree-preserving rewirings), ER, BA, WS, shuffled seats. 30 graphs.
- Arms: BM averaged, BM accumulated (aspiration A·k_i), Fermi averaged, Fermi accumulated (mutation 0.01). Accumulated scaled by ⟨k⟩.
- Sweep: BM β × a (A = a·b) × b; Fermi κ × b. 10 reps.
- Outcome: mean cooperation over last 500 of 1500 steps (BM), last 1000 of 3000 (Fermi); drift, oscillation, assortment recorded per run.

## Decisions

Settled choices (game, networks, agents, measures) in `design.md`.

- Game: weak PD (R = 1, T = b, S = P = 0; Nowak & May 1992). "Weak" is the game's name: S = P = 0, the mildest PD.
- Networks: Facebook, Deezer, GitHub, LastFM, Twitch ENGB + synthetic equivalents (decided; supersedes SNAP email/GrQc list).
- Replication target: Cameron & Cintrón-Arias 2020 = Caltech Facebook100, donation game b=150 c=1, death–birth. NOT ego-Facebook / Fermi / b=1.5. Different game from ours → not a direct check. Use Ezaki & Masuda 2017 rings for BM instead (`validate_bm.py`), Ichinose & Sayama 2017 for Fermi (`validate_fermi.py`).

## Key refs

- Nowak & May 1992, Nature. Weak PD on lattices.
- Ezaki, Horita, Takezawa & Masuda 2016, PLOS Comput Biol. BM model. **Varied β**: cooperation more frequent at large β (Fig 2B). Lattice only, no static vs shuffled. Pavlov = β → ∞ limit.
  - Ezaki & Masuda 2017 then fixed β = 0.2 for static vs shuffled ("β did not considerably alter the behavior … unless extreme"). Their payoff range ~7 vs ours ~1.5 → their β = 0.2 ≈ our β ≈ 1.
  - Our sweep (2026-10-08): fixed-partner effect grows with β (A ≤ 1: 0.015 at β = 0.1 → 0.047 at β = 100). Not shown before as far as found (15 citing papers of E&M 2017 checked, Semantic Scholar). Caveat: shuffling also removes fixed degree position.
- Ezaki & Masuda 2017, PLOS ONE. BM on rings, static vs shuffled partners.
- Santos & Pacheco 2005, PRL. Heterogeneous networks + accumulated payoffs → cooperation.
- Ichinose & Sayama 2017, Artificial Life (arXiv 1412.2311). Averaged vs accumulated payoffs on randomised BA.
- Pal & Sengupta 2021, bioRxiv 10.1101/2021.09.16.460462. BM with summed payoffs, initial aspiration ∝ k_i.
- Cameron & Cintrón-Arias 2020, arXiv:2004.06817. Real-network PD.
- Lozano, Arenas & Sánchez 2008, PLOS ONE. Community structure on real networks.
- Gracia-Lázaro et al. 2012, PNAS; Rand et al. 2014, PNAS. Human experiments.
- Fu et al. 2007, Phys Lett A. Online social network PD (guide wrongly cites as Liu, Chen & Perc).
