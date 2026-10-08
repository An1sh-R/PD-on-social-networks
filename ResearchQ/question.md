# Research Question

## Question

How does topology sensitivity of Bush–Mosteller (BM) aspiration learning change with learning sensitivity β on real social networks? Which structural features drive it? Fermi imitation = topology-sensitive baseline.

"Topology sensitivity" = spread of equilibrium cooperation across networks (real + degree-preserving randomised versions), at fixed game params.

## Why this question

- Weak-selection theory (Du, Wu & Wang 2015; Zhou, Wu, Du & Wang 2021): aspiration rules → same outcome on any network. Imitation rules → network-dependent.
- Counter-evidence: Ezaki & Masuda 2017, BM on rings → degree k matters (b/c vs k).
- Theory = memoryless aspiration rule, weak selection, regular/weighted graphs. BM = propensity p carried across rounds. Never tested on real networks.
- Gap: no paper found testing aspiration learning vs imitation on empirical networks across learning intensity, separating structural features. (Targeted search only, not systematic review. Check citations of Zhou 2021, Du 2015, Ezaki & Masuda 2017 before committing.)

## Hypotheses

- H1: low β → BM insensitive to topology (theory anchor).
- H2: sensitivity grows with β (BM drifts from theory).
- H3: Fermi sensitive at all κ.
- H4: features beyond degree sequence (clustering, communities, assortativity) explain the growth.
- H5 (mentor's shuffled seats): fixed partners matter more for Fermi than BM. Re-seating every round should wipe out Fermi cooperation (imitation needs stable clusters) but hurt BM less, mainly at low β. At high β, BM partner effect may grow (fits Ezaki & Masuda 2017 network reciprocity).

## Design sketch

(Up to date version: `design.md`. Three comparisons per real graph: real vs shuffled seats = fixed partners; real vs degseq = structure beyond degree; degseq vs ER = degree heterogeneity.)

- Networks: Facebook, LastFM, Twitch ENGB, Deezer, GitHub (cleaned: no self-loops, largest component).
- Null models per real network: degree-preserving rewire (5 rewirings) + shuffled seats. (Original plan of a ladder + assortativity → + clustering → + communities: dropped for now.)
- Agents: BM (tanh stimulus, ε = flip prob, per Ezaki 2016 / Ezaki & Masuda 2017). Fermi later.
- Sweep: β × aspiration (a = A/b) × b. Payoff scheme only matters for Fermi (later).
- Outcome: mean cooperation over last 500 of 1500 steps; drift and oscillation recorded per run.

## Decisions

Settled choices (game, networks, agents, measures) in `design.md`.

- Game: weak PD (decided).
- Networks: Facebook, Deezer, GitHub, LastFM, Twitch ENGB + synthetic equivalents (decided; supersedes SNAP email/GrQc list above).
- Replication target: Cameron & Cintrón-Arias 2020 = Caltech Facebook100, donation game b=150 c=1, death–birth. NOT ego-Facebook / Fermi / b=1.5. Different game from ours → not a direct check. Use Ezaki & Masuda 2017 rings for BM instead.
- Define "weak" for BM: small β = analogy to weak selection, not equivalence.

## Key refs

- Du, Wu & Wang 2015, Sci Rep. Aspiration ≈ well-mixed.
- Zhou, Wu, Du & Wang 2021, Nat Commun. Robust on all networks, weak selection.
- Ezaki, Horita, Takezawa & Masuda 2016, PLOS Comput Biol. BM model. **Varied β**: cooperation more frequent at large β (Fig 2B). Lattice only, no static vs shuffled. Pavlov = β → ∞ limit.
  - Ezaki & Masuda 2017 then fixed β = 0.2 for static vs shuffled ("β did not considerably alter the behavior … unless extreme"). Their payoff range ~7 vs ours ~1.5 → their β = 0.2 ≈ our β ≈ 1.
  - Our sweep (2026-10-08): fixed-partner effect grows with β (A ≤ 1: 0.015 at β = 0.1 → 0.047 at β = 100). Not shown before as far as found (15 citing papers of E&M 2017 checked, Semantic Scholar). Caveat: shuffling also removes fixed degree position.
- Ezaki & Masuda 2017, PLOS ONE. BM on rings.
- Cameron & Cintrón-Arias 2020, arXiv:2004.06817. Real-network PD.
- Ohtsuki et al. 2006, Nature. b/c > k (death–birth only).
- Lozano, Arenas & Sánchez 2008, PLOS ONE. Community structure on real networks.
- Gracia-Lázaro et al. 2012, PNAS; Rand et al. 2014, PNAS. Human experiments.
- Fu et al. 2007, Phys Lett A. Online social network PD (guide wrongly cites as Liu, Chen & Perc).
