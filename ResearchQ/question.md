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

## Design sketch

- Networks: SNAP ego-Facebook, email-Eu-core, ca-GrQc. Clean first: drop self-loops, keep largest component.
- Null models per real network: degree-preserving rewire → + assortativity → + clustering → + communities. Each step isolates one feature.
- Agents: BM (tanh stimulus, ε noise, per Ezaki 2016) vs Fermi.
- Sweep: β (and aspiration A), b, payoff scheme (accumulated vs averaged). Payoff scheme crossed = confound control.
- Outcome: mean cooperation over late window, check plateau per run.

## Decisions

Settled choices (game, networks, agents, measures) in `design.md`.

- Game: weak PD (decided).
- Networks: Facebook, Deezer, GitHub, LastFM, Twitch ENGB + synthetic equivalents (decided; supersedes SNAP email/GrQc list above).
- Replication target: Cameron & Cintrón-Arias 2020 = Caltech Facebook100, donation game b=150 c=1, death–birth. NOT ego-Facebook / Fermi / b=1.5. Different game from ours → not a direct check. Use Ezaki & Masuda 2017 rings for BM instead.
- Define "weak" for BM: small β = analogy to weak selection, not equivalence.

## Key refs

- Du, Wu & Wang 2015, Sci Rep. Aspiration ≈ well-mixed.
- Zhou, Wu, Du & Wang 2021, Nat Commun. Robust on all networks, weak selection.
- Ezaki, Horita, Takezawa & Masuda 2016, PLOS Comput Biol. BM model.
- Ezaki & Masuda 2017, PLOS ONE. BM on rings.
- Cameron & Cintrón-Arias 2020, arXiv:2004.06817. Real-network PD.
- Ohtsuki et al. 2006, Nature. b/c > k (death–birth only).
- Lozano, Arenas & Sánchez 2008, PLOS ONE. Community structure on real networks.
- Gracia-Lázaro et al. 2012, PNAS; Rand et al. 2014, PNAS. Human experiments.
- Fu et al. 2007, Phys Lett A. Online social network PD (guide wrongly cites as Liu, Chen & Perc).
