# Project rules

Research project: BM aspiration learning vs Fermi imitation on real networks. Question in `ResearchQ/question.md`. Decisions in `ResearchQ/design.md`. Read both before coding.

## Communication

- Sacrifice grammar for concision. Short fragments ok.

## Code

- Undergrad-readable. Plain loops, plain functions, descriptive names.
- No clever one-liners, no nested comprehensions, no metaprogramming, no fancy abstractions.
- Libraries: mesa + networkx for simulation. numpy/pandas/matplotlib ok for data and plots. Nothing exotic.
- Short comments saying why, not what.
- Pin mesa version in requirements.txt. Use a venv (Python 3.12+, `py -3.14 -m venv .venv`).
- Run scripts from project root: `python src/networks.py`.

## Layout (keep flat)

```
Experimentation/
  CLAUDE.md
  requirements.txt
  ResearchQ/        guide.pdf, question.md, design.md
  data/             edge lists (real) + synthetic/ (generated once, fixed)
  src/              ALL code lives here, flat, no subfolders
    networks.py       load, clean, synthetic equivalents, topology metrics   [done]
    plot_graphs.py    draw each real graph + synthetic equivalents -> graphs/ [done]
    model.py          mesa agents (BM, Fermi) + model                         [done]
    test_model.py     simple asserts for model.py (python src/test_model.py)  [done]
    run.py            parameter sweep -> results/
    analyze.py        stats + figures
  graphs/           network drawings (one png per real graph)
  results/          csv, result figures
```

- Real edge lists in `data/` (5 graphs). Synthetic equivalents cached in `data/synthetic/`. Spare Twitch regions in `../snaps/`.
- Scripts import each other directly (`from networks import ...`). No packages, no `__init__.py`.
- Split a file only when it passes ~300 lines.
- No utils grab-bag.

## Git

- Remote: https://github.com/An1sh-R/PD-on-social-networks.git (branch main).
- Commit + push automatically at each milestone (a finished, checked piece of work). Short message, say what and why.
- Never commit: `.venv/`, `ResearchQ/guide.pdf`.
