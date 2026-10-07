"""Fast numpy version of the BM model in model.py, used by the main sweep (run.py).

Same rules as BMAgent + CooperationModel (weak PD, average payoff, synchronous
rounds), but all players live in arrays: entry i = the player on node i. One round
is a few array operations instead of a Python loop over every player and neighbour
(~10-20x faster). model.py stays the readable reference; check_fast_model.py
checks that the two agree.

Random numbers are drawn differently from the Mesa version, so the same seed gives
a different run. Averages over seeds must match, single runs will not.
"""
import networkx as nx
import numpy as np


def neighbour_matrix(G):
    """Sparse adjacency matrix A (A[i, j] = 1 if i and j are neighbours) and degrees.
    Nodes must be labelled 0..n-1 (see networks.py), so row i = node i."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=float)
    degree = np.asarray(A.sum(axis=1)).ravel()
    return A, degree


def weak_pd_payoffs(A, degree, coop, b):
    """Average weak PD payoff of every player. coop: True where the player cooperated.
    Only cooperating neighbours pay anything: 1 to a C player, b to a D player."""
    n_coop_neighbours = A @ coop.astype(float)  # row i = cooperators next to node i
    total = np.where(coop, n_coop_neighbours * 1.0, n_coop_neighbours * b)
    return total / degree


def bm_update(p, coop, payoff, beta, aspiration):
    """Bush-Mosteller update for all players at once (same four cases as BMAgent.update)."""
    stimulus = np.tanh(beta * (payoff - aspiration))
    # cooperated: satisfied -> p moves up towards 1, unsatisfied -> down towards 0
    if_cooperated = np.where(stimulus >= 0, p + (1 - p) * stimulus, p + p * stimulus)
    # defected: satisfied -> p moves down towards 0, unsatisfied -> up towards 1
    if_defected = np.where(stimulus >= 0, p - p * stimulus, p - (1 - p) * stimulus)
    return np.where(coop, if_cooperated, if_defected)


def run_bm(A, degree, b, beta, aspiration, epsilon, steps, record_from=0,
           initial_coop=0.5, shuffle_seats=False, seed=None):
    """One BM run. Same arguments as CooperationModel (agent_type="bm").

    Returns:
        curve: cooperation fraction in every round (length steps)
        p: final cooperation probability of every player
        node_cooperation: per player, fraction of counted rounds spent cooperating
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    p = np.full(n, float(initial_coop))
    coop_rounds = np.zeros(n)  # rounds spent cooperating, from record_from on
    rounds_counted = 0
    curve = np.zeros(steps)

    for t in range(steps):
        # with probability epsilon the player slips and plays the opposite of what it
        # intended (Ezaki & Masuda 2017)
        prob_cooperate = (1 - epsilon) * p + epsilon * (1 - p)
        coop = rng.random(n) < prob_cooperate
        payoff = weak_pd_payoffs(A, degree, coop, b)

        curve[t] = coop.mean()
        if t >= record_from:
            rounds_counted += 1
            coop_rounds += coop

        p = bm_update(p, coop, payoff, beta, aspiration)

        if shuffle_seats:
            # players take new random seats: reorder everything that belongs to a player
            new_order = rng.permutation(n)
            p = p[new_order]
            coop_rounds = coop_rounds[new_order]

    node_cooperation = coop_rounds / rounds_counted
    return curve, p, node_cooperation
