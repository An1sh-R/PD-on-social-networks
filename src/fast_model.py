"""Fast numpy version of the models in model.py, used by the main sweep (run.py).

Same rules as BMAgent / FermiAgent + CooperationModel (weak PD, synchronous rounds),
but all players live in arrays: entry i = the player on node i. One round is a few
array operations instead of a Python loop over every player and neighbour (~10-20x
faster). model.py stays the readable reference; check_fast_model.py checks that the
two agree.

Random numbers are drawn differently from the Mesa version, so the same seed gives
a different run. Averages over seeds must match, single runs will not.
"""
import networkx as nx
import numpy as np
from scipy.special import expit  # safe sigmoid, no overflow for huge payoff gaps


def neighbour_matrix(G):
    """Sparse adjacency matrix A (A[i, j] = 1 if i and j are neighbours) and degrees.
    Nodes must be labelled 0..n-1 (see networks.py), so row i = node i."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=float)
    degree = np.asarray(A.sum(axis=1)).ravel()
    return A, degree


def weak_pd_payoffs(A, degree, coop, b, payoff_scheme="average"):
    """Weak PD payoff of every player. coop: True where the player cooperated.
    Only cooperating neighbours pay anything: 1 to a C player, b to a D player.
    payoff_scheme: "average" (divide by degree) or "accumulated" (sum over neighbours)."""
    n_coop_neighbours = A @ coop.astype(float)  # row i = cooperators next to node i
    total = np.where(coop, n_coop_neighbours * 1.0, n_coop_neighbours * b)
    if payoff_scheme == "average":
        return total / degree
    return total


def bm_update(p, coop, payoff, beta, aspiration):
    """Bush-Mosteller update for all players at once (same four cases as BMAgent.update).
    aspiration: one number, or one value per player (accumulated payoffs)."""
    stimulus = np.tanh(beta * (payoff - aspiration))
    # cooperated: satisfied -> p moves up towards 1, unsatisfied -> down towards 0
    if_cooperated = np.where(stimulus >= 0, p + (1 - p) * stimulus, p + p * stimulus)
    # defected: satisfied -> p moves down towards 0, unsatisfied -> up towards 1
    if_defected = np.where(stimulus >= 0, p - p * stimulus, p - (1 - p) * stimulus)
    return np.where(coop, if_cooperated, if_defected)


def run_bm(A, degree, b, beta, aspiration, epsilon, steps, record_from=0,
           initial_coop=0.5, shuffle_seats=False, payoff_scheme="average", seed=None):
    """One BM run. Same arguments as CooperationModel (agent_type="bm").

    payoff_scheme "accumulated": payoff = sum over neighbours and the aspiration of
    the seat is aspiration * degree (A_i = A k_i), so a hub is not satisfied just for
    having many neighbours. Same as average payoff with beta_i = beta * k_i.

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
    if payoff_scheme == "accumulated":
        seat_aspiration = aspiration * degree  # belongs to the seat, not the player
    else:
        seat_aspiration = aspiration

    for t in range(steps):
        # with probability epsilon the player slips and plays the opposite of what it
        # intended (Ezaki & Masuda 2017)
        prob_cooperate = (1 - epsilon) * p + epsilon * (1 - p)
        coop = rng.random(n) < prob_cooperate
        payoff = weak_pd_payoffs(A, degree, coop, b, payoff_scheme)

        curve[t] = coop.mean()
        if t >= record_from:
            rounds_counted += 1
            coop_rounds += coop

        p = bm_update(p, coop, payoff, beta, seat_aspiration)

        if shuffle_seats:
            # players take new random seats: reorder everything that belongs to a player
            new_order = rng.permutation(n)
            p = p[new_order]
            coop_rounds = coop_rounds[new_order]

    node_cooperation = coop_rounds / rounds_counted
    return curve, p, node_cooperation


def random_neighbours(A, degree, rng):
    """One uniformly random neighbour per player. In the CSR matrix the neighbours of
    node i are A.indices[A.indptr[i] : A.indptr[i + 1]], so pick a random position there."""
    n = A.shape[0]
    row_start = A.indptr[:-1]
    offset = np.floor(rng.random(n) * degree).astype(np.int64)
    return A.indices[row_start + offset]


def fermi_copy_probability(payoff, other, kappa):
    """Probability that each player copies its chosen neighbour (same as FermiAgent.update)."""
    return expit((payoff[other] - payoff) / kappa)


def run_fermi(A, degree, b, kappa, steps, payoff_scheme="average", record_from=0,
              initial_coop=0.5, shuffle_seats=False, mutation=0.0, seed=None):
    """One Fermi imitation run. Same arguments as CooperationModel (agent_type="fermi").
    Returns curve and node_cooperation like run_bm (no p: Fermi players have no propensity)."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    coop = rng.random(n) < initial_coop  # strategy = action, True = C
    coop_rounds = np.zeros(n)
    rounds_counted = 0
    curve = np.zeros(steps)

    for t in range(steps):
        payoff = weak_pd_payoffs(A, degree, coop, b, payoff_scheme)

        curve[t] = coop.mean()
        if t >= record_from:
            rounds_counted += 1
            coop_rounds += coop

        # everyone compares with one random neighbour, using this round's frozen
        # actions and payoffs (synchronous)
        other = random_neighbours(A, degree, rng)
        copies = rng.random(n) < fermi_copy_probability(payoff, other, kappa)
        coop = np.where(copies, coop[other], coop)
        if mutation > 0:
            # strategy flips with probability mutation (same as FermiAgent.update)
            flips = rng.random(n) < mutation
            coop = coop != flips

        if shuffle_seats:
            new_order = rng.permutation(n)
            coop = coop[new_order]
            coop_rounds = coop_rounds[new_order]

    node_cooperation = coop_rounds / rounds_counted
    return curve, node_cooperation
