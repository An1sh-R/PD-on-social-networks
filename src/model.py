"""Mesa model: weak Prisoner's Dilemma on a fixed network.

Two kinds of player (see ResearchQ/design.md):
- FermiAgent: copies a random neighbour with the Fermi probability (imitation).
- BMAgent: Bush-Mosteller aspiration learning (reinforcement learning).

Weak PD payoffs (row player): C vs C = 1, D vs C = b, C vs D = 0, D vs D = 0.

One round (all players act together):
1. everyone picks an action
2. everyone plays all neighbours and gets a payoff
3. record cooperation
4. everyone updates (Fermi: maybe copy a neighbour; BM: change p)
Step 4 only reads actions and payoffs, which are frozen by then, so updating
players one after another is the same as updating them all at once.
"""
import math

import mesa
from scipy.special import expit  # safe sigmoid, no overflow for huge payoff gaps


class Player(mesa.Agent):
    """Shared part: play the weak PD against every neighbour."""

    def __init__(self, model, payoff_scheme):
        super().__init__(model)
        self.neighbors = []  # filled in by the model
        self.payoff_scheme = payoff_scheme  # "average" or "accumulated"
        self.action = "C"
        self.payoff = 0.0
        self.coop_rounds = 0  # rounds spent cooperating (after the record start)

    def compute_payoff(self):
        n_coop_neighbors = 0
        for other in self.neighbors:
            if other.action == "C":
                n_coop_neighbors += 1

        if self.model.game == "weak_pd":
            # Only cooperating neighbours pay anything: 1 to a C player, b to a D player.
            if self.action == "C":
                total = n_coop_neighbors * 1.0
            else:
                total = n_coop_neighbors * self.model.b
        else:
            # Donation game: every cooperating neighbour gives me b;
            # if I cooperate I pay c to each neighbour.
            total = n_coop_neighbors * self.model.b
            if self.action == "C":
                total = total - self.model.c * len(self.neighbors)

        if self.payoff_scheme == "average":
            total = total / len(self.neighbors)
        self.payoff = total


class FermiAgent(Player):
    """Imitation: compare payoff with one random neighbour, maybe copy their strategy."""

    def __init__(self, model, payoff_scheme, initial_coop):
        super().__init__(model, payoff_scheme)
        if model.random.random() < initial_coop:
            self.strategy = "C"
        else:
            self.strategy = "D"

    def choose_action(self):
        self.action = self.strategy

    def update(self):
        other = self.model.random.choice(self.neighbors)
        payoff_gap = other.payoff - self.payoff
        copy_probability = expit(payoff_gap / self.model.kappa)
        if self.model.random.random() < copy_probability:
            self.strategy = other.action


class BMAgent(Player):
    """Bush-Mosteller (Ezaki et al. 2016). p = probability of cooperating."""

    def __init__(self, model, initial_coop):
        # BM always uses average payoff: the aspiration A needs a fixed scale.
        super().__init__(model, "average")
        self.p = initial_coop

    def choose_action(self):
        eps = self.model.epsilon
        # With probability eps the player slips and plays the opposite of what it
        # intended (Ezaki & Masuda 2017): intends C with prob p, then flips.
        prob_cooperate = (1 - eps) * self.p + eps * (1 - self.p)
        if self.model.random.random() < prob_cooperate:
            self.action = "C"
        else:
            self.action = "D"

    def update(self):
        stimulus = math.tanh(self.model.beta * (self.payoff - self.model.aspiration))
        p = self.p
        if self.action == "C":
            if stimulus >= 0:
                self.p = p + (1 - p) * stimulus
            else:
                self.p = p + p * stimulus
        else:
            if stimulus >= 0:
                self.p = p - p * stimulus
            else:
                self.p = p - (1 - p) * stimulus


def cooperation_fraction(model):
    """Fraction of players who played C this round."""
    n_coop = 0
    for player in model.players:
        if player.action == "C":
            n_coop += 1
    return n_coop / len(model.players)


class CooperationModel(mesa.Model):
    def __init__(self, G, agent_type, b=1.5, payoff_scheme="average", kappa=0.1,
                 beta=1.0, aspiration=0.5, epsilon=0.2, initial_coop=0.5,
                 record_from=0, game="weak_pd", c=1.0, shuffle_seats=False, seed=None):
        """
        G: networkx graph, connected, nodes labelled 0..n-1 (see networks.py)
        agent_type: "fermi" or "bm"
        b: temptation payoff of the weak PD (b > 1)
        payoff_scheme: "average" or "accumulated" (Fermi only; BM is always average)
        kappa: Fermi noise. beta, aspiration, epsilon: BM sensitivity, aspiration A, slip rate.
        record_from: from this round on (0 = first round), count each node's cooperation
        game: "weak_pd" (main study) or "donation" (only to check BM against Ezaki & Masuda 2017)
        c: cost of cooperating in the donation game (b = benefit there)
        shuffle_seats: if True, players get new random nodes after every round. The
            network stays the same, but nobody keeps fixed partners (Ezaki & Masuda 2017).
            node_cooperation() is then per player, not per node.
        seed: int, makes the run repeatable
        """
        super().__init__(rng=seed)  # int seed makes runs repeatable
        if agent_type not in ("fermi", "bm"):
            raise ValueError("agent_type must be 'fermi' or 'bm'")
        if payoff_scheme not in ("average", "accumulated"):
            raise ValueError("payoff_scheme must be 'average' or 'accumulated'")
        if game not in ("weak_pd", "donation"):
            raise ValueError("game must be 'weak_pd' or 'donation'")

        self.game = game
        self.c = c
        self.b = b
        self.kappa = kappa
        self.beta = beta
        self.aspiration = aspiration
        self.epsilon = epsilon
        self.record_from = record_from
        self.rounds_played = 0  # our own counter: mesa's self.steps is already +1 inside step()
        self.rounds_counted = 0
        self.G = G
        self.shuffle_seats = shuffle_seats
        # Neighbour lists looked up once: re-seating every round would otherwise
        # ask networkx for them again each round (slow on big graphs).
        self.neighbor_nodes = []
        for node in range(G.number_of_nodes()):
            self.neighbor_nodes.append(list(G.neighbors(node)))

        # One player per node. Node ids are 0..n-1 so list index = node id.
        self.players = []
        for node in range(G.number_of_nodes()):
            if agent_type == "fermi":
                player = FermiAgent(self, payoff_scheme, initial_coop)
            else:
                player = BMAgent(self, initial_coop)
            self.players.append(player)

        self.seat_players(self.players)

        self.datacollector = mesa.DataCollector(
            model_reporters={"cooperation": cooperation_fraction}
        )

    def step(self):
        for player in self.players:
            player.choose_action()
        for player in self.players:
            player.compute_payoff()

        self.datacollector.collect(self)
        if self.rounds_played >= self.record_from:
            self.rounds_counted += 1
            for player in self.players:
                if player.action == "C":
                    player.coop_rounds += 1

        for player in self.players:
            player.update()
        self.rounds_played += 1

        if self.shuffle_seats:
            new_order = list(self.players)
            self.random.shuffle(new_order)
            self.seat_players(new_order)

    def seat_players(self, order):
        """order[node] = player sitting on that node. Sets everyone's neighbour list."""
        for node in range(len(order)):
            order[node].neighbors = [order[v] for v in self.neighbor_nodes[node]]

    def run(self, n_steps):
        for _ in range(n_steps):
            self.step()

    def node_cooperation(self):
        """Per node: fraction of counted rounds spent cooperating (list indexed by node)."""
        return [p.coop_rounds / self.rounds_counted for p in self.players]
