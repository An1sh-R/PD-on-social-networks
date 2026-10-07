"""Simple checks for model.py. Run from project root: python src/test_model.py

Each check is a plain function that fails with an assert if something is wrong.
"""
import math

import networkx as nx

from model import CooperationModel

TOLERANCE = 1e-9


def close(a, b):
    return abs(a - b) < TOLERANCE


def test_payoff_star_all_neighbors_cooperate():
    # node 0 is the centre, nodes 1-3 are leaves. b = 1.5
    G = nx.star_graph(3)
    model = CooperationModel(G, "fermi", b=1.5, payoff_scheme="accumulated", seed=1)
    for leaf in model.players[1:]:
        leaf.action = "C"
    model.players[0].action = "D"
    for player in model.players:
        player.compute_payoff()
    assert close(model.players[0].payoff, 3 * 1.5)  # defector centre, 3 cooperating leaves
    assert close(model.players[1].payoff, 0.0)  # C leaf next to D centre gets nothing


def test_payoff_star_mixed():
    G = nx.star_graph(3)
    model = CooperationModel(G, "fermi", b=1.5, payoff_scheme="accumulated", seed=1)
    model.players[0].action = "C"
    model.players[1].action = "C"
    model.players[2].action = "C"
    model.players[3].action = "D"
    for player in model.players:
        player.compute_payoff()
    assert close(model.players[0].payoff, 2.0)  # C centre, 2 C leaves
    assert close(model.players[1].payoff, 1.0)  # C leaf, C centre
    assert close(model.players[3].payoff, 1.5)  # D leaf, C centre


def test_payoff_average_divides_by_degree():
    G = nx.star_graph(3)
    model = CooperationModel(G, "fermi", b=1.5, payoff_scheme="average", seed=1)
    for leaf in model.players[1:]:
        leaf.action = "C"
    model.players[0].action = "D"
    for player in model.players:
        player.compute_payoff()
    assert close(model.players[0].payoff, 1.5)  # 4.5 / 3


def test_payoff_donation_game():
    # star, b=6, c=1, average payoff. Centre C, leaves C, C, D.
    G = nx.star_graph(3)
    model = CooperationModel(G, "bm", game="donation", b=6.0, c=1.0, seed=1)
    model.players[0].action = "C"
    model.players[1].action = "C"
    model.players[2].action = "C"
    model.players[3].action = "D"
    for player in model.players:
        player.compute_payoff()
    assert close(model.players[0].payoff, (2 * 6 - 3 * 1) / 3)  # gets 2 gifts, pays 3 neighbours
    assert close(model.players[1].payoff, 6 - 1)  # C leaf: gift from centre, pays centre
    assert close(model.players[3].payoff, 6)  # D leaf: gift, pays nothing


def test_bm_always_uses_average_payoff():
    G = nx.star_graph(3)
    model = CooperationModel(G, "bm", payoff_scheme="accumulated", seed=1)
    assert model.players[0].payoff_scheme == "average"


def test_bm_update_four_cases():
    # beta=1, A=0.5, p=0.5. payoff 1 -> stimulus +tanh(0.5), payoff 0 -> -tanh(0.5)
    G = nx.path_graph(2)
    model = CooperationModel(G, "bm", beta=1.0, aspiration=0.5, seed=1)
    s = math.tanh(0.5)
    cases = [
        ("C", 1.0, 0.5 + 0.5 * s),  # cooperated, satisfied: p goes up
        ("C", 0.0, 0.5 - 0.5 * s),  # cooperated, unhappy: p goes down
        ("D", 1.0, 0.5 - 0.5 * s),  # defected, satisfied: p goes down
        ("D", 0.0, 0.5 + 0.5 * s),  # defected, unhappy: p goes up
    ]
    for action, payoff, expected_p in cases:
        player = model.players[0]
        player.p = 0.5
        player.action = action
        player.payoff = payoff
        player.update()
        assert close(player.p, expected_p), (action, payoff, player.p, expected_p)


def test_bm_p_stays_between_0_and_1():
    G = nx.path_graph(2)
    model = CooperationModel(G, "bm", beta=1000.0, aspiration=0.5, seed=1)
    player = model.players[0]
    for _ in range(2000):
        player.action = model.random.choice(["C", "D"])
        player.payoff = model.random.random() * 2.0
        player.update()
        assert -1e-12 <= player.p <= 1 + 1e-12, player.p


def test_bm_slip_flips_intended_action():
    # eps = 0.2: a sure cooperator (p=1) still defects 20% of the time, a sure
    # defector (p=0) cooperates 20% of the time (flip, not a random pick)
    G = nx.path_graph(2)
    model = CooperationModel(G, "bm", epsilon=0.2, seed=1)
    player = model.players[0]
    for p, expected_coop in [(1.0, 0.8), (0.0, 0.2), (0.5, 0.5)]:
        player.p = p
        n_coop = 0
        for _ in range(20000):
            player.choose_action()
            if player.action == "C":
                n_coop += 1
        assert abs(n_coop / 20000 - expected_coop) < 0.015, (p, n_coop / 20000)


def test_fermi_huge_payoff_gap_does_not_overflow():
    G = nx.path_graph(2)
    model = CooperationModel(G, "fermi", kappa=0.1, seed=1)
    a, b_player = model.players
    a.strategy, a.action, a.payoff = "C", "C", 0.0
    b_player.strategy, b_player.action, b_player.payoff = "D", "D", 1e6
    a.update()  # gap/kappa = 1e7, math.exp would overflow
    assert a.strategy == "D"  # much better neighbour is always copied

    a.strategy, a.action, a.payoff = "C", "C", 1e6
    b_player.payoff = 0.0
    a.update()
    assert a.strategy == "C"  # much worse neighbour is never copied


def test_fermi_initial_strategies():
    G = nx.cycle_graph(20)
    all_c = CooperationModel(G, "fermi", initial_coop=1.0, seed=1)
    all_d = CooperationModel(G, "fermi", initial_coop=0.0, seed=1)
    assert all(p.strategy == "C" for p in all_c.players)
    assert all(p.strategy == "D" for p in all_d.players)


def test_uniform_populations_stay_uniform():
    G = nx.cycle_graph(20)
    for start, expected in [(1.0, 1.0), (0.0, 0.0)]:
        model = CooperationModel(G, "fermi", initial_coop=start, seed=1)
        model.run(10)
        assert model.datacollector.get_model_vars_dataframe()["cooperation"].iloc[-1] == expected


def test_run_records_every_step():
    G = nx.connected_watts_strogatz_graph(100, 4, 0.1, seed=1)
    for agent_type in ["fermi", "bm"]:
        model = CooperationModel(G, agent_type, seed=1)
        model.run(10)
        data = model.datacollector.get_model_vars_dataframe()
        assert len(data) == 10
        assert data["cooperation"].between(0, 1).all()
        assert model.steps == 10


def test_node_cooperation_window():
    G = nx.connected_watts_strogatz_graph(100, 4, 0.1, seed=1)
    model = CooperationModel(G, "bm", record_from=5, seed=1)
    model.run(10)
    assert model.rounds_counted == 5  # rounds 5..9 (round numbers start at 0)
    node_coop = model.node_cooperation()
    assert len(node_coop) == 100
    assert all(0 <= x <= 1 for x in node_coop)


def test_same_seed_same_result():
    G = nx.connected_watts_strogatz_graph(100, 4, 0.1, seed=1)
    for agent_type in ["fermi", "bm"]:
        runs = []
        for _ in range(2):
            model = CooperationModel(G, agent_type, seed=7)
            model.run(20)
            runs.append(list(model.datacollector.get_model_vars_dataframe()["cooperation"]))
        assert runs[0] == runs[1]


def test_shuffle_seats_keeps_seat_degrees():
    # star: whoever sits in the centre must have 3 neighbours, leaves 1
    G = nx.star_graph(3)
    model = CooperationModel(G, "bm", shuffle_seats=True, seed=3)
    centre_players = set()
    for _ in range(30):
        model.step()
        degrees = sorted(len(p.neighbors) for p in model.players)
        assert degrees == [1, 1, 1, 3]
        for p in model.players:
            if len(p.neighbors) == 3:
                centre_players.add(p)
            for other in p.neighbors:
                assert p in other.neighbors  # links stay two-way
    assert len(centre_players) > 1  # the centre seat really changes hands


def test_bad_arguments_rejected():
    G = nx.path_graph(2)
    for kwargs in [{"agent_type": "q-learning"}, {"agent_type": "fermi", "payoff_scheme": "total"}]:
        try:
            CooperationModel(G, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"no error for {kwargs}")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"all {len(tests)} checks passed")
