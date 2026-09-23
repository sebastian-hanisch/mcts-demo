"""Die zentrale Korrektheits-Kette für MCTS: gültige Pfade, Kosten nie unter dem Optimum (gegen Brute-Force), Determinismus je Seed, UCB1 gegen Handrechnung,
Baum-Invarianten (Besuchszahlen, Rewards, einfache Pfade), Rückpropagation, Buchführung der Knotenbesuche, Anytime-Folge, Sonderfälle und die Kopie der
Vergleichsgrößen."""

import math

import numpy as np
import pytest

import mcts_algorithm as A
import mcts_graph as G
import mcts_scenario as S

EPS = 1e-9
INF = float("inf")


def _brute_force_shortest_cost(graph, start, goal):
    best = None
    stack = [(start, [start], 0.0)]
    while stack:
        node, path, cost = stack.pop()
        if node == goal:
            if best is None or cost < best:
                best = cost
            continue
        for v, w in zip(graph.neighbors[node], graph.weights[node]):
            if v not in path:
                stack.append((v, path + [v], cost + w))
    return best


def _run(inst, iterations=400, seed=0, **kw):
    return A.mcts_search(inst.graph, inst.start, inst.goal, iterations, seed=seed, **kw)


def _assert_valid(inst, res):
    if not res.solved:
        assert res.path == [] and res.cost == INF
        return
    assert res.path[0] == inst.start and res.path[-1] == inst.goal and len(set(res.path)) == len(res.path)
    for u, v in zip(res.path[:-1], res.path[1:]):
        assert v in inst.graph.neighbors[u]
    assert G.path_cost(inst.graph, res.path) == pytest.approx(res.cost, abs=1e-6)


# --- Gültigkeit und Untergrenze ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rollout", A.ROLLOUT_MODES)
@pytest.mark.parametrize("reward", A.REWARD_MODES)
@pytest.mark.parametrize("seed", range(6))
def test_result_is_a_valid_simple_path_with_recomputed_cost(seed, reward, rollout):
    inst = S.grid_instance(side=7, obstacle_pct=20, seed=seed)
    _assert_valid(inst, _run(inst, 300, seed=seed, reward=reward, rollout=rollout))


@pytest.mark.parametrize("seed", range(12))
def test_never_beats_the_brute_force_optimum(seed):
    inst = S.grid_instance(side=5, obstacle_pct=15, seed=seed)
    optimum = _brute_force_shortest_cost(inst.graph, inst.start, inst.goal)
    for kw in (dict(), dict(rollout="guided"), dict(c=0.0), dict(horizon=3)):
        res = _run(inst, 500, seed=seed, **kw)
        if res.solved:
            assert res.cost >= optimum - 1e-6


@pytest.mark.parametrize("seed", range(6))
def test_robust_path_is_a_valid_solution_when_it_reaches_the_goal(seed):
    inst = S.grid_instance(side=6, obstacle_pct=10, seed=seed)
    res = _run(inst, 1500, seed=seed)
    if res.robust_path:
        assert res.robust_path[0] == inst.start and res.robust_path[-1] == inst.goal
        assert G.path_cost(inst.graph, res.robust_path) == pytest.approx(res.robust_cost, abs=1e-6) and res.robust_cost >= res.cost - 1e-9
    else:
        assert res.robust_cost == INF


def test_deterministic_per_seed_and_different_seeds_differ():
    inst = S.grid_instance(side=9, obstacle_pct=20, seed=3)
    a, b, c = _run(inst, 600, seed=5), _run(inst, 600, seed=5), _run(inst, 600, seed=6)
    assert (a.path, a.cost, a.visits_total, a.tree_state, a.anytime) == (b.path, b.cost, b.visits_total, b.tree_state, b.anytime)
    assert (a.tree_state, a.visits_total) != (c.tree_state, c.visits_total)


def test_invalid_arguments_are_rejected():
    inst = S.grid_instance(side=5, obstacle_pct=0, seed=1)
    for kw in (dict(iterations=-1), dict(horizon=0), dict(c=-0.1), dict(rollout="x"), dict(reward="x")):
        args = dict(iterations=10, horizon=5, c=1.0, rollout="uniform", reward="goal")
        args.update(kw)
        with pytest.raises(ValueError):
            A.mcts_search(inst.graph, inst.start, inst.goal, args.pop("iterations"), **args)


# --- UCB1 und Selektion ---------------------------------------------------------------------------------------------------------------------


def test_ucb1_matches_the_hand_computation():
    assert A.ucb1(3.0, 4, 10, 1.0) == pytest.approx(0.75 + math.sqrt(math.log(10) / 4))
    assert A.ucb1(3.0, 4, 10, 0.0) == pytest.approx(0.75)
    assert A.ucb1(0.0, 1, 1, 5.0) == 0.0                                   # ln 1 = 0
    assert A.ucb1(2.0, 2, 8, 2.0) > A.ucb1(2.0, 4, 8, 2.0)                 # weniger besucht -> mehr Erkundungsbonus


def _two_branch_graph():
    """S -> A -> Z (kurz) und S -> B -> Z (lang), nur eine Kante pro Schritt."""
    xy = [(0, 0), (5, 1), (5, -8), (10, 0)]
    return G.from_edges(4, xy, [(0, 1, 5.1), (1, 3, 5.1), (0, 2, 9.4), (2, 3, 9.4)]), 0, 3


def test_small_c_concentrates_visits_on_the_better_branch_and_large_c_balances_them():
    graph, s, t = _two_branch_graph()
    greedy = A.mcts_search(graph, s, t, 600, seed=1, c=0.05)
    balanced = A.mcts_search(graph, s, t, 600, seed=1, c=50.0)
    a, b = 1, 2                                                             # Zustände A und B; Wurzelkinder in dieser Reihenfolge
    def root_visits(res, state):
        return next(res.tree_visits[i] for i, (st, p) in enumerate(zip(res.tree_state, res.tree_parent)) if p == 0 and st == state)
    assert root_visits(greedy, a) > 5 * root_visits(greedy, b)
    assert abs(root_visits(balanced, a) - root_visits(balanced, b)) <= 0.1 * 600           # nahezu gleich verteilt
    assert greedy.cost == pytest.approx(10.2) and balanced.cost == pytest.approx(10.2)


# --- Baum-Invarianten und Rückpropagation --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [dict(), dict(rollout="guided"), dict(reward="progress"), dict(c=0.0), dict(horizon=2)])
@pytest.mark.parametrize("seed", range(5))
def test_tree_invariants(seed, kw):
    inst = S.grid_instance(side=8, obstacle_pct=20, seed=seed)
    res = _run(inst, 400, seed=seed, **kw)
    n = res.tree_size
    assert len(res.tree_state) == len(res.tree_parent) == len(res.tree_visits) == len(res.tree_reward) == len(res.tree_iteration) == n
    assert res.tree_state[0] == inst.start and res.tree_parent[0] == -1 and res.tree_visits[0] == res.iterations
    kids = {i: [] for i in range(n)}
    for i in range(1, n):
        kids[res.tree_parent[i]].append(i)
        assert res.tree_parent[i] < i and res.tree_state[i] in inst.graph.neighbors[res.tree_state[res.tree_parent[i]]]
    for i in range(n):
        path, node = [], i
        while node != -1:
            path.append(res.tree_state[node])
            node = res.tree_parent[node]
        assert len(set(path)) == len(path)                                  # kein Zustand doppelt auf einem Weg
        assert 0.0 <= res.tree_reward[i] <= res.tree_visits[i] + EPS
        if kids[i]:
            assert res.tree_visits[i] == sum(res.tree_visits[k] for k in kids[i]) + (0 if i == 0 else 1)
    assert list(res.tree_iteration) == sorted(res.tree_iteration) and res.tree_iteration[0] == 0


@pytest.mark.parametrize("kw", [dict(), dict(rollout="guided", reward="progress")])
def test_backpropagation_root_reward_is_the_sum_of_all_iteration_rewards(kw):
    inst = S.grid_instance(side=7, obstacle_pct=15, seed=2)
    res = _run(inst, 300, seed=4, **kw)
    assert len(res.replay) == 300 and all(0.0 <= r <= 1.0 for _n, _roll, r in res.replay)
    assert res.tree_reward[0] == pytest.approx(sum(r for _n, _roll, r in res.replay), abs=1e-9)
    new_nodes = [nid for nid, _roll, _r in res.replay if nid != -1]
    assert new_nodes == list(range(1, len(new_nodes) + 1)) and res.tree_size == len(new_nodes) + 1


def test_replay_is_capped_at_the_recording_limit():
    inst = S.grid_instance(side=6, obstacle_pct=10, seed=1)
    assert len(_run(inst, A.REPLAY_ITERATIONS + 50, seed=1).replay) == A.REPLAY_ITERATIONS


def test_goal_only_reward_without_a_goal_rollout_gives_all_zero_values_and_breadth_first_growth():
    inst = S.grid_instance(side=14, obstacle_pct=0, seed=1)
    res = _run(inst, 250, seed=1, horizon=1)
    assert not res.solved and res.goal_rollouts == 0 and all(w == 0.0 for w in res.tree_reward)
    root_kids = [res.tree_visits[i] for i in range(1, res.tree_size) if res.tree_parent[i] == 0]
    assert max(root_kids) - min(root_kids) <= 1                             # gleiche Werte -> UCB1 besucht gleichmäßig


def test_progress_reward_only_pays_for_non_goal_endings_up_to_one_half():
    inst = S.grid_instance(side=12, obstacle_pct=15, seed=7)
    goal_only = _run(inst, 200, seed=3, reward="goal", horizon=8)
    progress = _run(inst, 200, seed=3, reward="progress", horizon=8)
    assert all(r == 0.0 for _n, _roll, r in goal_only.replay if not goal_only.solved)
    assert any(0.0 < r <= 0.5 for _n, _roll, r in progress.replay) and max(r for _n, _roll, r in progress.replay) <= 1.0


# --- Buchführung und Anytime -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [dict(), dict(rollout="guided"), dict(horizon=5)])
@pytest.mark.parametrize("seed", range(4))
def test_visit_accounting_matches_the_recorded_visits(seed, kw):
    inst = S.grid_instance(side=9, obstacle_pct=20, seed=seed)
    res = _run(inst, 500, seed=seed, **kw)
    assert res.visits_total == pytest.approx(res.state_visits.sum() + res.rollout_steps)
    assert res.rollout_steps == pytest.approx(res.rollout_visits.sum())
    assert res.tree_size <= res.iterations + 1 and res.goal_rollouts <= res.iterations


@pytest.mark.parametrize("seed", range(6))
def test_anytime_sequence_is_strictly_improving_and_consistent_with_the_result(seed):
    inst = S.grid_instance(side=9, obstacle_pct=15, seed=seed)
    res = _run(inst, 800, seed=seed)
    if not res.solved:
        assert res.anytime == [] and res.first_solution_iter == 0
        return
    its, vis, costs = zip(*res.anytime)
    assert all(b >= a for a, b in zip(its, its[1:]))
    assert all(b >= a for a, b in zip(vis, vis[1:])) and all(b < a - 1e-12 for a, b in zip(costs, costs[1:]))
    assert costs[-1] == pytest.approx(res.cost) and (res.first_solution_iter, res.first_solution_visits) == (its[0], vis[0]) and vis[-1] <= res.visits_total


def test_more_iterations_never_worsen_the_best_cost_for_the_same_seed():
    inst = S.grid_instance(side=9, obstacle_pct=15, seed=4)
    costs = [_run(inst, k, seed=2).cost for k in (100, 300, 900, 2700)]
    assert all(b <= a + 1e-9 for a, b in zip(costs, costs[1:]))


# --- Sonderfälle -------------------------------------------------------------------------------------------------------------------------


def test_start_equals_goal_and_unreachable_goal():
    inst = S.grid_instance(side=5, obstacle_pct=0, seed=1)
    trivial = A.mcts_search(inst.graph, inst.start, inst.start, 50, seed=0)
    assert trivial.path == [inst.start] and trivial.cost == 0.0 and trivial.solved
    graph = G.from_edges(4, [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 1, 1.0), (2, 3, 1.0)])
    res = A.mcts_search(graph, 0, 3, 300, seed=0)
    assert not res.solved and res.failed and res.path == [] and res.cost == INF and res.goal_rollouts == 0


@pytest.mark.parametrize("rollout", A.ROLLOUT_MODES)
def test_chain_graph_is_always_solved_with_the_unit_reward(rollout):
    n = 6
    graph = G.from_edges(n, [(float(i), 0.0) for i in range(n)], [(i, i + 1, 1.0) for i in range(n - 1)])
    res = A.mcts_search(graph, 0, n - 1, 60, seed=0, rollout=rollout)
    assert res.path == list(range(n)) and res.cost == n - 1 and res.first_solution_iter == 1
    assert max(r for _n, _roll, r in res.replay) == pytest.approx(1.0)      # Luftlinie == Weg


def test_small_open_grid_with_a_generous_budget_converges_to_the_optimum():
    hits = 0
    for seed in range(20):
        inst = S.grid_instance(side=4, obstacle_pct=0, seed=seed)
        opt = A.uniform_cost_search(inst.graph, inst.start, inst.goal).cost
        hits += _run(inst, 1500, seed=seed, c=4.0).cost <= opt + 1e-9
    assert hits >= 19


def test_zero_iterations_return_an_unsolved_root_only_tree():
    inst = S.grid_instance(side=6, obstacle_pct=10, seed=1)
    res = _run(inst, 0)
    assert not res.solved and res.tree_size == 1 and res.visits_total == 0 and res.replay == []


# --- handgebaute Falle ----------------------------------------------------------------------------------------------------------------------


def test_trap_gbfs_is_fooled_but_mcts_without_a_heuristic_finds_the_optimum():
    inst = S.trap_instance()
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    assert A.greedy_best_first(inst.graph, inst.start, inst.goal).cost > ucs.cost + 1e-6
    for seed in range(10):
        res = _run(inst, 200, seed=seed)
        assert res.cost == pytest.approx(ucs.cost, abs=1e-9) and res.path == ucs.path


# --- Kopie treu und geerbte Eigenschaften ---------------------------------------------------------------------------------------------------


def test_copied_comparison_searches_reproduce_the_sibling_numbers():
    inst = S.grid_instance(side=12, obstacle_pct=15, seed=35)
    a = A.a_star(inst.graph, inst.start, inst.goal)
    u = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    assert a.cost == pytest.approx(175.90, abs=0.01) and a.expansions == 82 and u.cost == pytest.approx(a.cost, abs=EPS)


def test_inherited_heuristic_is_admissible_and_a_star_is_optimal():
    for seed in range(10):
        inst = S.grid_instance(side=7, obstacle_pct=20, seed=seed)
        h = A.heuristic(inst.graph.xy, inst.goal)
        ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
        for node in range(inst.graph.n):
            assert h[node] <= A.uniform_cost_search(inst.graph, node, inst.goal).cost + EPS
        assert A.a_star(inst.graph, inst.start, inst.goal).cost == pytest.approx(ucs.cost, abs=EPS)
