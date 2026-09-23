import numpy as np
import pytest

import mcts_algorithm as A
import mcts_graph as G
import mcts_scenario as S


def _bfs_connected(graph, start, goal):
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        if u == goal:
            return True
        for v in graph.neighbors[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return goal in seen


@pytest.mark.parametrize("seed", range(30))
def test_start_and_goal_are_always_connected(seed):
    inst = S.grid_instance(side=10, obstacle_pct=35, seed=seed)
    assert _bfs_connected(inst.graph, inst.start, inst.goal)


@pytest.mark.parametrize("seed", range(10))
def test_edge_weights_match_euclidean_distance_between_endpoints(seed):
    inst = S.grid_instance(side=10, obstacle_pct=20, seed=seed)
    xy = inst.graph.xy
    for u in range(inst.graph.n):
        for v, w in zip(inst.graph.neighbors[u], inst.graph.weights[u]):
            assert w == pytest.approx(float(np.hypot(*(xy[u] - xy[v]))), abs=1e-9)


def test_higher_obstacle_percent_blocks_more_cells_on_average():
    low = [len(S.grid_instance(side=14, obstacle_pct=5, seed=s).blocked_xy) for s in range(10)]
    high = [len(S.grid_instance(side=14, obstacle_pct=35, seed=s).blocked_xy) for s in range(10)]
    assert np.mean(high) > np.mean(low)


def test_zero_obstacle_percent_leaves_every_cell_open():
    inst = S.grid_instance(side=10, obstacle_pct=0, seed=1)
    assert len(inst.blocked_xy) == 0 and inst.graph.n == 100


def test_all_edge_weights_are_positive():
    inst = S.grid_instance(side=12, obstacle_pct=30, seed=7)
    assert all(w > 0 for u in range(inst.graph.n) for w in inst.graph.weights[u])


def test_trap_instance_has_the_expected_shape_and_a_connected_start_goal():
    inst = S.trap_instance()
    assert inst.graph.n == 8 and len(inst.labels) == 8 and inst.labels[inst.start] == "Start" and inst.labels[inst.goal] == "Ziel"
    assert _bfs_connected(inst.graph, inst.start, inst.goal) and len(inst.blocked_xy) == 0
    assert sum(len(n) for n in inst.graph.neighbors) == 2 * len(S.TRAP_LINKS)


def test_trap_optimum_is_the_detour_and_greedy_takes_the_lure():
    inst = S.trap_instance()
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    gbfs = A.greedy_best_first(inst.graph, inst.start, inst.goal)
    assert [inst.labels[n] for n in ucs.path] == ["Start", "Umweg1", "Umweg2", "Ziel"]
    assert "Köder" in [inst.labels[n] for n in gbfs.path] and gbfs.cost > ucs.cost


def test_grid_instances_have_no_labels():
    assert S.grid_instance(side=6, obstacle_pct=10, seed=1).labels is None


def test_path_cost_matches_a_hand_walked_path():
    inst = S.trap_instance()
    idx = {n: i for i, n in enumerate(inst.labels)}
    walked = ["Start", "Umweg1", "Umweg2", "Ziel"]
    expected = sum(float(np.hypot(*(inst.graph.xy[idx[a]] - inst.graph.xy[idx[b]]))) for a, b in zip(walked[:-1], walked[1:]))
    assert G.path_cost(inst.graph, [idx[n] for n in walked]) == pytest.approx(expected)
