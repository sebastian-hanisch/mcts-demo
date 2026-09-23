"""Jede Zahl aus README, App-Texten und mcts_constants.py, nachgerechnet über die echten Auswertungsfunktionen (ev.sweep/ev.run_config/ev.ablation) auf den 5 festen
Instanzen (Seeds 100000-100004) x 6 MCTS-Seeds, Rastergröße 12, Hindernisdichte 15 %, 4000 Iterationen, c = 1, Horizont 100, sofern nicht anders angegeben.
Toleranzen sind großzügig gegenüber Rundung (alles ist deterministisch), aber enger als jede Aussage."""

from functools import lru_cache

import pytest

import mcts_algorithm as A
import mcts_constants as C
import mcts_evaluation as ev
import mcts_scenario as S

BASE = ev.Settings()


@lru_cache(maxsize=None)
def _rows(param):
    return ev.sweep(param, BASE)


@lru_cache(maxsize=None)
def _abl(iterations, side):
    return ev.ablation(ev.Settings(iterations=iterations, side=side))


def _col(param, key):
    return [r[key] for r in _rows(param)]


def _at(param, value):
    return next(r for r in _rows(param) if r["value"] == value)


def _close(values, expected, abs_tol):
    assert len(values) == len(expected)
    for v, e in zip(values, expected):
        assert v == pytest.approx(e, abs=abs_tol), (values, expected)


# --- Frage 1: findet reines MCTS Routen, und wie gut? --------------------------------------------------------------------------------------


def test_pure_mcts_at_the_default_finds_a_route_every_time_but_never_the_optimum():
    r = _at("iterations", 4000)
    assert r["solved_share"] == 100.0 and r["optimal_share"] == 0.0 and r["near_share"] == 0.0 and r["gap"] == pytest.approx(17.1, abs=0.1)
    assert r["ucs_expansions"] == 121.0 and r["astar_expansions"] == 93.0 and r["visits_total"] == pytest.approx(81147, rel=0.01)


def test_gap_falls_with_iterations_but_slowly():
    _close(_col("iterations", "gap"), [40.0, 28.2, 17.1, 8.5], 0.15)
    _close(_col("iterations", "near_share"), [0.0, 0.0, 0.0, 20.0], 0.1)
    assert _col("iterations", "value") == [250, 1000, 4000, 16000]


# --- Frage 2: spärlicher Reward ----------------------------------------------------------------------------------------------------------------


def test_goal_rollout_share_and_gap_over_the_grid_size():
    _close(_col("side", "goal_share"), [95.3, 5.1, 2.6, 0.3, 0.2], 0.1)
    _close(_col("side", "gap"), [5.6, 2.0, 21.5, 41.7, 67.8], 0.15)
    assert _at("side", 22)["solved_share"] == pytest.approx(93.3, abs=0.1) and all(r["solved_share"] == 100.0 for r in _rows("side")[:4])


def test_short_horizon_finds_routes_in_only_forty_percent_and_longer_horizons_are_alike():
    assert _at("horizon", 10)["solved_share"] == 40.0 and all(_at("horizon", h)["solved_share"] == 100.0 for h in (20, 50, 100, 200))
    _close([_at("horizon", h)["gap"] for h in (20, 50, 100, 200)], [17.6, 19.2, 17.1, 17.1], 0.15)


# --- Frage 3: eingeschmuggeltes h -------------------------------------------------------------------------------------------------------------


def test_ablation_at_the_default():
    res = _abl(4000, 12)
    _close([res[v]["gap"] for v in res], [17.1, 12.4, 2.4, 3.2], 0.15)
    _close([res[v]["near_share"] for v in res], [0.0, 10.0, 83.3, 80.0], 0.1)
    assert all(res[v]["solved_share"] == 100.0 for v in res) and res["rollout"]["goal_share"] > 70 > 5 > res["pure"]["goal_share"]
    assert res["pure"]["gbfs_gap"] == pytest.approx(14.7, abs=0.1)


def test_ablation_with_more_iterations():
    res = _abl(16000, 12)
    _close([res[v]["gap"] for v in res], [8.5, 9.1, 1.3, 0.5], 0.15)
    _close([res[v]["near_share"] for v in res], [20.0, 23.3, 93.3, 100.0], 0.1)


def test_ablation_on_small_and_large_grids():
    small, large = _abl(4000, 6), _abl(4000, 22)
    assert small["pure"]["gap"] == pytest.approx(5.6, abs=0.1) and small["pure"]["optimal_share"] == pytest.approx(23.3, abs=0.1)
    assert small["rollout"]["gap"] == 0.0 and small["rollout"]["optimal_share"] == 100.0
    assert large["pure"]["gap"] == pytest.approx(67.8, abs=0.15) and large["rollout"]["gap"] == pytest.approx(7.0, abs=0.15)


# --- Frage 4: UCB-Konstante c ------------------------------------------------------------------------------------------------------------------


def test_c_barely_matters_on_size_twelve():
    _close(_col("c", "gap"), [13.3, 16.5, 17.1, 17.6, 14.6, 16.8, 16.7], 0.15)
    assert _col("c", "value") == list(C.C_OPTIONS)


def test_c_matters_a_lot_on_tiny_open_grids():
    seeds = tuple(range(100000, 100010))
    base = ev.Settings(side=4, obstacle_pct=0, iterations=1000)
    opt = [ev.run_config(base, instance_seeds=seeds, chain_seeds=(0, 1, 2), c=c) for c in (0.25, 1.0, 4.0, 16.0)]
    _close([r["optimal_share"] for r in opt], [10.0, 66.7, 100.0, 100.0], 0.1)
    _close([r["gap"] for r in opt], [12.9, 0.0, 0.0, 0.0], 0.1)
    for side, expected in ((5, 66.7), (6, 6.7)):
        r = ev.run_config(ev.Settings(side=side, obstacle_pct=0, iterations=1000), instance_seeds=seeds, chain_seeds=(0, 1, 2), c=4.0)
        assert r["optimal_share"] == pytest.approx(expected, abs=0.1)


# --- Hindernisse und Falle --------------------------------------------------------------------------------------------------------------------


def test_obstacle_sweep_mcts_gets_better_with_dense_obstacles():
    _close(_col("obstacle_pct", "gap"), [20.2, 17.2, 18.1, 15.8, 7.4], 0.15)
    r = _at("obstacle_pct", 40)
    assert r["optimal_share"] == 10.0 and r["near_share"] == 40.0


def test_trap_mcts_finds_the_detour_for_every_seed_while_greedy_is_fooled():
    inst = S.trap_instance()
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    for seed in range(10):
        assert A.mcts_search(inst.graph, inst.start, inst.goal, 200, seed=seed).cost == pytest.approx(ucs.cost, abs=1e-9)
    assert A.greedy_best_first(inst.graph, inst.start, inst.goal).cost / ucs.cost - 1 == pytest.approx(0.0747, abs=0.0005)


def test_constants_are_consistent_with_the_measured_sweeps():
    assert list(C.ITERATION_OPTIONS) == _col("iterations", "value") and list(C.SCALING_SIDES) == _col("side", "value")
    assert list(C.HORIZON_OPTIONS) == _col("horizon", "value") and list(C.OBSTACLE_SWEEP) == _col("obstacle_pct", "value")
    assert C.SWEEP_SEEDS[0] == 100000 and len(C.SWEEP_SEEDS) == 5 and len(C.CHAIN_SEEDS) == 6
