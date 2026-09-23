"""Auswertung: Analysis-Felder gegen unabhängige Neuberechnung, run_config/sweep/ablation/anytime_curves auf kleinen Populationen."""

import math

import pytest

import mcts_algorithm as A
import mcts_constants as C
import mcts_evaluation as ev
import mcts_scenario as S

SMALL = dict(instance_seeds=C.SWEEP_SEEDS[:2], chain_seeds=(0, 1))


def test_default_settings_come_from_the_constants_and_are_members_of_the_controls():
    s = ev.Settings()
    assert (s.iterations, s.c, s.horizon, s.seed, s.chain_seed) == (C.DEFAULT_ITERATIONS, C.DEFAULT_C, C.DEFAULT_HORIZON, C.DEFAULT_SEED, C.DEFAULT_CHAIN_SEED)
    assert s.iterations in C.ITERATIONS and s.c in C.C_OPTIONS and s.horizon in C.HORIZON_OPTIONS and s.rollout in C.ROLLOUTS and s.reward in C.REWARDS
    assert set(C.ITERATION_OPTIONS) <= set(C.ITERATIONS)


@pytest.mark.parametrize("seed", [35, 100001])
def test_analysis_fields_match_an_independent_recomputation(seed):
    s = ev.Settings(seed=seed, iterations=800)
    a = ev.analyse(s)
    inst = S.grid_instance(s.side, s.obstacle_pct, seed)
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    mcts = A.mcts_search(inst.graph, inst.start, inst.goal, s.iterations, seed=s.chain_seed, c=s.c, horizon=s.horizon)
    assert a.ucs.cost == pytest.approx(ucs.cost) and a.mcts.cost == mcts.cost and a.mcts.visits_total == mcts.visits_total
    assert a.gap == pytest.approx(100 * (mcts.cost - ucs.cost) / ucs.cost) and a.gbfs_gap >= 0.0
    assert a.goal_share == pytest.approx(100 * mcts.goal_rollouts / 800) and a.optimal == (mcts.cost <= ucs.cost + 1e-9)


def test_visits_to_thresholds_come_from_the_anytime_sequence():
    a = ev.analyse(ev.Settings(network="trap", iterations=200))
    assert a.gap == 0.0 and a.optimal and a.visits_to_optimum == a.mcts.anytime[-1][1] and a.visits_to_near <= a.visits_to_optimum
    b = ev.analyse(ev.Settings(iterations=250, horizon=10))
    assert not b.mcts.solved and math.isnan(b.gap) and not b.optimal and b.visits_to_near == ev.INF and b.visits_to_optimum == ev.INF


def test_references_are_shared_and_only_the_trap_has_labels():
    a, b = ev.analyse(ev.Settings(iterations=100)), ev.analyse(ev.Settings(iterations=100, chain_seed=9))
    assert a.ucs is b.ucs and a.astar is b.astar and a.inst is b.inst and a.inst.labels is None
    assert ev.analyse(ev.Settings(network="trap", iterations=50)).inst.labels is not None


def test_run_config_keys_counts_and_ranges_on_a_small_population():
    r = ev.run_config(ev.Settings(iterations=500), **SMALL)
    assert r["n_runs"] == 4
    for key in ("solved_share", "optimal_share", "near_share", "robust_share"):
        assert 0.0 <= r[key] <= 100.0
    assert r["optimal_share"] <= r["near_share"] <= r["solved_share"]
    for key in ("gap", "goal_share", "first_visits", "visits_total", "tree_size", "ucs_expansions", "astar_expansions", "gbfs_gap"):
        assert r[f"{key}_lo"] <= r[key] <= r[f"{key}_hi"] or math.isnan(r[key])


def test_run_config_overrides_settings_and_ignores_the_base_seeds():
    a = ev.run_config(ev.Settings(seed=1, chain_seed=5), iterations=300, **SMALL)
    b = ev.run_config(ev.Settings(seed=999, chain_seed=77, iterations=300), **SMALL)
    assert a == pytest.approx(b, nan_ok=True)


def test_unsolved_runs_are_reported_separately_and_never_mixed_into_the_gap():
    r = ev.run_config(ev.Settings(iterations=250, horizon=10), **SMALL)
    assert r["solved_share"] < 100.0 and (math.isnan(r["gap"]) or r["gap"] >= 0.0)


def test_sweep_returns_one_row_per_value_and_more_iterations_do_not_reduce_the_solved_share():
    rows = ev.sweep("iterations", ev.Settings(), values=(100, 400))
    assert [r["value"] for r in rows] == [100, 400] and rows[1]["solved_share"] >= rows[0]["solved_share"]
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS) == {"iterations", "c", "horizon", "side", "obstacle_pct"}


def test_ablation_has_the_four_variants_and_the_pure_one_uses_neither_switch():
    res = ev.ablation(ev.Settings(iterations=300, side=6))
    assert list(res) == ["pure", "reward", "rollout", "both"] and set(ev.VARIANT_LABELS) == set(res)
    assert ev.VARIANTS["pure"] == ("uniform", "goal") and ev.VARIANTS["both"] == ("guided", "progress")
    assert res["pure"] == pytest.approx(ev.run_config(ev.Settings(iterations=300, side=6)), nan_ok=True)


def test_anytime_curves_are_per_seed_and_consistent():
    s = ev.Settings(iterations=400, side=8)
    curves = ev.anytime_curves(s, n=3)
    assert len(curves) == 3 and curves[0] == ev.analyse(s).mcts.anytime and curves[1] == ev.analyse(ev.Settings(iterations=400, side=8, chain_seed=s.chain_seed + 1)).mcts.anytime
