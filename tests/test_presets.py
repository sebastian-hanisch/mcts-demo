"""Presets: Vollständigkeit, gültige Werte, der Median der Lücke bleibt bei den Raster-Presets in der gemessenen Spannweite über die 5 festen Instanzen x 6 Seeds
(vollständig deterministisch), und jedes Preset zeigt, was sein Name und sein Hilfetext sagen."""

import pytest

import mcts_algorithm as A
import mcts_constants as C
import mcts_evaluation as ev
import mcts_presets as P


def _settings(p):
    return ev.Settings(network=p["network"], side=p["side"], obstacle_pct=p["obstacle_pct"], seed=p["seed"], chain_seed=p["chain_seed"], iterations=p["iterations"],
                       c=p["c"], horizon=p["horizon"], rollout=p["rollout"], reward=p["reward"])


def _analyse(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def test_every_preset_has_help_and_the_grid_presets_a_band():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 9
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]
    assert set(C.PRESET_EXPECTED_BANDS) <= {n for n, p in C.PRESETS.items() if p["network"] == "grid"}


def test_preset_values_are_valid_members_of_the_controls():
    for p in C.PRESETS.values():
        assert p["network"] in C.NETWORKS and p["iterations"] in C.ITERATIONS and p["c"] in C.C_OPTIONS and p["horizon"] in C.HORIZON_OPTIONS
        assert p["rollout"] in C.ROLLOUTS and p["reward"] in C.REWARDS
        assert C.SIDE_MIN <= p["side"] <= C.SIDE_MAX and C.OBSTACLE_MIN <= p["obstacle_pct"] <= C.OBSTACLE_MAX and 0 <= p["seed"] <= C.SEED_MAX and 0 <= p["chain_seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_grid_preset_median_gap_stays_in_its_measured_band(name):
    lo, hi = C.PRESET_EXPECTED_BANDS[name]
    row = ev.run_config(_settings(C.PRESETS[name]))
    assert lo <= row["gap"] <= hi, row["gap"]


def test_standard_preset_numbers():
    a = _analyse("Standardfall (Voreinstellung)")
    assert a.mcts.solved and a.gap == pytest.approx(22.8, abs=0.05) and (a.ucs.expansions, a.astar.expansions) == (126, 82) and a.ucs.cost == pytest.approx(175.9, abs=0.05)
    assert a.mcts.visits_total == pytest.approx(106000, rel=0.01) and a.gbfs_gap == pytest.approx(22.5, abs=0.05) and not a.optimal


def test_guided_rollout_preset_cuts_the_gap_on_the_same_instance():
    g, s = _analyse("Geführter Rollout (mit h)"), _analyse("Standardfall (Voreinstellung)")
    assert g.gap == pytest.approx(4.5, abs=0.05) and g.gap < s.gap and g.goal_share > 50 > s.goal_share


def test_progress_reward_preset_helps_only_a_little():
    p, s = _analyse("Fortschritts-Reward (mit h)"), _analyse("Standardfall (Voreinstellung)")
    assert p.gap == pytest.approx(19.5, abs=0.05) and p.gap < s.gap and p.gap > 10


def test_big_map_preset_has_a_sparse_reward_and_a_late_first_route():
    a = _analyse("Große Karte (Größe 22)")
    assert a.gap == pytest.approx(84.5, abs=0.05) and a.goal_share < 0.5 and a.mcts.first_solution_visits == pytest.approx(28000, rel=0.02)
    assert ev.run_config(_settings(C.PRESETS["Große Karte (Größe 22)"]))["solved_share"] == pytest.approx(93.3, abs=0.1)


def test_small_grid_preset_is_stuck_with_small_c_and_optimal_with_c_four():
    p = C.PRESETS["Kleines Raster, c zu klein"]
    a = _analyse("Kleines Raster, c zu klein")
    assert a.gap == pytest.approx(19.7, abs=0.05) and p["c"] == 0.25
    better = ev.analyse(ev.Settings(**{**_settings(p).__dict__, "c": 4.0}))
    assert better.gap == 0.0 and better.optimal
    base = _settings(p)
    assert ev.run_config(base)["optimal_share"] == pytest.approx(13.3, abs=0.1) and ev.run_config(base, c=4.0)["optimal_share"] == 100.0


def test_trap_preset_greedy_is_fooled_and_mcts_finds_the_detour_immediately():
    a = _analyse("Heuristik-Falle")
    assert a.gbfs_gap == pytest.approx(7.47, abs=0.01) and a.gap == 0.0 and a.mcts.first_solution_visits <= 12 and a.visits_to_optimum == 4
    assert (a.ucs.expansions, a.astar.expansions) == (8, 7) and a.mcts.visits_total == pytest.approx(1196, abs=1)


def test_dense_obstacle_preset_and_its_median_advantage():
    assert _analyse("Viele Hindernisse (40 %)").gap == pytest.approx(14.3, abs=0.05) and _analyse("Viele Hindernisse (40 %)").ucs.expansions == 88


def test_short_horizon_preset_finds_no_route_for_this_seed():
    a = _analyse("Horizont zu kurz")
    assert not a.mcts.solved and a.goal_share == 0.0 and a.mcts.goal_rollouts == 0
    assert ev.run_config(_settings(C.PRESETS["Horizont zu kurz"]))["solved_share"] == pytest.approx(40.0, abs=0.1)


def test_few_iterations_preset_numbers():
    a = _analyse("Wenige Iterationen (250)")
    assert a.gap == pytest.approx(36.8, abs=0.05) and a.mcts.visits_total == pytest.approx(6735, abs=1) and a.mcts.visits_total < 8000


def test_bounds_and_permalink_constants():
    assert P.bounds("side_slider") == (C.SIDE_MIN, C.SIDE_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("chain_seed_input") == (0, C.SEED_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.REPLAY_ITERATIONS == A.REPLAY_ITERATIONS


def test_permalink_casters_accept_members_and_reject_everything_else():
    c = P.SETTING_SPECS
    assert c["network_select"].caster("trap") == "trap" and c["iterations_select"].caster("1000") == 1000 and c["c_select"].caster("0.25") == 0.25
    assert c["horizon_select"].caster("10") == 10 and c["rollout_select"].caster("guided") == "guided" and c["reward_select"].caster("progress") == "progress"
    for key, bad in (("network_select", "x"), ("iterations_select", "999"), ("c_select", "3"), ("c_select", "x"), ("horizon_select", "11"), ("rollout_select", "x"), ("reward_select", "x")):
        with pytest.raises(ValueError):
            c[key].caster(bad)
