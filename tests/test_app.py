"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jede aufgezeichnete Iteration, beide Instanz-Typen, erfolglose Läufe, Randwerte, Würfel-Knöpfe,
Permalink-Grenzen, Instanzwechsel, Sweeps und Ablation auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mcts_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="mcts_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label.startswith(label))


def test_default_run_has_no_exception_and_shows_the_four_metrics():
    at = _run()
    _ok(at)
    assert {"MCTS: Lücke", "UCS / A*", "Ziel-Rollouts", "Greedy: Lücke"} <= {m.label for m in at.metric}
    assert _metric(at, "MCTS: Lücke") == "22.80 %" and _metric(at, "UCS / A*") == "126 / 82" and _metric(at, "Greedy") == "22.53 %"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["network_select"] == p["network"] and at.session_state["iterations_select"] == p["iterations"] and at.session_state["c_select"] == p["c"]
    assert at.session_state["horizon_select"] == p["horizon"] and at.session_state["rollout_select"] == p["rollout"] and at.session_state["reward_select"] == p["reward"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs_for_every_network(step):
    for network in C.NETWORKS:
        at = _run(network_select=network, step=step, iterations_select=250)
        _ok(at)
        assert at.get("plotly_chart") and at.session_state["mcts_step"] == step


@pytest.mark.parametrize("step", [1, 2, 3])
def test_unsolved_runs_render_in_every_step_and_are_flagged(step):
    at = _run(horizon_select=10, step=step)
    _ok(at)
    assert _metric(at, "MCTS: Lücke") == "kein Pfad" and any("keine Route" in w.value for w in at.warning)


def test_replay_slider_walks_through_all_recorded_iterations_and_survives_an_instance_change():
    at = _run(step=2, iterations_select=1000)
    _ok(at)
    slider = at.slider(key="mcts_iter")
    assert slider.max == C.REPLAY_ITERATIONS
    slider.set_value(slider.max).run()
    _ok(at)
    assert at.session_state["mcts_iter"] == slider.max
    at.session_state["iterations_select"] = 100                        # weniger aufgezeichnete Iterationen: gespeicherter Wert wird geklemmt
    at.run()
    _ok(at)
    assert at.session_state["mcts_iter"] <= 100


def test_step_two_with_zero_or_one_recorded_iteration_does_not_crash():
    for kw in (dict(iterations_select=100, network_select="trap"), dict(network_select="trap", iterations_select=100, horizon_select=10)):
        _ok(_run(step=2, **kw))


@pytest.mark.parametrize("kw", [
    dict(side_slider=C.SIDE_MIN), dict(side_slider=C.SIDE_MAX, iterations_select=250), dict(obstacle_slider=C.OBSTACLE_MIN), dict(obstacle_slider=C.OBSTACLE_MAX),
    dict(iterations_select=C.ITERATIONS[0]), dict(c_select=C.C_OPTIONS[0]), dict(c_select=C.C_OPTIONS[-1]), dict(horizon_select=C.HORIZON_OPTIONS[0]),
    dict(horizon_select=C.HORIZON_OPTIONS[-1]), dict(rollout_select="guided"), dict(reward_select="progress"), dict(rollout_select="guided", reward_select="progress", iterations_select=250),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))
    _ok(_run(step=2, **kw))
    _ok(_run(step=3, **kw))


def test_the_knowledge_switches_show_a_note_and_change_the_result():
    plain, guided = _run(), _run(rollout_select="guided")
    assert not plain.info and any("Heuristik-Wissen" in i.value for i in guided.info)
    assert _metric(guided, "MCTS: Lücke") == "4.52 %"


def test_dice_buttons_change_the_seeds():
    at = _run()
    old, old_chain = at.session_state["seed_input"], at.session_state["chain_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    next(b for b in at.button if b.label == "🎲 Neuer Lauf").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["chain_seed_input"] != old_chain


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(side="9999", obstacle="9999", iters="999", c="3", horizon="11", rollout="x", reward="x", network="nope", chain="99999999").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert ss["side_slider"] == C.SIDE_MAX and ss["obstacle_slider"] == C.OBSTACLE_MAX and ss["chain_seed_input"] == C.SEED_MAX
    assert (ss["iterations_select"], ss["c_select"], ss["horizon_select"], ss["rollout_select"], ss["reward_select"], ss["network_select"]) == (
        C.DEFAULT_ITERATIONS, C.DEFAULT_C, C.DEFAULT_HORIZON, "uniform", "goal", "grid")


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(iters="250", c="0.25", horizon="20", rollout="guided", reward="progress", network="trap", chain="7").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["iterations_select"], ss["c_select"], ss["horizon_select"], ss["rollout_select"], ss["reward_select"], ss["network_select"], ss["chain_seed_input"]) == (
        250, 0.25, 20, "guided", "progress", "trap", 7)


def test_sidebar_hides_grid_only_controls_for_the_trap_but_keeps_the_mcts_controls():
    at = _run(network_select="trap")
    _ok(at)
    assert not any(s.key == "side_slider" for s in at.slider)
    assert {"iterations_select", "c_select", "horizon_select"} <= {s.key for s in at.select_slider}
    assert any(n.key == "chain_seed_input" for n in at.number_input)


def test_changing_the_instance_while_on_step_two_does_not_crash():
    at = _run(step=2, iterations_select=250)
    _ok(at)
    at.session_state["side_slider"] = C.SIDE_MIN
    at.run()
    _ok(at)
    at.session_state["network_select"] = "trap"
    at.run()
    _ok(at)


@pytest.mark.parametrize("param", ["c", "horizon", "side", "obstacle_pct"])
@pytest.mark.parametrize("metric", ["gap", "shares", "goal", "visits"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(side_slider=6, iterations_select=250, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_iteration_sweep_runs_on_demand():
    at = _run(side_slider=6)
    at.selectbox(key="sweep_select").set_value("iterations").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)


def test_ablation_runs_on_demand_and_shows_one_metric_per_variant():
    at = _run(side_slider=6, iterations_select=250)
    next(b for b in at.button if b.key == "abl_start").click().run()
    _ok(at)
    labels = {m.label for m in at.metric}
    assert {"rein", "+ Reward (h)", "+ Rollout (h)", "beides (h)"} <= labels


def test_anytime_section_and_footer_and_limits_are_present():
    at = _run()
    assert any("Wie schnell wird die Route besser?" in m.value for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Wer setzt an" in m.value and "Ohne Heuristik geht es" in m.value for m in at.markdown)
