"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Buttons (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import mcts_constants as C

NETWORKS = C.NETWORKS


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _int_choice(options):
    def cast(value):
        value = int(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _float_choice(options):
    def cast(value):
        value = float(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


def _choice_from(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "network_select": SettingSpec("network", _choice_from(NETWORKS), "grid"),
    "side_slider": SettingSpec("side", int, C.DEFAULT_SIDE, C.SIDE_MIN, C.SIDE_MAX),
    "obstacle_slider": SettingSpec("obstacle", int, C.DEFAULT_OBSTACLE, C.OBSTACLE_MIN, C.OBSTACLE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "chain_seed_input": SettingSpec("chain", int, C.DEFAULT_CHAIN_SEED, 0, C.SEED_MAX),
    "iterations_select": SettingSpec("iters", _int_choice(C.ITERATIONS), C.DEFAULT_ITERATIONS),
    "c_select": SettingSpec("c", _float_choice(C.C_OPTIONS), C.DEFAULT_C),
    "horizon_select": SettingSpec("horizon", _int_choice(C.HORIZON_OPTIONS), C.DEFAULT_HORIZON),
    "rollout_select": SettingSpec("rollout", _choice_from(C.ROLLOUTS), "uniform"),
    "reward_select": SettingSpec("reward", _choice_from(C.REWARDS), "goal"),
}
PRESET_KEYS = {"network": "network_select", "side": "side_slider", "obstacle_pct": "obstacle_slider", "seed": "seed_input", "chain_seed": "chain_seed_input",
               "iterations": "iterations_select", "c": "c_select", "horizon": "horizon_select", "rollout": "rollout_select", "reward": "reward_select"}
STEPS = {"side_slider": 1, "obstacle_slider": C.OBSTACLE_STEP}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            lo = SETTING_SPECS[key].lo
            st.session_state[key] = int(lo + round((st.session_state[key] - lo) / step) * step)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def randomize_chain_seed():
    st.session_state["chain_seed_input"] = random.randint(0, C.SEED_MAX)
