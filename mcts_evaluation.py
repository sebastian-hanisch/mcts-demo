"""Auswertung: was leistet Monte Carlo Tree Search OHNE handgemachte Heuristik auf dem Pfadproblem? Ein MCTS-Lauf (Iterationen, UCB-Konstante c,
Rollout-Horizont, Rollout-Politik, Reward) gegen Uniform-Cost-Search (das Optimum und die ehrliche h-freie Basis), A* und Greedy Best-First Search auf
demselben Graphen. MCTS hat Zufall im Kern: Kennzahlen laufen über 5 feste Instanzen (Seeds 100000-100004) x 6 Ketten-Seeds, Median mit
10./90. Perzentil; Läufe OHNE Lösung werden getrennt ausgewiesen und nie in Lücken gemischt.

- **Lücke** = 100 * (Kosten der besten gefundenen Route - Optimum) / Optimum, nur über Läufe mit Lösung; **Erfolgsquote** = Anteil ALLER Läufe mit Lösung;
  **Optimal-Anteil** = Anteil ALLER Läufe mit Lücke 0; **nah** = Anteil ALLER Läufe mit Lücke <= 5 %.
- **Knotenbesuche** (Aufwandseinheit von MCTS): jede berührte Baum-Stufe und jeder Rollout-Schritt zählt 1 - NICHT dasselbe wie eine Expansion von UCS/A*
  (die Zahlen sind nur der Größenordnung nach vergleichbar; die Einheit steht überall dabei).
- **Ziel-Rollouts** = Anteil der Iterationen, die das Ziel erreichen (Maß für den spärlichen Reward)."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import mcts_algorithm as A
import mcts_constants as C
import mcts_scenario as S

INF = float("inf")
NEAR_GAP = 5.0


@dataclass(frozen=True)
class Settings:
    network: str = "grid"
    side: int = C.DEFAULT_SIDE
    obstacle_pct: int = C.DEFAULT_OBSTACLE
    seed: int = C.DEFAULT_SEED                  # Instanz-Seed
    chain_seed: int = C.DEFAULT_CHAIN_SEED      # Seed des MCTS-Laufs
    iterations: int = C.DEFAULT_ITERATIONS
    c: float = C.DEFAULT_C
    horizon: int = C.DEFAULT_HORIZON
    rollout: str = "uniform"
    reward: str = "goal"


@lru_cache(maxsize=512)
def _grid(side, obstacle_pct, seed):
    return S.grid_instance(side, obstacle_pct, seed)


def instance_of(settings):
    if settings.network == "grid":
        return _grid(settings.side, settings.obstacle_pct, settings.seed)
    return S.trap_instance()


@lru_cache(maxsize=512)
def _references(network, side, obstacle_pct, seed):
    inst = instance_of(Settings(network=network, side=side, obstacle_pct=obstacle_pct, seed=seed))
    g, s, t = inst.graph, inst.start, inst.goal
    return A.uniform_cost_search(g, s, t), A.a_star(g, s, t), A.greedy_best_first(g, s, t)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    mcts: A.MCTSResult
    ucs: A.SearchResult
    astar: A.SearchResult
    gbfs: A.SearchResult

    def _gap(self, cost):
        return 100.0 * (cost - self.ucs.cost) / self.ucs.cost

    @property
    def gap(self):
        return self._gap(self.mcts.cost) if self.mcts.solved else float("nan")

    @property
    def robust_gap(self):
        return self._gap(self.mcts.robust_cost) if self.mcts.robust_path else float("nan")

    @property
    def gbfs_gap(self):
        return self._gap(self.gbfs.cost)

    @property
    def optimal(self):
        return self.mcts.solved and self.mcts.cost <= self.ucs.cost + 1e-9

    @property
    def goal_share(self):
        return 100.0 * self.mcts.goal_rollouts / max(1, self.mcts.iterations)

    @property
    def visits_to_near(self):
        """Knotenbesuche bis die beste Route eine Lücke <= 5 % hat (INF, falls nie)."""
        for _it, visits, cost in self.mcts.anytime:
            if self._gap(cost) <= NEAR_GAP:
                return visits
        return INF

    @property
    def visits_to_optimum(self):
        for _it, visits, cost in self.mcts.anytime:
            if cost <= self.ucs.cost + 1e-9:
                return visits
        return INF


def analyse(settings):
    inst = instance_of(settings)
    mcts = A.mcts_search(inst.graph, inst.start, inst.goal, settings.iterations, seed=settings.chain_seed, c=settings.c, horizon=settings.horizon,
                         rollout=settings.rollout, reward=settings.reward)
    ucs, astar, gbfs = _references(settings.network, settings.side, settings.obstacle_pct, settings.seed)
    return Analysis(settings, inst, mcts, ucs, astar, gbfs)


# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if not np.isnan(v) and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def run_config(base, instance_seeds=C.SWEEP_SEEDS, chain_seeds=C.CHAIN_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = [analyse(replace(s0, seed=i, chain_seed=k)) for i in instance_seeds for k in chain_seeds]
    n = len(rows)
    out = {
        "n_runs": n,
        "solved_share": 100.0 * sum(r.mcts.solved for r in rows) / n,
        "optimal_share": 100.0 * sum(r.optimal for r in rows) / n,
        "near_share": 100.0 * sum(r.mcts.solved and r.gap <= NEAR_GAP for r in rows) / n,
        "robust_share": 100.0 * sum(bool(r.mcts.robust_path) for r in rows) / n,
    }
    for key, values in (
        ("gap", [r.gap for r in rows]),
        ("robust_gap", [r.robust_gap for r in rows]),
        ("goal_share", [r.goal_share for r in rows]),
        ("first_visits", [float(r.mcts.first_solution_visits) if r.mcts.solved else float("nan") for r in rows]),
        ("visits_near", [r.visits_to_near for r in rows]),
        ("visits_optimum", [r.visits_to_optimum for r in rows]),
        ("visits_total", [float(r.mcts.visits_total) for r in rows]),
        ("tree_size", [float(r.mcts.tree_size) for r in rows]),
        ("ucs_expansions", [float(r.ucs.expansions) for r in rows]),
        ("astar_expansions", [float(r.astar.expansions) for r in rows]),
        ("gbfs_gap", [r.gbfs_gap for r in rows]),
    ):
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats(values)
    return out


SWEEP_VALUES = {"iterations": C.ITERATION_OPTIONS, "c": C.C_OPTIONS, "horizon": C.HORIZON_OPTIONS, "side": C.SCALING_SIDES, "obstacle_pct": C.OBSTACLE_SWEEP}
SWEEP_LABELS = {"iterations": "Iterationen", "c": "UCB-Konstante c", "horizon": "Rollout-Horizont", "side": "Rastergröße (Seitenlänge)", "obstacle_pct": "Hindernisdichte (%)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


VARIANTS = {"pure": ("uniform", "goal"), "reward": ("uniform", "progress"), "rollout": ("guided", "goal"), "both": ("guided", "progress")}
VARIANT_LABELS = {"pure": "rein (ohne Heuristik)", "reward": "+ Fortschritts-Reward (h)", "rollout": "+ geführter Rollout (h)", "both": "beides (h)"}


def ablation(base=Settings()):
    """Dieselbe Konfiguration mit und ohne eingeschmuggeltes Heuristik-Wissen: {Variante: run_config}."""
    return {v: run_config(base, rollout=r, reward=w) for v, (r, w) in VARIANTS.items()}


def anytime_curves(settings, n=6):
    """Anytime-Kurven ([(Iteration, Knotenbesuche, beste Kosten)]) von `n` Ketten-Seeds ab dem Seed der Einstellungen - für EINE Instanz."""
    inst = instance_of(settings)
    return [A.mcts_search(inst.graph, inst.start, inst.goal, settings.iterations, seed=settings.chain_seed + k, c=settings.c, horizon=settings.horizon,
                          rollout=settings.rollout, reward=settings.reward).anytime for k in range(n)]
