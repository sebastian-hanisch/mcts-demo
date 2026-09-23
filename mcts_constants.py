"""Konstanten der MCTS-Demo: Raster-Geometrie (wortgleich zu den Geschwistern), Regler, gemessene Werte, Presets."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
JITTER = 0.35                    # Lageabweichung je Zelle, Anteil des Zellenabstands

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE, SIDE_STEP = 4, 25, 12, 1     # Rastergröße (Zellen je Kante)
OBSTACLE_MIN, OBSTACLE_MAX, DEFAULT_OBSTACLE, OBSTACLE_STEP = 0, 40, 15, 5   # Prozent gesperrte Zellen
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 1

ITERATIONS = (100, 250, 1000, 4000, 16000, 64000)      # Regler; MUSS die Voreinstellung enthalten (st.select_slider snappt sonst still)
DEFAULT_ITERATIONS = 4000
ITERATION_OPTIONS = (250, 1000, 4000, 16000)           # Sweep
C_OPTIONS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_C = 1.0
HORIZON_OPTIONS = (10, 20, 50, 100, 200)
DEFAULT_HORIZON = 100
SCALING_SIDES = (6, 10, 14, 18, 22)
OBSTACLE_SWEEP = (0, 10, 20, 30, 40)
SWEEP_SEEDS = tuple(range(100000, 100005))             # 5 feste Instanzen (Konvention der Linie)
CHAIN_SEEDS = tuple(range(6))                          # 6 MCTS-Seeds je Instanz -> 30 Läufe je Konfiguration
REPLAY_ITERATIONS = 500                                # so viele Iterationen werden für die Wiedergabe aufgezeichnet (= mcts_algorithm.REPLAY_ITERATIONS)

NETWORKS = ("grid", "trap")
NETWORK_LABELS = {"grid": "Raster", "trap": "Heuristik-Falle (handgebaut)"}
ROLLOUTS = ("uniform", "guided")
ROLLOUT_LABELS = {"uniform": "gleichverteilt (ohne Heuristik)", "guided": "geführt durch h (Softmax)"}
REWARDS = ("goal", "progress")
REWARD_LABELS = {"goal": "nur Ziel (ohne Heuristik)", "progress": "Ziel + Fortschritt (mit h)"}

# --- Gemessene Werte (MEDIAN über 5 feste Instanzen, Seeds 100000-100004, x 6 MCTS-Seeds = 30 Läufe; Rastergröße 12, Hindernisdichte 15 %,
# --- 4000 Iterationen, c = 1, Horizont 100, sofern nicht anders angegeben; 2026-09-23/24, alle Werte über ev.run_config/ev.sweep/ev.ablation
# --- nachgerechnet, s. tests/test_claims.py). Lücke nur über Läufe MIT Lösung; Erfolgs-, Nah- (Lücke <= 5 %) und Optimal-Anteil über ALLE Läufe.
# --- Einheit "Knotenbesuche" (MCTS) ist NICHT die "Expansion" von Uniform-Cost/A* - nur der Größenordnung nach vergleichbar. ---
# ZENTRALE FRAGE 1 - findet MCTS ohne Heuristik überhaupt eine Route, und wie gut? Reines MCTS (gleichverteilter Rollout, Reward nur am Ziel)
#   findet in 100 % der Läufe eine Route, aber in KEINEM Lauf die optimale und im Median 17.1 % über dem Optimum (nur 0 % der Läufe <= 5 %).
#   Uniform-Cost findet das Optimum mit 121 Expansionen (A*: 93), MCTS braucht für die Lücke 17 % im Median ~81 000 Knotenbesuche insgesamt.
#   Lücke über Iterationen 250/1000/4000/16000: 40.0/28.2/17.1/8.5 %, Anteil <= 5 %: 0/0/0/20 % - sie fällt, aber langsam.
# ZENTRALE FRAGE 2 - spärlicher Reward? Anteil der Iterationen, die das Ziel erreichen, über die Rastergröße 6/10/14/18/22: 95.3/5.1/2.6/0.3/0.2 %;
#   Lücke 5.6/2.0/21.5/41.7/67.8 % (bei Größe 22 finden 93 % der Läufe überhaupt eine Route). Zu kurzer Horizont (10) bei Größe 12: nur 40 % der
#   Läufe finden eine Route; ab 20 ist der Horizont ohne große Wirkung (Lücke 17.6/19.2/17.1/17.1 % für 20/50/100/200).
# ZENTRALE FRAGE 3 - wie viel des Erfolgs ist eingeschmuggeltes h? Ablation (Größe 12, 4000 Iterationen): Lücke im Median / Anteil <= 5 %:
#   rein 17.1 % / 0 %; + Fortschritts-Reward 12.4 % / 10 %; + geführter Rollout 2.4 % / 83 %; beides 3.2 % / 80 %. Bei 16000 Iterationen 8.5/9.1/1.3/0.5 %
#   (Anteil <= 5 %: 20/23/93/100 %). Größe 6: rein 5.6 % (23 % optimal), geführt 0.0 % (100 % optimal); Größe 22: rein 67.8 %, geführt 7.0 %. Der
#   geführte Rollout ist der Hebel; Greedy Best-First hat auf denselben Instanzen 14.7 % Lücke - MCTS mit geführtem Rollout ist also "Greedy mit Zufall
#   plus Baum". Ein Fortschritts-Reward allein bringt wenig.
# ZENTRALE FRAGE 4 - UCB-Konstante c: auf dem 4x4-Raster ohne Hindernisse (1000 Iterationen, 10 Instanzen x 3 Seeds) finden c = 0.25/1/4/16 in
#   10/67/100/100 % der Läufe das Optimum (Lücke im Median 12.9/0.0/0.0/0.0 %); auf Größe 5/6 flacher (c = 4: 67/7 % optimal). Auf Größe 12 ist c fast
#   wirkungslos (Lücke 13.3/16.5/17.1/17.6/14.6/16.8/16.7 % für c = 0.25 bis 16). Vermutete Ursache (nicht isoliert): der Reward d0/L liegt eng beieinander,
#   UCB1 braucht dann ein großes c, um Alternativen weiter zu erkunden.
# HINDERNISSE 0/10/20/30/40 %: Lücke 20.2/17.2/18.1/15.8/7.4 % - MCTS wird mit vielen Hindernissen BESSER (Optimal-Anteil 10 % bei 40 %, Anteil <= 5 % dort 40 %).
# HANDGEBAUTE FALLE (8 Knoten, Greedy fällt auf den Köder herein: 7.5 % Lücke): MCTS ohne Heuristik findet den Umweg (Lücke 0.0 %) bereits mit
#   4 Knotenbesuchen bis zur ersten optimalen Route (Seed 1, 250 Iterationen; in allen 10 getesteten Seeds optimal); Uniform-Cost braucht 8, A* 7 Expansionen.

PRESETS = {
    "Standardfall (Voreinstellung)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "chain_seed": 1, "iterations": 4000, "c": 1.0, "horizon": 100, "rollout": "uniform", "reward": "goal"},
    "Geführter Rollout (mit h)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "chain_seed": 1, "iterations": 4000, "c": 1.0, "horizon": 100, "rollout": "guided", "reward": "goal"},
    "Fortschritts-Reward (mit h)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "chain_seed": 1, "iterations": 4000, "c": 1.0, "horizon": 100, "rollout": "uniform", "reward": "progress"},
    "Große Karte (Größe 22)": {"network": "grid", "side": 22, "obstacle_pct": 15, "seed": 35, "chain_seed": 1, "iterations": 4000, "c": 1.0, "horizon": 100, "rollout": "uniform", "reward": "goal"},
    "Kleines Raster, c zu klein": {"network": "grid", "side": 4, "obstacle_pct": 0, "seed": 1, "chain_seed": 1, "iterations": 1000, "c": 0.25, "horizon": 100, "rollout": "uniform", "reward": "goal"},
    "Heuristik-Falle": {"network": "trap", "side": 12, "obstacle_pct": 15, "seed": 35, "chain_seed": 1, "iterations": 250, "c": 1.0, "horizon": 100, "rollout": "uniform", "reward": "goal"},
    "Viele Hindernisse (40 %)": {"network": "grid", "side": 12, "obstacle_pct": 40, "seed": 35, "chain_seed": 1, "iterations": 4000, "c": 1.0, "horizon": 100, "rollout": "uniform", "reward": "goal"},
    "Horizont zu kurz": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "chain_seed": 1, "iterations": 4000, "c": 1.0, "horizon": 10, "rollout": "uniform", "reward": "goal"},
    "Wenige Iterationen (250)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "chain_seed": 1, "iterations": 250, "c": 1.0, "horizon": 100, "rollout": "uniform", "reward": "goal"},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Größe 12, 15 % Hindernisse, 4000 Iterationen, reines MCTS: Seed 35 findet eine Route mit 22.8 % Lücke (im Median über 30 Läufe 17.1 %, nie optimal), nach etwa 106 000 Knotenbesuchen insgesamt. Uniform-Cost findet das Optimum (175.9 km) mit 126 Expansionen, A* mit 82.",
    "Geführter Rollout (mit h)": "Dieselbe Instanz, aber der Rollout zieht per Softmax über die Luftlinie zum Ziel: Lücke 4.5 % statt 22.8 % (im Median 2.4 % statt 17.1 %, 83 % der Läufe unter 5 %). Das ist Heuristik-Wissen - kein MCTS ohne h mehr.",
    "Fortschritts-Reward (mit h)": "Der Reward belohnt auch Läufe ohne Ziel mit bis zu 0.5 je nach Luftlinie am Ende: Lücke 19.5 % (im Median 12.4 %) - viel weniger Hebel als der geführte Rollout.",
    "Große Karte (Größe 22)": "Größe 22: nur 0.1 % der Iterationen erreichen das Ziel, die erste Route kommt nach etwa 28 000 Knotenbesuchen, die Lücke liegt bei 84.5 % (im Median 67.8 %; 7 % der Läufe finden gar keine Route). Spärlicher Reward.",
    "Kleines Raster, c zu klein": "4x4-Raster ohne Hindernisse, 1000 Iterationen, c = 0.25: Lücke 19.7 %, MCTS bleibt an der ersten guten Route hängen (über 5 Instanzen x 6 Seeds sind nur 13 % der Läufe optimal). Mit c = 4 findet derselbe Lauf das Optimum, in 100 % der 30 Läufe.",
    "Heuristik-Falle": "Handgebauter Graph mit 8 Knoten: Greedy Best-First folgt der Heuristik in eine Sackgasse (7.5 % Lücke). MCTS ohne Heuristik findet den kürzeren Umweg schon beim 4. Knotenbesuch - für einen Graphen, den Uniform-Cost mit 8 Expansionen löst.",
    "Viele Hindernisse (40 %)": "Mit 40 % Hindernissen verengen sich die Wege: MCTS wird besser (im Median Lücke 7.4 % statt 17.1 %, 40 % der Läufe unter 5 %), Seed 35 hat 14.3 %. Uniform-Cost braucht dort nur 88 Expansionen.",
    "Horizont zu kurz": "Der Rollout-Horizont 10 ist kürzer als jeder Weg zum Ziel: kein einziger Rollout erreicht es, Seed 35 findet in 4000 Iterationen keine Route (im Median über 30 Läufe finden nur 40 % eine).",
    "Wenige Iterationen (250)": "Nur 250 Iterationen (etwa 7000 Knotenbesuche): Lücke 36.8 % (im Median 40.0 %), keine Route unter 5 %.",
}
# Beobachtete Spannweite des MEDIANS der Lücke (%) über die 5 festen Instanzen x 6 Seeds (mit Sicherheitsabstand), nur für die Raster-Presets
# mit Lösung in der Mehrzahl der Läufe; die Falle ist ein fester Graph, ihre Zahlen stehen als Tests.
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": (10.0, 25.0),
    "Geführter Rollout (mit h)": (0.5, 6.0),
    "Fortschritts-Reward (mit h)": (7.0, 18.0),
    "Große Karte (Größe 22)": (50.0, 85.0),
    "Kleines Raster, c zu klein": (5.0, 18.0),
    "Viele Hindernisse (40 %)": (3.0, 12.0),
    "Wenige Iterationen (250)": (28.0, 52.0),
}
