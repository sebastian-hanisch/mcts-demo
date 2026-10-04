"""Monte Carlo Tree Search - geht es auch ohne Heuristik? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes und letztes Stück der Heuristische-Baumsuche-Linie der "Konzepte"-Reihe. Alle sieben Stücke davor teilen eine Annahme: es gibt eine handgemachte
Heuristik h(n). Monte Carlo Tree Search (UCT, Kocsis & Szepesvári 2006) ersetzt sie durch zufällige Rollouts. Hier wird gemessen, was davon auf einem
deterministischen Pfadproblem mit Reward nur am Ziel übrig bleibt - und wie viel des Erfolgs eingeschmuggeltes Heuristik-Wissen ist.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import streamlit as st

import mcts_constants as C
from mcts_evaluation import SWEEP_LABELS, VARIANT_LABELS, Settings, ablation, analyse, anytime_curves, sweep
from mcts_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from mcts_visualization import (
    build_ablation_bars,
    build_anytime,
    build_instance,
    build_replay,
    build_share_bars,
    build_sweep,
    build_tree_map,
)

st.set_page_config(page_title="Monte Carlo Tree Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _curves(settings):
    return anytime_curves(settings, n=6)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _ablation(base):
    return ablation(base)


st.title("🎲 Monte Carlo Tree Search – geht es auch ohne Heuristik?")
st.markdown(
    """
**Achtes und letztes Stück der Heuristische-Baumsuche-Linie.** Alle sieben Stücke davor teilen eine Annahme: es gibt eine handgemachte Heuristik
h(n), hier die Luftlinie zum Ziel. **Monte Carlo Tree Search** (MCTS) ersetzt sie durch **zufällige Rollouts**: den Wert eines Knotens schätzt es, indem
es von dort einen Zufallslauf bis zum Ziel simuliert; **UCB1** entscheidet, ob es einen bekannten guten Ast vertieft oder einen seltener besuchten
erkundet, und der Baum wächst dorthin, wo die Simulationen Gutes versprechen. In Spielen (Go, Schach) ist das der Standard, wo keine brauchbare
Heuristik existiert.

Auf einem **deterministischen Pfadproblem mit Reward nur am Ziel** ist die Frage offen: reichen Zufallsläufe, um das Ziel überhaupt zu finden? Wie nah
kommt MCTS ans Optimum, und was kostet das gegenüber Uniform-Cost, das ebenfalls ohne Heuristik das Optimum findet? Und wie viel des Erfolgs ist
Heuristik-Wissen, das man dem Rollout oder dem Reward heimlich mitgibt? Gemessen, nicht angenommen.
"""
)
st.caption(
    "Setzt auf [greedy-best-first-demo](https://github.com/sebastian-hanisch/greedy-best-first-demo) und [astar-demo](https://github.com/sebastian-hanisch/astar-demo) auf "
    "(derselbe Graph, dieselben Instanzen; Uniform-Cost, A\\* und Greedy Best-First als Vergleich). Nicht gebaut: Weighted A\\*/ARA\\*, Stochastic Beam Search."
)

with st.expander("So funktioniert MCTS (UCT)", expanded=True):
    st.markdown(
        """
1. **Selektion:** von der Wurzel absteigen; unter den Kindern zählt **UCB1 = mittlerer Reward + c · √(ln N<sub>Eltern</sub> / N<sub>Kind</sub>)** - hoher Reward lockt zum Vertiefen, wenig besucht lockt zum Erkunden.
2. **Expansion:** ein noch nicht probiertes Kind wird an den Baum gehängt. Der Baum lebt über **Pfade**; ein Zustand wiederholt sich nie auf einem Weg.
3. **Simulation (Rollout):** von dort ein selbstvermeidender Zufallslauf, bis das Ziel erreicht ist, eine Sackgasse kommt oder der **Horizont** endet. Reward: Ziel erreicht mit Kosten L → **d0 / L** (d0 = Luftlinie Start-Ziel, also höchstens 1), sonst 0.
4. **Rückpropagation:** der Reward geht in alle Vorfahren. Ergebnis ist die **beste je gefundene Route** (jede Simulation, die das Ziel erreicht, liefert einen gültigen Pfad).
        """,
        unsafe_allow_html=True,
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    network = st.radio("Instanz", options=list(C.NETWORKS), format_func=lambda n: C.NETWORK_LABELS[n], key="network_select",
                        help="Die handgebaute Falle täuscht die Heuristik (Greedy Best-First scheitert dort) - Rastergröße/Hindernisdichte/Seed wirken dort nicht.")
    if network == "grid":
        side = st.slider("Rastergröße (Seitenlänge)", *bounds("side_slider"), key="side_slider",
                          help="Je größer das Raster, desto seltener erreicht ein Zufallslauf das Ziel (Ziel-Rollouts bei Größe 10: 5 %, Größe 22: 0.2 %). Die Lücke wächst von Größe 10 (2.0 %) bis Größe 22 (67.8 %) deutlich; bei Größe 6 liegt sie mit 5.6 % noch über der bei 10 (Median über 30 Läufe), sie steigt also nicht streng mit der Größe.")
        obstacle_pct = st.slider("Hindernisdichte [%]", *bounds("obstacle_slider"), key="obstacle_slider", step=C.OBSTACLE_STEP,
                                  help="Mit vielen Hindernissen wird MCTS besser (bei 40 %: Lücke 7 % statt 17 %): die Wege verengen sich, es gibt weniger Irrwege.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        side, obstacle_pct, seed = C.DEFAULT_SIDE, C.DEFAULT_OBSTACLE, C.DEFAULT_SEED
    chain_seed = st.number_input("Seed des MCTS-Laufs", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
                                 help="MCTS würfelt (Expansion, Rollouts): derselbe Seed liefert denselben Lauf.")
    st.button("🎲 Neuer Lauf", width="stretch", on_click=randomize_chain_seed)
    iterations = st.select_slider("Iterationen", options=list(C.ITERATIONS), key="iterations_select", format_func=lambda v: f"{v:,}".replace(",", "."),
                                  help="Zahl der Selektion-Expansion-Simulation-Rückpropagation-Durchläufe. Größenordnung des Aufwands: rund 25 Knotenbesuche je Iteration.")
    c = st.select_slider("UCB-Konstante c", options=list(C.C_OPTIONS), key="c_select", format_func=lambda v: f"{v:g}",
                         help="Gewicht des Erkundungsterms. Auf kleinen Rastern entscheidend (Größe 4, 1000 Iterationen: c = 0.25 findet in 10 % der Läufe das Optimum, c = 4 in 100 %), auf Größe 12 fast ohne Wirkung.")
    horizon = st.select_slider("Rollout-Horizont", options=list(C.HORIZON_OPTIONS), key="horizon_select",
                               help="Maximale Länge eines Zufallslaufs. Zu kurz (10): das Ziel wird kaum erreicht (Größe 12: 60 % der Läufe finden nie eine Route); ab 20 ohne große Wirkung.")
    st.caption("Heuristik-Wissen einschmuggeln (die zentrale Ablation):")
    rollout = st.radio("Rollout-Politik", options=list(C.ROLLOUTS), format_func=lambda v: C.ROLLOUT_LABELS[v], key="rollout_select",
                       help="Geführt: der nächste Schritt wird per Softmax über -h (Luftlinie zum Ziel) gezogen - das ist Heuristik-Wissen im Rollout.")
    reward = st.radio("Reward", options=list(C.REWARDS), format_func=lambda v: C.REWARD_LABELS[v], key="reward_select",
                      help="Mit Fortschritt bekommt auch ein Lauf ohne Ziel bis zu 0.5 Reward je nach Luftlinie am Ende - Heuristik-Wissen im Reward.")

sync_query_params({"network_select": network, "side_slider": int(side), "obstacle_slider": int(obstacle_pct), "seed_input": int(seed), "chain_seed_input": int(chain_seed),
                   "iterations_select": int(iterations), "c_select": float(c), "horizon_select": int(horizon), "rollout_select": rollout, "reward_select": reward})

settings = Settings(network, int(side), int(obstacle_pct), int(seed), int(chain_seed), int(iterations), float(c), int(horizon), rollout, reward)
with st.spinner("Rechne..."):
    a = _analysis(settings)
mcts = a.mcts
hand = network != "grid"
knowledge = rollout == "guided" or reward == "progress"

# --- MCTS in Aktion ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 MCTS in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Baum wächst", 3: "3 · Besuche und Route"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="mcts_step", format_func=lambda s: STEP_LABELS[s])
if knowledge:
    st.info("Diese Einstellung nutzt Heuristik-Wissen (geführter Rollout und/oder Fortschritts-Reward): das ist nicht mehr MCTS ohne Heuristik.")

if step == 1:
    if hand:
        st.markdown(f"**{a.inst.graph.n} Knoten**: Start, Ziel, ein Köder nahe am Ziel (Sackgasse, in die die Heuristik lockt) und ein Umweg (der kürzere Weg)")
    else:
        st.markdown(f"**{a.inst.graph.n} Zellen** ({len(a.inst.blocked_xy)} Hindernisse), Start (grün) und Ziel (rot)")
    st.plotly_chart(build_instance(a.inst), width="stretch", key="s1_map")
elif step == 2:
    n_rec = len(mcts.replay)
    if n_rec >= 1:
        if "mcts_iter" in st.session_state:
            st.session_state["mcts_iter"] = min(max(1, int(st.session_state["mcts_iter"])), n_rec)
        k = st.slider("Iteration", 1, max(2, n_rec), key="mcts_iter", help=f"Wiedergabe der ersten {C.REPLAY_ITERATIONS} Iterationen: Baumkanten bis zur Iteration, der neue Knoten und sein Rollout.") if n_rec > 1 else 1
        k = min(k, n_rec)
        new_node, roll, reward_k = mcts.replay[k - 1]
        st.markdown(
            f"**Iteration {k} von {mcts.iterations}:** Baum mit {sum(1 for t in mcts.tree_iteration if t <= k)} Knoten; Rollout mit {len(roll)} Schritten"
            + (" erreichte das Ziel" if reward_k > 0 and roll and roll[-1] == a.inst.goal else " erreichte das Ziel nicht") + f", Reward {reward_k:.2f}."
        )
        st.plotly_chart(build_replay(a.inst, mcts, k), width="stretch", key=f"s2_map_{k}")
    else:
        st.info("Keine Iterationen gewählt.")
else:
    st.plotly_chart(build_tree_map(a.inst, mcts, a.ucs.path), width="stretch", key="s3_map")
    st.caption("Kreisfläche = Besuche im Baum, Farbe = Rollout-Besuche (logarithmisch). Rot: beste gefundene Route, grün gestrichelt: der robuste Pfad (immer das meistbesuchte Kind; nur eingezeichnet, wenn er im Ziel endet), blau: Optimum.")

st.markdown("---")

# --- Ergebnis ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis gegen Uniform-Cost, A\\* und Greedy")
st.caption(
    "**Lücke:** Kosten der besten gefundenen Route gegenüber dem Optimum (Uniform-Cost). **Knotenbesuche** (MCTS) und **Expansionen** (Uniform-Cost, A\\*) sind "
    "verschiedene Einheiten - nur der Größenordnung nach vergleichbar. **Ziel-Rollouts:** Anteil der Iterationen, die das Ziel erreichen (spärlicher Reward)."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("MCTS: Lücke", "kein Pfad" if not mcts.solved else f"{a.gap:.2f} %", delta=f"{mcts.visits_total:,} Besuche".replace(",", "."), delta_color="off")
m2.metric("UCS / A*", f"{a.ucs.expansions} / {a.astar.expansions}", delta="beide optimal", delta_color="off")
m3.metric("Ziel-Rollouts", f"{a.goal_share:.1f} %", delta="der Iterationen", delta_color="off")
m4.metric("Greedy: Lücke", f"{a.gbfs_gap:.2f} %", delta="mit Heuristik", delta_color="off")
if not mcts.solved:
    st.warning("MCTS hat in dieser Konfiguration keine Route zum Ziel gefunden - kein Pfad wird behauptet.")
elif mcts.robust_path:
    st.caption(f"Robuster Pfad (meistbesuchte Kinder) endet im Ziel: Lücke {a.robust_gap:.2f} %.")
else:
    st.caption("Der robuste Pfad (meistbesuchte Kinder) endet nicht im Ziel - die beste Route stammt aus einem einzelnen Rollout, nicht aus der Baumstatistik.")

st.markdown("---")

# --- Anytime ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⏱️ Wie schnell wird die Route besser?")
st.caption(
    "Beste gefundene Kosten über die Knotenbesuche für sechs Läufe (Seeds ab dem eingestellten). Senkrecht: der Aufwand von Uniform-Cost und A\\* in Expansionen - "
    "eine andere Einheit, aber MCTS braucht Größenordnungen mehr."
)
curves = _curves(settings)
st.plotly_chart(build_anytime(curves, a.ucs.cost, a.ucs.expansions, a.astar.expansions if a.astar is not None else None, max(1, mcts.visits_total)), width="stretch", key="anytime")

st.markdown("---")

# --- Ablation ----------------------------------------------------------------------------------------------------------------------------------

if network == "grid":
    st.subheader("🔬 Wie viel des Erfolgs ist eingeschmuggeltes Heuristik-Wissen?")
    st.caption(
        "Dieselbe Konfiguration in vier Varianten über 5 feste Instanzen (Seeds 100000–100004) x 6 Läufe: rein (gleichverteilter Rollout, Reward nur am Ziel), "
        "mit Fortschritts-Reward, mit geführtem Rollout, mit beidem. Die letzten drei benutzen die Luftlinie zum Ziel - also h."
    )
    key_abl = replace(settings, seed=0, chain_seed=0, rollout="uniform", reward="goal")
    if st.button("Ablation über 30 Läufe je Variante messen (kann eine Minute dauern)", key="abl_start"):
        st.session_state["abl_done"] = st.session_state.get("abl_done", set()) | {key_abl}
    if key_abl in st.session_state.get("abl_done", set()):
        with st.spinner("Rechne 4 Varianten x 30 Läufe..."):
            abl = _ablation(key_abl)
        st.plotly_chart(build_ablation_bars(abl, VARIANT_LABELS), width="stretch", key="abl_bars")
        short = {"pure": "rein", "reward": "+ Reward (h)", "rollout": "+ Rollout (h)", "both": "beides (h)"}
        cols = st.columns(len(abl))
        for col, (variant, res) in zip(cols, abl.items()):
            col.metric(short[variant], "-" if res["gap"] != res["gap"] else f"{res['gap']:.1f} %", delta=f"≤ 5 %: {res['near_share']:.0f} %", delta_color="off")
        st.caption("Kennzahlen: Lücke im Median über die Läufe mit Route; darunter der Anteil aller Läufe mit Lücke ≤ 5 %.")
        st.caption(f"Instanzen: Rastergröße {settings.side}, Hindernisdichte {settings.obstacle_pct} %, {settings.iterations} Iterationen, c = {settings.c:g}, Horizont {settings.horizon}.")

    st.markdown("---")

    st.subheader("📐 Wie hängen Erfolg und Lücke von den Reglern ab?")
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
    metric = st.radio("Kennzahl", options=["gap", "shares", "goal", "visits"],
                       format_func=lambda k: {"gap": "Lücke (%)", "shares": "Lösung / ≤ 5 % / optimal (%)", "goal": "Ziel-Rollouts (%)", "visits": "Knotenbesuche"}[k],
                       key="sweep_metric", horizontal=True)
    base_sweep = replace(settings, seed=0, chain_seed=0)
    if st.button("Sweep über 30 Läufe je Wert berechnen (kann eine Minute dauern)", key="sweep_start"):
        st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
    if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 30 Läufe je Wert..."):
            rows_sweep = _sweep(sweep_param, base_sweep)
        label = SWEEP_LABELS[sweep_param]
        if metric == "gap":
            st.plotly_chart(build_sweep(rows_sweep, label, [("gap", "MCTS: Lücke", "#e45756"), ("gbfs_gap", "Greedy Best-First: Lücke", "#888888")], "Lücke zum Optimum (%)"),
                            width="stretch", key="sweep_gap")
        elif metric == "shares":
            st.plotly_chart(build_share_bars(rows_sweep, label), width="stretch", key="sweep_shares")
        elif metric == "goal":
            st.plotly_chart(build_sweep(rows_sweep, label, [("goal_share", "Ziel-Rollouts", "#4c78a8")], "Anteil der Iterationen (%)"), width="stretch", key="sweep_goal")
        else:
            st.plotly_chart(build_sweep(rows_sweep, label, [("visits_total", "Knotenbesuche gesamt", "#7b3fbf"), ("first_visits", "bis zur ersten Route", "#f58518")], "Knotenbesuche"),
                            width="stretch", key="sweep_visits")
        st.caption(
            "Median über 5 feste Instanzen (Seeds 100000–100004) x 6 Läufe, Band = 10. bis 90. Perzentil. Lücken nur über Läufe mit Lösung; Läufe ohne Route stehen im Balken-Modus. "
            "Die übrigen Regler stehen wie in der Seitenleiste."
        )

    st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ohne Heuristik geht es** | Nur grob. Auf dem Standardraster (Größe 12, 15 % Hindernisse, 4000 Iterationen) findet reines MCTS zwar in **100 %** der Läufe eine Route, aber nie die optimale und im Median 17 % über dem Optimum; Uniform-Cost findet das Optimum mit 121 Expansionen. Die Lücke fällt mit den Iterationen (250 / 1000 / 4000 / 16000: 40 / 28 / 17 / 8.5 %), aber langsam. | Uniform-Cost, A\\* (die Standardantwort) |
| **Zufallsläufe erreichen das Ziel** | Nur auf kleinen Rastern: Ziel-Rollouts 95 % bei Größe 6, 5 % bei Größe 10, 0.2 % bei Größe 22; die Lücke wächst ab Größe 10 mit der Größe auf 42 % (Größe 18) und 68 % (Größe 22). Ein zu kurzer Horizont (10) verhindert selbst bei Größe 12 in 60 % der Läufe jede Route. | Ein Reward mit Fortschrittssignal - das ist dann Heuristik |
| **Der Erfolg gehört MCTS** | Zum großen Teil nicht: mit **geführtem Rollout** (Softmax über die Luftlinie) sinkt die Lücke von 17 % auf 2.4 %, der Anteil Läufe ≤ 5 % von 0 % auf 83 %. Ein Fortschritts-Reward allein bringt wenig (12 %). Was übrig bleibt, ist Greedy Best-First mit Zufall. | - |
| **UCB1 mit c = 1 passt überall** | Nein: auf dem 4x4-Raster mit 1000 Iterationen findet c = 0.25 in 10 % der Läufe das Optimum, c = 1 in 67 %, c = 4 in 100 %. Auf Größe 12 ändert c fast nichts (Lücke 13 bis 18 % über c = 0.25 bis 16). Vermutete Ursache (nicht isoliert): der Reward d0 / L liegt eng beieinander, daher braucht UCB1 ein großes c, um Alternativen weiter zu erkunden. | Reward-Normalisierung, Einzelspieler-Varianten (SP-MCTS; hier nicht gebaut) |
| **Deterministisches Problem** | Auf der handgebauten Falle findet MCTS ohne Heuristik den Umweg (Greedy fällt auf den Köder herein, 7.5 % Lücke), braucht aber für einen Graphen mit 8 Knoten schon 4 Knotenbesuche bis zur ersten optimalen Route und rund 1200 bei 250 Iterationen. Ein deterministisches, vollständig bekanntes Netz ist nicht das Terrain von MCTS: Uniform-Cost und A\\* sind dort exakt und billiger. | Spiele, Simulatoren, stochastische Übergänge (hier nicht gebaut) |
| **Baum ohne Transpositionen, Vanilla-UCT** | Derselbe Zustand kann in vielen Ästen stehen; RAVE, AlphaZero-Netze und Transpositionstabellen sind nicht gebaut. | Weiterentwicklungen von MCTS |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Baum.** Ein Knoten $v$ trägt einen Zustand $s(v)$ (auf dem Weg von der Wurzel keine Wiederholung), Besuche $N(v)$ und die Reward-Summe $W(v)$.

**Selektion (UCB1, Auer et al. 2002).** Vom Knoten $u$ mit vollständig expandierten Kindern wähle
$$v^\* = \arg\max_{v \in \text{children}(u)} \; \frac{W(v)}{N(v)} + c \sqrt{\frac{\ln N(u)}{N(v)}}.$$
Hat $u$ noch nicht probierte Kinder, wird zuerst eines davon (zufällig) expandiert.

**Simulation und Reward.** Ein selbstvermeidender Zufallslauf ab dem neuen Knoten (höchstens $H$ Schritte). Erreicht er das Ziel mit Gesamtkosten $L$, ist
$r = \min(1, d_0 / L)$ mit der Luftlinie $d_0$ zwischen Start und Ziel; sonst $r = 0$. Variante *Fortschritt*: $r = \tfrac12 \max(0, 1 - h(s_{\text{Ende}}) / d_0)$.
Variante *geführt*: Schrittwahl mit Wahrscheinlichkeit $\propto \exp\!\big(-(h(v) - h_{\min}) / (\tau \bar w)\big)$, $\tau = 0.5$, $\bar w$ = mittlere Kantenlänge.

**Rückpropagation.** Für jeden Vorfahren $N \mathrel{+}= 1$, $W \mathrel{+}= r$. Die Ausgabe ist die **beste beobachtete Route**; der *robuste Pfad* folgt von der Wurzel dem jeweils meistbesuchten Kind.

**Aufwand.** Knotenbesuche = berührte Baumstufen + Rollout-Schritte (nicht dasselbe wie eine Expansion von Uniform-Cost/A\*).

**Literatur.** Kocsis, L., & Szepesvári, C. (2006). *Bandit Based Monte-Carlo Planning.* ECML 2006, LNCS 4212, 282-293. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem.* Machine Learning 47, 235-256. Browne, C. B. et al. (2012). *A Survey of Monte Carlo Tree Search Methods.* IEEE Transactions on Computational Intelligence and AI in Games 4(1), 1-43.

Implementiert in `mcts_algorithm.py` (Suchkerne aus den Geschwistern, `mcts_search` neu), `mcts_graph.py`/`mcts_scenario.py` (Graph, Raster und Falle), `mcts_evaluation.py` (Kennzahlen, Sweeps, Ablation).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Heuristische Baumsuche: Greedy bis MCTS](https://sebastianhanisch.net/konzepte-heuristische-baumsuche.html)."
)
