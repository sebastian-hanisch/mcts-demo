# Monte Carlo Tree Search – geht es auch ohne Heuristik? – Streamlit-Demo

Achtes und letztes Stück der **Heuristische-Baumsuche-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning". Alle sieben Stücke davor ([greedy-best-first-demo](../greedy-best-first-demo), [astar-demo](../astar-demo), [ida-star-demo](../ida-star-demo), [beam-search-demo](../beam-search-demo), [monobeam-demo](../monobeam-demo), [beam-stack-demo](../beam-stack-demo), [diverse-beam-demo](../diverse-beam-demo)) teilen eine Annahme: es gibt eine handgemachte Heuristik h(n), hier die Luftlinie zum Ziel. **Monte Carlo Tree Search** (MCTS, hier UCT: Kocsis & Szepesvári 2006) ersetzt sie durch **zufällige Rollouts**: den Wert eines Knotens schätzt es, indem es von dort einen Zufallslauf bis zum Ziel simuliert; **UCB1** balanciert Erkundung und Ausbeutung, der Baum wächst asymmetrisch zu den Ästen, in denen die Simulationen Gutes versprechen. In Spielen (Go, Schach) ist das der Standard, wo keine brauchbare Heuristik existiert. Auf einem **deterministischen Pfadproblem mit Reward nur am Ziel** ist die Frage offen und wird hier gemessen.

**Einordnung in die Linie:** MCTS ist ein **unabhängiger Ast der Wurzel** (keine Fortsetzung von Beam Search oder A\*). Derselbe Graph und dieselben Instanzen wie in den Geschwistern (die Suchkerne `_search`, `greedy_best_first`, `uniform_cost_search`, `a_star` sind wortgleich kopiert und reproduzieren deren Zahlen, als Test hinterlegt); Uniform-Cost (das Optimum **ohne** Heuristik - die ehrliche Basis), A\* und Greedy Best-First dienen als Vergleichsgrößen. Neu ist `mcts_search` und **Zufall im Kern** (Seed des Laufs als Regler).

```
Greedy Best-First Search (Wurzel)                                                          [gebaut]
 ├─ Beam Search → {Diverse Beam Search, Monobeam}                                          [gebaut]
 ├─ A* → Iterative Deepening A* (IDA*)                                                     [gebaut]
 └─ Monte Carlo Tree Search (MCTS)                                       [DIESES STÜCK - Linie vollständig]
Beam Search + A* → Beam Stack Search (Konvergenzpunkt)                                     [gebaut]
```

Ergebnis in Kürze: **Ohne Heuristik findet MCTS Routen, aber schlechte und teuer.** Auf dem Standardraster (Größe 12, 15 % Hindernisse, 4000 Iterationen) findet reines MCTS in 100 % der Läufe eine Route - in **keinem** Lauf die optimale und im Median **17.1 %** über dem Optimum; Uniform-Cost findet das Optimum mit 121 Expansionen, MCTS braucht im Median rund 81 000 Knotenbesuche (andere Einheit, aber Größenordnungen). Mit wachsendem Raster kollabiert der spärliche Reward (Ziel-Rollouts 95 % bei Größe 6, **0.2 %** bei Größe 22; Lücke bis 68 %). Und der Erfolg gehört zum großen Teil nicht MCTS: mit einem **durch h geführten Rollout** sinkt die Lücke auf **2.4 %** (83 % der Läufe unter 5 %) - das ist Greedy Best-First mit Zufall und Baum, nicht mehr "ohne Heuristik". Die Vorab-Erwartung "reines MCTS leidet an spärlichem Reward und wird zur blinden Suche" ist bestätigt, aber weniger dramatisch als vermutet: bis Größe 14 findet es in 100 % der Läufe eine Route, es bleibt nur weit vom Optimum.

| Frage | Ergebnis (Rastergröße 12, Hindernisdichte 15 %, 4000 Iterationen, c = 1, Horizont 100, sofern nicht anders angegeben; **Median** über 5 feste Instanzen, Seeds 100000–100004, x 6 MCTS-Seeds = 30 Läufe; Lücke nur über Läufe mit Route; Erfolgs-, Nah- und Optimal-Anteil über alle Läufe) |
|---|---|
| **Findet reines MCTS Routen?** | ✅ 100 % der Läufe (Größe 22: 93 %) |
| **... wie gute?** | ❌ optimal in **0 %**, Lücke im Median **17.1 %**, Lücke ≤ 5 % in 0 % der Läufe; Greedy Best-First hat auf denselben Instanzen 14.7 % |
| **... wie schnell wird es besser?** | Lücke bei 250/1000/4000/16000 Iterationen: **40.0/28.2/17.1/8.5 %**, Anteil ≤ 5 %: 0/0/0/20 % - sie fällt, aber langsam |
| **Aufwand gegen Uniform-Cost** | Uniform-Cost 121 Expansionen (A\*: 93) für das Optimum; MCTS ~81 000 Knotenbesuche für Lücke 17 % (**Einheiten verschieden**: Baumstufe + Rollout-Schritt gegen Knotenexpansion - nur der Größenordnung nach vergleichbar) |
| **Spärlicher Reward?** | Ziel-Rollouts (Anteil der Iterationen, die das Ziel erreichen) bei Größe 6/10/14/18/22: **95.3/5.1/2.6/0.3/0.2 %**; Lücke 5.6/2.0/21.5/41.7/**67.8 %** |
| **Horizont** | Horizont 10 (Größe 12): nur **40 %** der Läufe finden eine Route; ab 20 ohne große Wirkung (Lücke 17.6/19.2/17.1/17.1 % für 20/50/100/200) |
| **Wie viel ist eingeschmuggeltes h?** | Ablation, Lücke im Median / Anteil ≤ 5 %: **rein 17.1 % / 0 %**; + Fortschritts-Reward 12.4 % / 10 %; **+ geführter Rollout 2.4 % / 83 %**; beides 3.2 % / 80 %. Bei 16000 Iterationen 8.5/9.1/1.3/0.5 % (Anteil ≤ 5 %: 20/23/93/100 %). Größe 6: rein 5.6 % (23 % optimal), geführt 0.0 % (100 % optimal); Größe 22: rein 67.8 %, geführt 7.0 % |
| **UCB-Konstante c** | ⚠️ auf dem 4x4-Raster ohne Hindernisse (1000 Iterationen, 10 Instanzen x 3 Seeds) findet c = 0.25/1/4/16 in **10/67/100/100 %** der Läufe das Optimum (Größe 5: c = 4 nur 67 %, Größe 6: 7 %); auf Größe 12 ist c fast wirkungslos (Lücke 13.3/16.5/17.1/17.6/14.6/16.8/16.7 % für c = 0.25 bis 16). Vermutete Ursache, **nicht isoliert**: der Reward d0 / L liegt eng beieinander, daher braucht UCB1 ein großes c |
| **Hindernisse (0/10/20/30/40 %)** | MCTS wird mit vielen Hindernissen **besser**: Lücke 20.2/17.2/18.1/15.8/**7.4 %** (bei 40 %: 10 % der Läufe optimal, 40 % unter 5 %) - die Wege verengen sich |
| **Handgebaute Falle** | Greedy Best-First fällt auf den Köder herein (7.5 % Lücke); MCTS ohne Heuristik findet den Umweg in allen 10 getesteten Seeds, mit Seed 1 nach **4 Knotenbesuchen** (Uniform-Cost 8, A\* 7 Expansionen) |

## Was die Demo zeigt

1. **MCTS in Aktion** (Schritt-Slider): **Instanz** → **Baum wächst** (Iterations-Slider über die ersten 500 Iterationen: Baumkanten bis zur Iteration, der neue Knoten und sein Rollout mit Reward) → **Besuche und Route** (Kreisfläche = Besuche im Baum, Farbe = Rollout-Besuche, beste gefundene Route, robuster Pfad = jeweils meistbesuchtes Kind, Optimum).
2. **Ergebnis:** Lücke, Expansionen von Uniform-Cost und A\*, Ziel-Rollouts, Lücke von Greedy; ein Lauf ohne Route wird ausdrücklich als "kein Pfad" ausgewiesen.
3. **⏱️ Anytime:** beste Kosten über die Knotenbesuche für sechs Läufe, senkrecht der Aufwand von Uniform-Cost und A\*.
4. **🔬 Ablation:** dieselbe Konfiguration rein / mit Fortschritts-Reward / mit geführtem Rollout / mit beidem über 30 Läufe.
5. **📐 Sweeps** über Iterationen, c, Horizont, Rastergröße und Hindernisdichte (Median, 10.–90. Perzentil-Band; Lücke, Lösung / ≤ 5 % / optimal, Ziel-Rollouts, Knotenbesuche).
6. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Instanz (Raster / **Heuristik-Falle**), Rastergröße (4–25), Hindernisdichte (0–40 %), Seed der Instanz (+ 🎲), **Seed des MCTS-Laufs** (+ 🎲), **Iterationen** (100–64 000), **UCB-Konstante c** (0.25–16), **Rollout-Horizont** (10–200), **Rollout-Politik** (gleichverteilt / geführt durch h) und **Reward** (nur Ziel / mit Fortschritt) - die beiden letzten sind die Ablations-Schalter; eine Einstellung mit Heuristik-Wissen wird mit einem Hinweis markiert.

## Messwerte der Presets

| Preset | Instanz | Ergebnis |
|---|---|---|
| Standardfall (Voreinstellung) | Größe 12, Seed 35, 4000 Iterationen | Lücke 22.8 % (im Median über 30 Läufe 17.1 %), ~106 000 Knotenbesuche; Uniform-Cost 126, A\* 82 Expansionen (Optimum 175.9 km) |
| Geführter Rollout (mit h) | dieselbe Instanz | Lücke 4.5 % (im Median 2.4 %, 83 % der Läufe unter 5 %) |
| Fortschritts-Reward (mit h) | dieselbe Instanz | Lücke 19.5 % (im Median 12.4 %) |
| Große Karte (Größe 22) | Seed 35 | Ziel-Rollouts 0.1 %, erste Route nach ~28 000 Knotenbesuchen, Lücke 84.5 % (im Median 67.8 %, 7 % der Läufe ohne Route) |
| Kleines Raster, c zu klein | Größe 4, keine Hindernisse, Seed 1, 1000 Iterationen, c = 0.25 | Lücke 19.7 % (13 % der 30 Läufe optimal); mit c = 4 Lücke 0.0 % (100 %) |
| Heuristik-Falle | handgebaut, 250 Iterationen | Lücke 0.0 % nach 4 Knotenbesuchen bis zur ersten optimalen Route; Greedy 7.5 % |
| Viele Hindernisse (40 %) | Größe 12, Seed 35 | Lücke 14.3 % (im Median 7.4 %); Uniform-Cost 88 Expansionen |
| Horizont zu kurz | Horizont 10 | keine Route (im Median finden nur 40 % der Läufe eine) |
| Wenige Iterationen (250) | Seed 35 | Lücke 36.8 % (im Median 40.0 %), ~6700 Knotenbesuche |

Die einzelne Instanz weicht von den Medianen ab - die Mediane sind die belastbaren Zahlen; die Raster-Presets prüfen sich zusätzlich über die 5 festen Instanzen x 6 Seeds gegen eine gemessene Spannweite des Medians der Lücke, die Aussagen der Presets sind als eigene Tests hinterlegt (`tests/test_presets.py`).

## Modell und Verfahren

- **Instanz und Graph** (`mcts_scenario.py`, `mcts_graph.py`): das gestörte Raster mit Hindernissen aus den Geschwistern (Kantengewicht = echter euklidischer Abstand, Heuristik = Luftlinie, automatisch zulässig) und die handgebaute Sackgassen-Falle aus der Wurzel (mit Knotennamen).
- **Suchkerne** (`mcts_algorithm.py`): `_search`, `greedy_best_first`, `uniform_cost_search`, `a_star` aus den Geschwistern; NEU `mcts_search(graph, start, goal, iterations, seed, c, horizon, rollout, reward)`. Der Baum lebt über **Pfade** (derselbe Zustand kann in verschiedenen Ästen stehen, Transpositionen werden nicht zusammengeführt; ein Zustand wiederholt sich nie auf einem Weg). Eine Iteration: **Selektion** per UCB1 (unbesuchte Kinder werden zuerst expandiert) → **Expansion** (zufällig gewähltes noch nicht probiertes Kind) → **Simulation** (selbstvermeidender Zufallslauf bis Ziel, Sackgasse oder Horizont) → **Rückpropagation**. **Reward:** erreicht die Simulation das Ziel mit Gesamtkosten L, ist er d0 / L (d0 = Luftlinie Start-Ziel, höchstens 1), sonst 0; Variante *Fortschritt* zahlt bis 0.5 je nach Luftlinie am Ende (nutzt h), Variante *geführt* zieht den Rollout-Schritt per Softmax über -h (nutzt h, Temperatur 0.5 Kantenlängen). Ergebnis: die **beste je beobachtete Route** (jede Simulation, die das Ziel erreicht, ist ein gültiger Pfad); der **robuste Pfad** folgt von der Wurzel dem meistbesuchten Kind. Nur `random()` des Mersenne Twister wird benutzt, damit ein Lauf über Python-Versionen reproduzierbar bleibt.
- **Auswertung** (`mcts_evaluation.py`): Lücke ggü. Uniform-Cost nur über gelöste Läufe, daneben stets Erfolgs-, Nah-(≤ 5 %) und Optimal-Anteil über alle Läufe; Knotenbesuche bis zur ersten Route / bis Lücke ≤ 5 %; `run_config`, `sweep`, `ablation`, `anytime_curves`.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Erwartung "reines MCTS leidet an spärlichem Reward und wird zur blinden Suche" - bestätigt, aber differenzierter.** Bis Größe 14 findet es in allen Läufen eine Route, ein zu kurzer Horizont (10) kostet dagegen 60 % der Routen bei Größe 12; die Qualität ist der Engpass (17 % Lücke, nie optimal), nicht das bloße Finden.
- **Der Erfolg ist zum großen Teil Heuristik.** Der geführte Rollout senkt die Lücke von 17.1 % auf 2.4 %; ein Fortschritts-Reward allein bringt wenig (12.4 %). Was mit h übrig bleibt, ist Greedy Best-First mit Zufall und Baum (Greedy allein: 14.7 %).
- **UCB1 mit c = 1 passt nicht überall.** Auf kleinen Rastern ist c entscheidend (4x4: 10 % / 67 % / 100 % optimal für c = 0.25 / 1 / 4), auf Größe 12 wirkungslos. Die Ursache (eng beieinander liegender Reward) ist eine Vermutung, nicht isoliert; Reward-Normalisierung und Einzelspieler-Varianten (SP-MCTS) sind nicht gebaut.
- **Deterministisches Problem ist nicht MCTS' Heimat.** Ein vollständig bekanntes Netz löst Uniform-Cost / A\* exakt und mit Größenordnungen weniger Aufwand (121 Expansionen gegen rund 81 000 Knotenbesuche, verschiedene Einheiten); MCTS gehört in Spiele, Simulatoren und stochastische Übergänge - nicht gebaut.
- **Nicht gebaut:** RAVE, Transpositionstabellen, AlphaZero-artige gelernte Bewertung, Weighted A\*/ARA\* und Stochastic Beam Search (zwei erwogene, nicht gewählte Erweiterungen der Linie).
- **Instanzen synthetisch:** ein Raster mit Jitter, Vierer-Nachbarschaft, keine gerichteten Kanten, dazu eine handgebaute Falle. Ursachen sind nicht isoliert; alle Medianwerte stehen auf 30 Läufen je Konfiguration (5 Instanzen x 6 Seeds).

## Verifikation

- **Gültige Route** (zusammenhängend, Start bis Ziel, ohne Wiederholung, Kosten gegen unabhängige Neuberechnung) für beide Rollout-Politiken und beide Rewards; **Kosten nie unter dem Optimum** (gegen vollständige Enumeration auf kleinen Instanzen); Determinismus je Seed, verschiedene Seeds liefern verschiedene Läufe.
- **UCB1** gegen Handrechnung; kleines c konzentriert die Besuche auf den besseren Ast, großes c verteilt sie nahezu gleich.
- **Baum-Invarianten** nach jedem Lauf: Besuchszahlen (N = 1 + Summe der Kinder, Wurzel = Iterationen), Rewards in [0, 1], kein Zustand doppelt auf einem Weg, Kinder = Nachbarn; **Rückpropagation:** die Reward-Summe der Wurzel ist die Summe aller aufgezeichneten Iterations-Rewards; ohne Zielfund wachsen alle Wurzelkinder gleichmäßig (Breitensuche-artig).
- **Buchführung:** Knotenbesuche == Baum-Besuche + Rollout-Schritte, Anytime-Folge streng fallend und konsistent, mehr Iterationen verschlechtern den besten Wert bei gleichem Seed nie.
- **Sonderfälle:** Start == Ziel, unerreichbares Ziel (nie gelöst), Kettengraph (erste Iteration löst), kleines Raster mit großzügigem Budget konvergiert zum Optimum (c = 4), null Iterationen; die Falle (10 Seeds optimal, Greedy nicht).
- **Alle Zahlen der App-Texte sind als Tests hinterlegt**, über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/`ev.ablation`), NIE über ein Ad-hoc-Skript; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt, jede aufgezeichnete Iteration, beide Instanz-Typen, erfolglose Läufe in jedem Schritt, Extremwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Sweeps und Ablation auf Abruf, Footer).

Literatur: Kocsis, L., & Szepesvári, C. (2006). *Bandit Based Monte-Carlo Planning.* ECML 2006, LNCS 4212, 282-293. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time Analysis of the Multiarmed Bandit Problem.* Machine Learning 47, 235-256. Browne, C. B. et al. (2012). *A Survey of Monte Carlo Tree Search Methods.* IEEE Transactions on Computational Intelligence and AI in Games 4(1), 1-43.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Instanz-Umschalter, Schritte (mit Iterations-Slider), Ergebnis, Anytime, 🔬 Ablation, 📐 Sweeps, 🚧 Grenzen, Mathe |
| `mcts_algorithm.py` | Suchkerne (Kopie aus den Geschwistern) + `mcts_search`, `ucb1` |
| `mcts_graph.py`, `mcts_scenario.py` | Graph, Rasterinstanz und Sackgassen-Falle (Kopie) |
| `mcts_constants.py` | Konstanten, Presets, gemessene Werte |
| `mcts_evaluation.py` | Lücke, Anteile, Knotenbesuche, Sweeps, Ablation, Anytime-Kurven |
| `mcts_presets.py`, `mcts_visualization.py` | Permalink/Presets, Plotly-Figuren (Besuchskarte, Baum-Wiedergabe, Anytime, Sweeps, Ablation) |
| `tests/` | Zentrale Korrektheitskette (UCB1, Baum-Invarianten, Rückpropagation), Szenario/Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
