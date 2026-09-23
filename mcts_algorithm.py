"""Suchkerne - `_search`, `greedy_best_first`, `uniform_cost_search`, `a_star` wortgleich aus `astar-demo`/`beam-search-demo` (dort korrektheitsgeprüft;
Vergleichsgrößen dieses Stücks: UCS = das Optimum OHNE Heuristik, A* und GBFS = mit) und NEU `mcts_search` (Monte Carlo Tree Search, UCT).

MCTS ersetzt die handgemachte Heuristik h(n) durch zufällige Rollouts: Werte werden aus Simulationen geschätzt, UCB1 balanciert Erkundung und
Ausbeutung, der Baum wächst asymmetrisch zu Ästen mit gutem Reward. Der Reward gibt es nur am Ziel (spärlich); wie weit das auf einem
deterministischen Pfadproblem trägt und wie viel davon eingeschmuggeltes Heuristik-Wissen ist (Reward "progress", Rollout "guided"), wird gemessen."""

import heapq
import math
import random
from dataclasses import dataclass, field

import numpy as np


def heuristic(xy, goal):
    """Euklidischer Abstand jedes Knotens zum Ziel - vektorisiert. Bei echten Kantengewichten (siehe
    `mcts_scenario.py`) automatisch zulässig (Dreiecksungleichung)."""
    return np.hypot(*(xy - xy[goal]).T)


@dataclass
class SearchResult:
    path: list                  # Knotenfolge Start..Ziel, oder [] falls kein Pfad existiert
    cost: float                 # Summe der Kantengewichte entlang des Pfades
    expansions: int             # Zahl der expandierten Knoten (Effizienz-Kennzahl dieses Stücks)
    order: list = field(default_factory=list)     # Reihenfolge der expandierten Knoten (für die Schritt-Visualisierung)
    stored: int = 0             # Speicher-Kennzahl: gespeicherte Knoten am Ende (A*/UCS/GBFS: entdeckte Knoten; IDA*: max. Pfadtiefe)


def _search(graph, start, goal, priority_fn, relax=True):
    """`priority_fn(node, g_cost) -> float` bestimmt die Warteschlangen-Priorität. `g_cost` ist der bislang
    aufgelaufene Pfadwert zu `node` (für Uniform-Cost-Search gebraucht, von Greedy Best-First ignoriert).

    `relax`: ob ein noch nicht expandierter, aber schon entdeckter Knoten einen GÜNSTIGEREN Elternknoten
    bekommt, sobald ein billigerer Weg zu ihm gefunden wird (klassische Dijkstra-Relaxation - für
    Uniform-Cost-Search nötig, damit es tatsächlich optimal bleibt). Bei `relax=False` behält ein Knoten für
    immer den ERSTEN gefundenen Elternknoten (echtes "kein Backtracking" - der Kern der GBFS-Schwäche: eine
    Relaxation hier würde den gemessenen Qualitätsverlust künstlich kleinrechnen, da GBFS dann doch beiläufig
    von g(n) profitieren würde, obwohl es g(n) laut Definition komplett ignoriert)."""
    counter = 0
    frontier = [(priority_fn(start, 0.0), counter, start, 0.0)]
    came_from = {start: None}
    g_cost = {start: 0.0}
    visited = set()
    order = []

    while frontier:
        _priority, _c, node, g = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return SearchResult(path, g, len(order), order, len(g_cost))
        for v, w in zip(graph.neighbors[node], graph.weights[node]):
            if v in visited:
                continue
            g_v = g + w
            is_new = v not in g_cost
            if is_new or (relax and g_v < g_cost[v]):
                g_cost[v] = g_v
                came_from[v] = node
                counter += 1
                heapq.heappush(frontier, (priority_fn(v, g_v), counter, v, g_v))

    return SearchResult([], float("inf"), len(order), order, len(g_cost))


def greedy_best_first(graph, start, goal):
    h = heuristic(graph.xy, goal)
    return _search(graph, start, goal, lambda node, g: h[node], relax=False)


def uniform_cost_search(graph, start, goal):
    return _search(graph, start, goal, lambda node, g: g, relax=True)


def a_star(graph, start, goal):
    h = heuristic(graph.xy, goal)
    return _search(graph, start, goal, lambda node, g: g + h[node], relax=True)


REWARD_MODES = ("goal", "progress")
ROLLOUT_MODES = ("uniform", "guided")
GUIDE_TAU = 0.5                 # Temperatur des geführten Rollouts in mittleren Kantenlängen (Softmax über -h)
REPLAY_ITERATIONS = 500         # so viele Iterationen werden für die Wiedergabe aufgezeichnet


def ucb1(w_sum, n, n_parent, c):
    """UCB1 (Auer et al. 2002): mittlerer Reward + c * sqrt(ln N_Eltern / N_Kind)."""
    return w_sum / n + c * math.sqrt(math.log(n_parent) / n)


@dataclass
class MCTSResult:
    path: list                  # beste je beobachtete Zielroute (jede Simulation, die das Ziel erreicht, ist ein gültiger Pfad), [] wenn keine
    cost: float
    solved: bool
    iterations: int
    visits_total: int           # Aufwandseinheit "Knotenbesuche": jede berührte Baum-Knoten-Stufe und jeder Rollout-Schritt zählt 1
    tree_size: int
    rollout_steps: int
    goal_rollouts: int          # Simulationen, die das Ziel erreicht haben (Baum-Ziel-Knoten mitgezählt)
    first_solution_iter: int = 0
    first_solution_visits: int = 0
    anytime: list = field(default_factory=list)          # [(Iteration, Knotenbesuche, beste Kosten)] bei jeder Verbesserung
    robust_path: list = field(default_factory=list)      # Weg des jeweils am häufigsten besuchten Kindes ab der Wurzel; endet er im Ziel, ist er eine Lösung
    robust_cost: float = float("inf")
    state_visits: object = None                          # np.ndarray: Baum-Besuche je Zustand
    rollout_visits: object = None                        # np.ndarray: Rollout-Besuche je Zustand
    tree_state: list = field(default_factory=list)       # Baumknoten i -> Zustand
    tree_parent: list = field(default_factory=list)      # Baumknoten i -> Elternknoten (-1 bei der Wurzel)
    tree_visits: list = field(default_factory=list)      # Baumknoten i -> N
    tree_reward: list = field(default_factory=list)      # Baumknoten i -> Summe der zurückpropagierten Rewards W
    tree_iteration: list = field(default_factory=list)   # Baumknoten i -> Iteration, in der er entstand (0 = Wurzel)
    replay: list = field(default_factory=list)           # je aufgezeichneter Iteration: (neuer Baumknoten oder -1, Rollout-Zustände, Reward)

    @property
    def failed(self):
        return not self.solved


def mcts_search(graph, start, goal, iterations, seed=0, c=1.0, horizon=60, rollout="uniform", reward="goal", h=None, tau=GUIDE_TAU):
    """Monte Carlo Tree Search (UCT, Kocsis & Szepesvári 2006) auf dem Pfadproblem.

    Der Baum lebt über PFADEN: ein Baumknoten ist ein Zustand mit dem Weg, auf dem er erreicht wurde (derselbe Zustand kann in verschiedenen
    Ästen vorkommen, Transpositionen werden nicht zusammengeführt). Ein Zustand wiederholt sich nie auf demselben Weg. Eine Iteration:
    Selektion (UCB1: Q + c * sqrt(ln N_Eltern / N_Kind), unbesuchte Kinder werden zuerst expandiert), Expansion (ein zufällig gewähltes noch nicht probiertes Kind),
    Simulation (selbstvermeidender Zufallslauf bis Ziel, Sackgasse oder `horizon` Schritte) und Rückpropagation desselben Rewards in alle Vorfahren.
    Reward: erreicht die Simulation das Ziel mit Gesamtkosten L, ist er d0 / L (d0 = Luftlinie Start-Ziel, also in (0, 1]); sonst 0 (`reward="goal"`) oder
    der Fortschritt 0.5 * max(0, 1 - h(Ende) / d0) (`reward="progress"`, benutzt die Heuristik). `rollout="guided"` zieht den nächsten Schritt per Softmax
    über -h (Temperatur `tau` Kantenlängen, benutzt die Heuristik), `"uniform"` gleichverteilt. Nur `random()` des Mersenne Twister wird verwendet, damit ein
    Lauf über Python-Versionen hinweg reproduzierbar bleibt."""
    if iterations < 0 or horizon < 1 or c < 0:
        raise ValueError("iterations >= 0, horizon >= 1 und c >= 0 nötig")
    if rollout not in ROLLOUT_MODES or reward not in REWARD_MODES:
        raise ValueError("unbekannter Rollout oder Reward")
    if h is None:
        h = heuristic(graph.xy, goal)
    h = [float(x) for x in h]
    neighbors, weights = graph.neighbors, graph.weights
    n = graph.n
    ws = [w for row in weights for w in row]
    unit = sum(ws) / len(ws) if ws else 1.0
    d0 = h[start]
    rng = random.Random(int(seed) * 1000003 + 12345)
    rand = rng.random
    inf = float("inf")

    state, parent, N, W, g_cost, children, untried, born = [start], [-1], [0], [0.0], [0.0], [[]], [], [0]
    untried.append(list(neighbors[start]))
    state_visits = np.zeros(n)
    rollout_visits = np.zeros(n)

    def path_of(node):
        out = []
        while node != -1:
            out.append(state[node])
            node = parent[node]
        out.reverse()
        return out

    best_cost, best_path = inf, []
    anytime, replay = [], []
    visits = rollout_steps = goal_rollouts = 0
    first_iter = first_visits = 0

    def register(cost, path, it):
        nonlocal best_cost, best_path, first_iter, first_visits
        if cost < best_cost - 1e-12:
            best_cost, best_path = cost, path
            if not anytime:
                first_iter, first_visits = it, visits
            anytime.append((it, visits, cost))

    if start == goal:
        register(0.0, [start], 0)

    for it in range(1, iterations + 1):
        if start == goal:
            break
        node = 0
        on_path = {start}
        new_node = -1
        roll_states = []
        r = 0.0
        while True:
            visits += 1
            state_visits[state[node]] += 1
            if state[node] == goal:                                   # terminaler Zielknoten: Reward aus den Pfadkosten
                r = min(1.0, d0 / g_cost[node]) if g_cost[node] > 0 else 1.0
                goal_rollouts += 1
                register(g_cost[node], path_of(node), it)
                break
            if untried[node]:                                         # Expansion: zufälliges noch nicht probiertes Kind
                pool = untried[node]
                s = pool.pop(int(rand() * len(pool)))
                w = weights[state[node]][neighbors[state[node]].index(s)]
                child = len(state)
                state.append(s); parent.append(node); N.append(0); W.append(0.0); g_cost.append(g_cost[node] + w)
                children.append([]); born.append(it)
                children[node].append(child)
                on_path.add(s)
                untried.append([v for v in neighbors[s] if v not in on_path])
                node, new_node = child, child
                visits += 1
                state_visits[s] += 1
                # Simulation ab dem neuen Knoten
                cur, cost, vis = s, g_cost[child], set(on_path)
                reached = cur == goal
                steps = 0
                while not reached and steps < horizon:
                    cand = [v for v in neighbors[cur] if v not in vis]
                    if not cand:
                        break
                    if rollout == "guided":
                        hm = min(h[v] for v in cand)
                        wts = [math.exp(-(h[v] - hm) / (tau * unit)) for v in cand]
                        u = rand() * sum(wts)
                        k = 0
                        acc = wts[0]
                        while acc < u and k < len(cand) - 1:
                            k += 1
                            acc += wts[k]
                    else:
                        k = int(rand() * len(cand))
                    nxt = cand[k]
                    cost += weights[cur][neighbors[cur].index(nxt)]
                    cur = nxt
                    vis.add(cur)
                    roll_states.append(cur)
                    rollout_visits[cur] += 1
                    steps += 1
                    reached = cur == goal
                rollout_steps += steps
                visits += steps
                if reached:
                    goal_rollouts += 1
                    r = min(1.0, d0 / cost) if cost > 0 else 1.0
                    register(cost, path_of(child) + roll_states, it)
                elif reward == "progress" and d0 > 0:
                    r = 0.5 * max(0.0, 1.0 - h[cur] / d0)
                break
            if not children[node]:                                    # Sackgasse: nichts mehr zu erkunden
                r = 0.0
                break
            best_child, best_score = -1, -inf
            for ch in children[node]:
                score = ucb1(W[ch], N[ch], N[node], c)
                if score > best_score:
                    best_child, best_score = ch, score
            node = best_child
            on_path.add(state[node])
        while node != -1:                                             # Rückpropagation
            N[node] += 1
            W[node] += r
            node = parent[node]
        if it <= REPLAY_ITERATIONS:
            replay.append((new_node, roll_states, r))

    robust, node = [start], 0
    while children[node]:
        node = max(children[node], key=lambda ch: (N[ch], -ch))
        robust.append(state[node])
    robust_cost = sum(weights[u][neighbors[u].index(v)] for u, v in zip(robust[:-1], robust[1:])) if robust[-1] == goal else inf
    return MCTSResult(path=best_path, cost=best_cost, solved=best_cost < inf, iterations=iterations, visits_total=visits, tree_size=len(state),
                      rollout_steps=rollout_steps, goal_rollouts=goal_rollouts, first_solution_iter=first_iter, first_solution_visits=first_visits, anytime=anytime,
                      robust_path=robust if robust[-1] == goal else [], robust_cost=robust_cost, state_visits=state_visits, rollout_visits=rollout_visits,
                      tree_state=state, tree_parent=parent, tree_visits=N, tree_reward=W, tree_iteration=born, replay=replay)
