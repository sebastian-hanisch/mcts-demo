"""Die Instanz dieser Demo: ein gestörtes Raster (Lagerhaus-/Straßennetz) mit einem Hindernis-Anteil, Start unten
links, Ziel oben rechts - Vierer-Nachbarschaft (hoch/runter/links/rechts), Kantengewicht = tatsächlicher
euklidischer Abstand der (leicht verschobenen) Zellmittelpunkte. Start und Ziel bleiben IMMER verbunden (wie
`dijkstra-demo`s Labyrinth: bei Bedarf werden einzelne Wände geöffnet, bis eine Verbindung existiert) - ohne diese
Garantie wäre "kein Pfad" der Normalfall bei hoher Hindernisdichte, nicht die Ausnahme, die dieses Stück zeigen
will.

Kantengewicht = echter euklidischer Abstand ⇒ die geradlinige Heuristik ist AUTOMATISCH zulässig (die
Dreiecksungleichung gilt für jeden Pfad zwischen zwei Punkten in der Ebene) - unabhängig von Hindernissen oder
Rasterform. Das Stück muss die Heuristik deshalb nicht künstlich verzerren, um Greedy Best-First Search zum
Scheitern zu bringen: die Hindernisse allein reichen (siehe `mcts_algorithm.py`, Verifikation)."""

from dataclasses import dataclass

import numpy as np

import mcts_constants as C
from mcts_graph import Graph, adjacency

NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class Instance:
    graph: Graph
    start: int
    goal: int
    side: int
    obstacle_pct: int
    seed: int
    blocked_xy: np.ndarray      # (k, 2) Koordinaten der gesperrten Zellen, nur für die Darstellung
    labels: object = None       # Knotennamen der handgebauten Falle, None beim Raster


def _components(open_, side):
    label = -np.ones((side, side), dtype=int)
    count = 0
    for i in range(side):
        for j in range(side):
            if open_[i, j] and label[i, j] < 0:
                label[i, j] = count
                stack = [(i, j)]
                while stack:
                    a, b = stack.pop()
                    for da, db in NEIGHBOR_OFFSETS:
                        a2, b2 = a + da, b + db
                        if 0 <= a2 < side and 0 <= b2 < side and open_[a2, b2] and label[a2, b2] < 0:
                            label[a2, b2] = count
                            stack.append((a2, b2))
                count += 1
    return label


def _ensure_start_goal_connected(open_, side, rng):
    """Öffnet bei Bedarf einzelne gesperrte Zellen (bevorzugt solche, die zwei verschiedene Gebiete berühren),
    bis Start (0,0) und Ziel (side-1,side-1) im selben Gebiet liegen - wie `dijkstra-demo`s Labyrinth-Generator."""
    while True:
        label = _components(open_, side)
        if label[0, 0] == label[side - 1, side - 1]:
            return
        candidates, near_start = [], []
        for i in range(side):
            for j in range(side):
                if open_[i, j]:
                    continue
                labs = {label[a, b] for da, db in NEIGHBOR_OFFSETS
                        if 0 <= (a := i + da) < side and 0 <= (b := j + db) < side and label[a, b] >= 0}
                if len(labs) >= 2:
                    candidates.append((i, j))
                if label[0, 0] in labs:
                    near_start.append((i, j))
        pool = candidates or near_start
        i, j = pool[int(rng.integers(len(pool)))]
        open_[i, j] = True


def build_grid(side, obstacle_pct, seed):
    rng = np.random.default_rng([int(seed), 707])
    spacing = C.AREA / max(1, side - 1)
    open_ = rng.random((side, side)) >= obstacle_pct / 100.0
    open_[0, 0] = open_[side - 1, side - 1] = True
    _ensure_start_goal_connected(open_, side, rng)

    node = -np.ones((side, side), dtype=int)
    cells = np.argwhere(open_)
    node[open_] = np.arange(len(cells))
    ii, jj = cells[:, 0].astype(float), cells[:, 1].astype(float)
    xy = np.stack([jj, ii], axis=1) * spacing + rng.uniform(-C.JITTER, C.JITTER, (len(cells), 2)) * spacing

    edges = []
    for idx, (i, j) in enumerate(cells):
        for da, db in ((1, 0), (0, 1)):                          # jede Kante nur einmal aufnehmen
            i2, j2 = i + da, j + db
            if i2 < side and j2 < side and open_[i2, j2]:
                v = node[i2, j2]
                w = float(np.hypot(*(xy[idx] - xy[v])))
                edges.append((idx, v, w))

    graph = Graph(xy, *adjacency(len(cells), edges))
    start, goal = int(node[0, 0]), int(node[side - 1, side - 1])
    blocked_ij = np.argwhere(~open_)
    blocked_xy = (np.stack([blocked_ij[:, 1], blocked_ij[:, 0]], axis=1).astype(float) * spacing) if len(blocked_ij) else np.zeros((0, 2))
    return graph, start, goal, blocked_xy


def grid_instance(side=C.DEFAULT_SIDE, obstacle_pct=C.DEFAULT_OBSTACLE, seed=C.DEFAULT_SEED):
    graph, start, goal, blocked_xy = build_grid(int(side), int(obstacle_pct), int(seed))
    return Instance(graph, start, goal, int(side), int(obstacle_pct), int(seed), blocked_xy)


# --- Handgebaute Sackgassen-Instanz -----------------------------------------------------------------------------------------------------------

# Eine kleine, von Hand entworfene Falle: vom Start aus liegt ein Köder-Knoten SEHR nah am Ziel (kleines h(n)),
# aber nur über einen langen Umweg-Korridor erreichbar, der ERST ganz am Ende beim Ziel ankommt; der tatsächlich
# kürzeste Weg führt stattdessen über einen zweiten Knotenpfad, dessen Knoten größeres h(n) haben, obwohl er
# insgesamt kürzer ist. Greedy Best-First Search (h(n) allein, kein Backtracking) verpflichtet sich auf den Köder
# und expandiert den GESAMTEN Korridor, bevor es überhaupt den kürzeren Pfad anschaut - und da der Korridor selbst
# beim Ziel ankommt, bricht die Suche dort ab, OHNE je den kürzeren Pfad zu betrachten.
TRAP_NODES = {
    "Start": (0.0, 0.0), "Köder": (8.0, 0.5), "Sackgasse1": (8.5, 1.5), "Sackgasse2": (8.7, 2.5),
    "Sackgassenende": (8.8, 3.5), "Umweg1": (2.0, 3.0), "Umweg2": (5.0, 5.0), "Ziel": (9.0, 0.2),
}
TRAP_LINKS = [
    ("Start", "Köder"), ("Köder", "Sackgasse1"), ("Sackgasse1", "Sackgasse2"), ("Sackgasse2", "Sackgassenende"),
    ("Sackgassenende", "Ziel"), ("Start", "Umweg1"), ("Umweg1", "Umweg2"), ("Umweg2", "Ziel"),
]


def trap_instance():
    names = list(TRAP_NODES.keys())
    idx = {n: i for i, n in enumerate(names)}
    xy = np.array([TRAP_NODES[n] for n in names])
    edges = [(idx[a], idx[b], float(np.hypot(*(xy[idx[a]] - xy[idx[b]])))) for a, b in TRAP_LINKS]
    graph = Graph(xy, *adjacency(len(names), edges))
    blocked_xy = np.zeros((0, 2))
    return Instance(graph, idx["Start"], idx["Ziel"], 0, 0, 0, blocked_xy, tuple(names))
