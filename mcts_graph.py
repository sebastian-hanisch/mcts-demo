"""Schlanke, eigene Graph-Repräsentation mit Koordinaten - Adjazenzlisten reichen für die Rastergrößen dieses
Stücks (kein CSR-Overengineering wie bei `dijkstra-demo`, das für viel größere echte Netze gebaut wurde)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    xy: np.ndarray              # (n, 2) Koordinaten in km
    neighbors: tuple            # neighbors[u] = tuple von Nachbarknoten
    weights: tuple               # weights[u] = tuple von Kantengewichten, passend zu neighbors[u]

    @property
    def n(self):
        return len(self.xy)


def adjacency(n, edges):
    """`edges`: Liste von (u, v, w), UNGERICHTET - jede Kante wird in beide Richtungen aufgenommen. Doppelte
    Kanten zwischen demselben Knotenpaar werden auf die günstigste reduziert. Gibt (neighbors, weights) zurück -
    wird auch direkt von `mcts_scenario.py` benutzt, wenn die `xy` dort schon vorliegen."""
    best = {}
    for u, v, w in edges:
        for a, b in ((u, v), (v, u)):
            key = (a, b)
            if key not in best or w < best[key]:
                best[key] = w
    adj = [[] for _ in range(n)]
    for (u, v), w in best.items():
        adj[u].append((v, w))
    neighbors = tuple(tuple(v for v, _w in sorted(adj[u])) for u in range(n))
    weights = tuple(tuple(w for _v, w in sorted(adj[u])) for u in range(n))
    return neighbors, weights


def from_edges(n, xy, edges):
    neighbors, weights = adjacency(n, edges)
    return Graph(np.asarray(xy, dtype=float), neighbors, weights)


def edge_cost(graph, u, v):
    """Kosten der Kante (u, v), oder None falls keine Kante existiert."""
    for w, cost in zip(graph.neighbors[u], graph.weights[u]):
        if w == v:
            return cost
    return None


def path_cost(graph, path):
    """Summe der Kantenkosten entlang `path` (Liste von Knotenindizes) - unabhängige Neuberechnung, nicht die
    vom Suchalgorithmus mitgeführte Summe."""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        cost = edge_cost(graph, u, v)
        if cost is None:
            raise ValueError(f"keine Kante zwischen {u} und {v}")
        total += cost
    return total
