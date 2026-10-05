"""Unabhängiges Orakel für MCTS: eine objektorientierte Neuimplementierung (Node-Klasse, Vorfahrenmengen statt
`on_path`-Buchführung, UCB1 nach Auer et al. mit Erst-Maximum in Kinderreihenfolge) mit demselben Mersenne-Twister-
Strom muss Baum (Zustände, Eltern, N, W, Entstehungsiteration), beste Route, Anytime-Folge, Knotenbesuche,
Rollout-Schritte, Ziel-Rollouts, Wiedergabe-Rewards und robusten Pfad Wert für Wert reproduzieren. Zusätzlich:
Pfadkosten über networkx-Gewichte, Untergrenze durch networkx-Dijkstra, UCB1 von Hand."""

import math
import random

import numpy as np
import pytest

import mcts_algorithm as A
import mcts_graph as G
import mcts_scenario as S

nx = pytest.importorskip("networkx")
INF = math.inf


class _Node:
    def __init__(self, state, parent, g, untried, born, idx):
        self.state, self.parent, self.g, self.untried = state, parent, g, untried
        self.children, self.N, self.W, self.born, self.idx = [], 0, 0.0, born, idx


def _clean_mcts(graph, start, goal, iters, seed, c=1.0, horizon=60, rollout="uniform", reward="goal", tau=0.5):
    nb, wt = graph.neighbors, graph.weights
    wmap = {(u, v): w for u in range(graph.n) for v, w in zip(nb[u], wt[u])}
    hh = [float(x) for x in np.hypot(*(graph.xy - graph.xy[goal]).T)]
    allw = [w for u in range(graph.n) for w in wt[u]]
    unit = sum(allw) / len(allw) if allw else 1.0
    d0 = hh[start]
    rng = random.Random(int(seed) * 1000003 + 12345)
    nodes = [_Node(start, None, 0.0, list(nb[start]), 0, 0)]
    best = {"cost": INF, "path": [], "any": []}
    st = {"visits": 0, "rsteps": 0, "groll": 0}
    rewards = []

    def path_of(nd):
        out = []
        while nd is not None:
            out.append(nd.state)
            nd = nd.parent
        return out[::-1]

    def reg(cost, path, it):
        if cost < best["cost"] - 1e-12:
            best["cost"], best["path"] = cost, path
            best["any"].append((it, st["visits"], cost))

    if start == goal:
        reg(0.0, [start], 0)
    else:
        for it in range(1, iters + 1):
            nd = nodes[0]
            while True:
                st["visits"] += 1
                if nd.state == goal:
                    r = min(1.0, d0 / nd.g) if nd.g > 0 else 1.0
                    st["groll"] += 1
                    reg(nd.g, path_of(nd), it)
                    break
                if nd.untried:
                    s = nd.untried.pop(int(rng.random() * len(nd.untried)))
                    anc = set(path_of(nd))
                    child = _Node(s, nd, nd.g + wmap[(nd.state, s)], [v for v in nb[s] if v not in anc | {s}], it, len(nodes))
                    nodes.append(child)
                    nd.children.append(child)
                    st["visits"] += 1
                    cur, cost, seen, roll = s, child.g, anc | {s}, []
                    while cur != goal and len(roll) < horizon:
                        cand = [v for v in nb[cur] if v not in seen]
                        if not cand:
                            break
                        if rollout == "guided":
                            m = min(hh[v] for v in cand)
                            ws = [math.exp(-(hh[v] - m) / (tau * unit)) for v in cand]
                            u = rng.random() * sum(ws)
                            k, acc = 0, ws[0]
                            while acc < u and k < len(cand) - 1:
                                k += 1
                                acc += ws[k]
                        else:
                            k = int(rng.random() * len(cand))
                        nxt = cand[k]
                        cost += wmap[(cur, nxt)]
                        cur = nxt
                        seen.add(cur)
                        roll.append(cur)
                    st["rsteps"] += len(roll)
                    st["visits"] += len(roll)
                    if cur == goal:
                        st["groll"] += 1
                        r = min(1.0, d0 / cost) if cost > 0 else 1.0
                        reg(cost, path_of(child) + roll, it)
                    elif reward == "progress" and d0 > 0:
                        r = 0.5 * max(0.0, 1 - hh[cur] / d0)
                    else:
                        r = 0.0
                    nd = child
                    break
                if not nd.children:
                    r = 0.0
                    break
                scores = [ch.W / ch.N + c * math.sqrt(math.log(nd.N) / ch.N) for ch in nd.children]
                nd = nd.children[scores.index(max(scores))]
            rewards.append(r)
            while nd is not None:
                nd.N += 1
                nd.W += r
                nd = nd.parent
    nd, rob = nodes[0], [start]
    while nd.children:
        nd = max(nd.children, key=lambda ch: (ch.N, -ch.idx))
        rob.append(nd.state)
    return nodes, best, st, rewards, rob


def _compare(graph, s, t, iters, seed, **kw):
    r = A.mcts_search(graph, s, t, iters, seed=seed, **kw)
    nodes, best, st, rewards, rob = _clean_mcts(graph, s, t, iters, seed, **kw)
    assert r.tree_state == [n.state for n in nodes]
    assert r.tree_parent == [-1 if n.parent is None else n.parent.idx for n in nodes]
    assert r.tree_visits == [n.N for n in nodes] and r.tree_iteration == [n.born for n in nodes]
    assert all(a == pytest.approx(n.W, abs=1e-9) for a, n in zip(r.tree_reward, nodes))
    assert r.path == best["path"] and r.cost == pytest.approx(best["cost"], abs=1e-9)
    assert [(a, b) for a, b, _ in r.anytime] == [(a, b) for a, b, _ in best["any"]]
    assert (r.visits_total, r.rollout_steps, r.goal_rollouts) == (st["visits"], st["rsteps"], st["groll"])
    assert all(x[2] == pytest.approx(y, abs=1e-9) for x, y in zip(r.replay, rewards))
    assert r.robust_path == (rob if rob[-1] == t else [])
    g = nx.Graph()
    g.add_nodes_from(range(graph.n))
    for u in range(graph.n):
        for v, w in zip(graph.neighbors[u], graph.weights[u]):
            g.add_edge(u, v, weight=w)
    if r.solved:
        assert sum(g[u][v]["weight"] for u, v in zip(r.path[:-1], r.path[1:])) == pytest.approx(r.cost, abs=1e-9)
        assert r.cost >= nx.dijkstra_path_length(g, s, t) - 1e-9
    else:
        assert r.cost == INF


def test_grid_runs_match_the_clean_room_implementation():
    rnd = random.Random(11)
    for _ in range(40):
        inst = S.grid_instance(rnd.randint(2, 6), rnd.choice([0, 15, 40]), rnd.randint(0, 10**6))
        kw = dict(c=rnd.choice([0.0, 0.25, 1.0, 4.0]), horizon=rnd.choice([1, 3, 10, 60]),
                  rollout=rnd.choice(["uniform", "guided"]), reward=rnd.choice(["goal", "progress"]))
        _compare(inst.graph, inst.start, inst.goal, rnd.choice([1, 20, 150, 300]), rnd.randint(0, 99), **kw)


def test_random_graphs_with_dead_ends_and_components_match_the_clean_room_implementation():
    rnd = random.Random(12)
    for _ in range(80):
        n = rnd.randint(1, 8)
        xy = [(rnd.randint(0, 4), rnd.randint(0, 4)) for _ in range(n)]
        edges = [(u, v, float(rnd.randint(1, 6))) for u, v in
                 ((rnd.randrange(n), rnd.randrange(n)) for _ in range(rnd.randint(0, 2 * n))) if u != v]
        kw = dict(c=rnd.choice([0.0, 1.0, 3.0]), horizon=rnd.choice([1, 2, 10]),
                  rollout=rnd.choice(["uniform", "guided"]), reward=rnd.choice(["goal", "progress"]))
        _compare(G.from_edges(n, xy, edges), rnd.randrange(n), rnd.randrange(n), rnd.choice([1, 30, 150]), rnd.randint(0, 99), **kw)


def test_ucb1_by_hand_and_trap_converges_to_the_dijkstra_optimum():
    assert A.ucb1(3.0, 4, 10, 1.0) == pytest.approx(0.75 + math.sqrt(math.log(10) / 4), abs=1e-15)
    inst = S.trap_instance()
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    for seed in range(5):
        assert A.mcts_search(inst.graph, inst.start, inst.goal, 200, seed=seed).cost == pytest.approx(ucs.cost, abs=1e-9)
