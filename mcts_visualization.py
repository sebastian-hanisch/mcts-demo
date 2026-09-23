"""Plotly-Abbildungen: Instanz (Raster oder handgebaute Falle mit Knotennamen), Besuchskarte des Baums (Größe/Farbe = Besuche je Zustand, dazu beste
gefundene, robuste und optimale Route), Wiedergabe des Baumwachstums je Iteration mit dem letzten Rollout, Anytime-Kurven, Sweeps, Anteile und
Ablations-Balken. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NODE_COLOR = "#4c78a8"
BLOCKED_COLOR = "#9d755d"
OPT_COLOR = "#4c78a8"
BEST_COLOR = "#e45756"
ROBUST_COLOR = "#54a24b"
ROLLOUT_COLOR = "#f58518"
GREY = "rgba(120,120,120,0.55)"
VARIANT_COLORS = {"pure": "#4c78a8", "reward": "#f58518", "rollout": "#54a24b", "both": "#7b3fbf"}
INF = float("inf")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _hand(inst):
    return inst.labels is not None


def _panel_base(fig, inst, node_alpha=0.3):
    """Hintergrund der Karte: bei der Falle Kanten und Knotennamen, sonst Raster-Zellen und Hindernisse."""
    xy = inst.graph.xy
    if _hand(inst):
        ex, ey = [], []
        for u in range(inst.graph.n):
            for v in inst.graph.neighbors[u]:
                if u < v:
                    ex += [xy[u, 0], xy[v, 0], None]
                    ey += [xy[u, 1], xy[v, 1], None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(120,120,120,0.5)", width=1.5), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=list(inst.labels), textposition="top center", textfont=dict(size=10, color="#888"),
                                 marker=dict(size=8, color="rgba(120,120,120,0.4)"), hoverinfo="skip", showlegend=False))
    else:
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=5, color=f"rgba(76,120,168,{node_alpha})"), hoverinfo="skip", showlegend=False))
        if len(inst.blocked_xy):
            fig.add_trace(go.Scatter(x=inst.blocked_xy[:, 0], y=inst.blocked_xy[:, 1], mode="markers", marker=dict(size=6, symbol="square", color=BLOCKED_COLOR),
                                     name="Hindernis", showlegend=False))


def _start_goal(fig, inst):
    xy = inst.graph.xy
    fig.add_trace(go.Scatter(x=[xy[inst.start, 0]], y=[xy[inst.start, 1]], mode="markers", marker=dict(size=16, symbol="star", color="#2ca02c", line=dict(width=1, color="white")),
                             name="Start", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xy[inst.goal, 0]], y=[xy[inst.goal, 1]], mode="markers", marker=dict(size=16, symbol="star", color="#d62728", line=dict(width=1, color="white")),
                             name="Ziel", hoverinfo="skip"))


def _map_axes(fig, inst, height):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    if not _hand(inst):
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return _base(fig, height)


def _line(fig, inst, path, name, color, width, dash="solid"):
    if path:
        p = inst.graph.xy[path]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=color, width=width, dash=dash), name=name))


def build_instance(inst):
    fig = go.Figure()
    _panel_base(fig, inst, node_alpha=0.9)
    if not _hand(inst):
        xy = inst.graph.xy
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=6, color=NODE_COLOR, line=dict(width=1, color="white")), name="Offene Zellen"))
    _start_goal(fig, inst)
    return _map_axes(fig, inst, 340 if _hand(inst) else 460)


def build_tree_map(inst, result, opt_path, show_rollouts=True):
    """Besuche je Zustand: Kreisfläche = Besuche im Baum, Farbe = Rollout-Besuche (logarithmisch); darüber die beste gefundene Route, der robuste Pfad
    (jeweils am häufigsten besuchtes Kind) und das Optimum."""
    fig = go.Figure()
    _panel_base(fig, inst)
    xy = inst.graph.xy
    tree = np.asarray(result.state_visits)
    roll = np.asarray(result.rollout_visits)
    touched = np.flatnonzero(tree + roll > 0)
    if len(touched):
        size = 6 + 20 * np.sqrt(tree[touched] / max(1.0, tree.max()))
        colors = np.log10(1 + roll[touched]) if show_rollouts else np.log10(1 + tree[touched])
        fig.add_trace(go.Scatter(x=xy[touched, 0], y=xy[touched, 1], mode="markers",
                                 marker=dict(size=size, color=colors, colorscale="Blues", cmin=0, showscale=True, opacity=0.85, line=dict(width=0.5, color="white"),
                                             colorbar=dict(title="log10(1 + Rollout-Besuche)", thickness=10, len=0.6)),
                                 name="Besuche", hoverinfo="skip", showlegend=False))
    _line(fig, inst, opt_path, "Optimum (Uniform-Cost)", "rgba(76,120,168,0.35)", 12)
    _line(fig, inst, result.robust_path, "Robuster Pfad (meistbesuchte Kinder)", ROBUST_COLOR, 3, "dash")
    _line(fig, inst, result.path, "Beste gefundene Route", BEST_COLOR, 3)
    _start_goal(fig, inst)
    return _map_axes(fig, inst, 340 if _hand(inst) else 480)


def build_replay(inst, result, k):
    """Baum nach den ersten k Iterationen (nur aufgezeichnete Iterationen): Baumkanten, der in Iteration k neu angehängte Knoten und der letzte Rollout."""
    fig = go.Figure()
    _panel_base(fig, inst)
    xy = inst.graph.xy
    k = max(0, min(k, len(result.replay)))
    ex, ey = [], []
    for i in range(1, result.tree_size):
        if result.tree_iteration[i] <= k:
            u, v = result.tree_state[result.tree_parent[i]], result.tree_state[i]
            ex += [xy[u, 0], xy[v, 0], None]
            ey += [xy[u, 1], xy[v, 1], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(76,120,168,0.55)", width=1.5), name="Baumkanten", hoverinfo="skip"))
    if k >= 1:
        new_node, roll, reward = result.replay[k - 1]
        if new_node != -1:
            head = result.tree_state[new_node]
            chain = [result.tree_state[result.tree_parent[new_node]], head] + list(roll)
            p = xy[chain]
            if roll:
                fig.add_trace(go.Scatter(x=p[1:, 0], y=p[1:, 1], mode="lines", line=dict(color=ROLLOUT_COLOR, width=2.5, dash="dot"), name=f"Rollout (Reward {reward:.2f})"))
            fig.add_trace(go.Scatter(x=[xy[head, 0]], y=[xy[head, 1]], mode="markers", marker=dict(size=13, color=ROLLOUT_COLOR, line=dict(width=1, color="white")),
                                     name="Neuer Knoten", hoverinfo="skip"))
    _start_goal(fig, inst)
    return _map_axes(fig, inst, 340 if _hand(inst) else 480)


def build_anytime(curves, optimum, ucs_expansions, astar_expansions, max_visits, unit_note="Knotenbesuche"):
    """Beste gefundene Kosten über die Knotenbesuche für mehrere Ketten-Seeds (Treppen), das Optimum gestrichelt, dazu senkrecht der Aufwand von Uniform-Cost und A*
    in Expansionen (andere Einheit - nur der Größenordnung nach vergleichbar)."""
    fig = go.Figure()
    for i, curve in enumerate(curves):
        if not curve:
            continue
        xs = [v for _it, v, _c in curve] + [max_visits]
        ys = [c for _it, _v, c in curve] + [curve[-1][2]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1.8, shape="hv", color=f"hsl({(i * 47) % 360},55%,45%)"), name=f"Seed {i + 1}", showlegend=len(curves) <= 8))
    fig.add_hline(y=optimum, line=dict(color=OPT_COLOR, dash="dash", width=1.5), annotation_text="Optimum", annotation_position="bottom right")
    fig.add_vline(x=ucs_expansions, line=dict(color="#888", dash="dot", width=1.5), annotation_text=f"UCS: {ucs_expansions:g} Expansionen", annotation_position="top left")
    if astar_expansions is not None:
        fig.add_vline(x=astar_expansions, line=dict(color="#54a24b", dash="dot", width=1.5), annotation_text=f"A*: {astar_expansions:g}", annotation_position="bottom left")
    fig.update_xaxes(title_text=f"{unit_note} (logarithmisch)", type="log", range=[0, np.log10(max(10.0, max_visits * 1.2))])
    fig.update_yaxes(title_text="Kosten der besten Route")
    return _base(fig, 380, legend_y=-0.3)


def build_sweep(rows, param_label, series, y_label, ref_line=None, ref_label=None):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    for key, name, color in series:
        ys = [r[key] for r in rows]
        lo = [r[f"{key}_lo"] for r in rows]
        hi = [r[f"{key}_hi"] for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color="#888", dash="dash", width=1.5), annotation_text=ref_label, annotation_position="top left")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label)
    return _base(fig, 360, legend_y=-0.3)


def build_share_bars(rows, param_label):
    """Anteil ALLER Läufe: mit Lösung, Lücke <= 5 %, optimal."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("solved_share", "Lösung gefunden", "#4c78a8"), ("near_share", "Lücke ≤ 5 %", "#f58518"), ("optimal_share", "optimal", "#54a24b")):
        fig.add_trace(go.Bar(x=xs, y=[r[key] for r in rows], name=name, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text="Anteil der Läufe (%)", range=[0, 100])
    return _base(fig, 340, legend_y=-0.35)


def build_ablation_bars(result, labels):
    """Vier Varianten (rein / Reward / Rollout / beides): links Anteile (Lösung, nah, optimal), rechts Lücke im Median und Ziel-Rollouts."""
    names = [labels[v] for v in result]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Anteil der Läufe (%)", "Ziel-Rollouts (% der Iterationen)"), horizontal_spacing=0.1)
    for key, name, color in (("solved_share", "Lösung gefunden", "#4c78a8"), ("near_share", "Lücke ≤ 5 %", "#f58518"), ("optimal_share", "optimal", "#54a24b")):
        fig.add_trace(go.Bar(x=names, y=[result[v][key] for v in result], name=name, marker_color=color), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[result[v]["goal_share"] for v in result], name="Ziel-Rollouts", marker_color=[VARIANT_COLORS[v] for v in result], showlegend=False), row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    return _base(fig, 360, legend_y=-0.4)
