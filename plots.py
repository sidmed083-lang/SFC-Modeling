"""Plotly figures.  Plotly is imported lazily so the core library does not need it."""

from __future__ import annotations

import math
import networkx as nx
import pandas as pd


def _go():
    import plotly.graph_objects as go

    return go


def line_figure(df: pd.DataFrame, title: str = "", yaxis: str = ""):
    go = _go()
    fig = go.Figure()
    for col in df.columns:
        name = " / ".join(map(str, col)) if isinstance(col, tuple) else str(col)
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name))
    fig.update_layout(title=title, xaxis_title="period", yaxis_title=yaxis, hovermode="x unified")
    return fig


def bar_figure(df: pd.DataFrame, title: str = "", yaxis: str = ""):
    go = _go()
    fig = go.Figure([go.Bar(x=df.index, y=df[c], name=str(c)) for c in df.columns])
    fig.update_layout(barmode="relative", title=title, xaxis_title="period", yaxis_title=yaxis)
    return fig


def matrix_heatmap(frame: pd.DataFrame, title: str = ""):
    """Heatmap of a TFM / BSM frame; sign is shown by colour (diverging scale)."""
    go = _go()
    z = frame.to_numpy(dtype=float)
    zmax = float(abs(z).max()) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(map(str, frame.columns)), y=list(map(str, frame.index)),
            zmin=-zmax, zmax=zmax, colorscale="RdBu", text=[[f"{v:.2f}" for v in r] for r in z],
            texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"))
    return fig


def network_figure(g: nx.DiGraph, title: str = "", seed: int = 1, min_share: float = 0.0):
    """Weighted directed graph: edge width ~ weight, node size ~ total strength."""
    go = _go()
    pos = nx.circular_layout(g) if len(g) <= 8 else nx.spring_layout(g, seed=seed, weight="weight")
    weights = [d["weight"] for _, _, d in g.edges(data=True)]
    wmax = max(weights) if weights else 1.0
    fig = go.Figure()
    for u, v, d in g.edges(data=True):
        if d["weight"] < min_share * wmax:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        width = 1 + 7 * d["weight"] / wmax
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=width, color="rgba(90,90,90,0.5)"),
            hoverinfo="text", showlegend=False,
            text=f"{u} → {v}: {d['weight']:.4g}<br>" + "<br>".join(f"{k}: {x:.4g}" for k, x in d["components"].items()),
        ))
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1, opacity=0.6,
                           standoff=18)
    strength = dict(g.degree(weight="weight"))
    smax = max(strength.values()) if strength and max(strength.values()) > 0 else 1.0
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in g], y=[pos[n][1] for n in g], mode="markers+text",
        text=list(g.nodes), textposition="top center",
        marker=dict(size=[18 + 30 * math.sqrt(strength[n] / smax) for n in g], color="#1f77b4"),
        hovertext=[f"{n}<br>strength: {strength[n]:.4g}" for n in g], hoverinfo="text", showlegend=False,
    ))
    kind = g.graph.get("kind", "")
    fig.update_layout(
        title=f"{title} [{kind}]" if kind else title,
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
