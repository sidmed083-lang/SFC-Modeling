"""Analysis of financial flow / exposure networks."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def adjacency(g: nx.DiGraph) -> pd.DataFrame:
    """Weighted adjacency matrix (row = source node, column = target node)."""
    nodes = list(g.nodes)
    return pd.DataFrame(nx.to_numpy_array(g, nodelist=nodes, weight="weight"), index=nodes, columns=nodes)


def node_summary(g: nx.DiGraph) -> pd.DataFrame:
    """Out/in strength, net position and weighted PageRank for every node."""
    out_s = dict(g.out_degree(weight="weight"))
    in_s = dict(g.in_degree(weight="weight"))
    pr = nx.pagerank(g, weight="weight") if g.number_of_edges() else {n: 1 / len(g) for n in g}
    df = pd.DataFrame(
        {"out_strength": out_s, "in_strength": in_s, "pagerank": pr}, index=list(g.nodes)
    )
    df["net_in_minus_out"] = df["in_strength"] - df["out_strength"]
    return df


def edge_table(g: nx.DiGraph) -> pd.DataFrame:
    rows = [
        {"source": u, "target": v, "weight": d["weight"],
         "components": ", ".join(f"{k}: {x:.4g}" for k, x in d["components"].items())}
        for u, v, d in g.edges(data=True)
    ]
    return pd.DataFrame(rows, columns=["source", "target", "weight", "components"])


def propagate_shock(
    g: nx.DiGraph,
    origin: str,
    amount: float,
    rounds: int = 5,
    pass_through: float = 1.0,
    direction: str = "downstream",
) -> pd.DataFrame:
    """Stylised propagation of a shock of size ``amount`` hitting ``origin``.

    ``direction="downstream"`` follows edges (flow networks: a fall in the
    origin's payments reduces receipts of its payees, who cut their own
    payments in proportion).  ``"upstream"`` goes against the edges (stock
    networks: a loss at a debtor is borne by its creditors in proportion to
    their claims).  ``pass_through`` in ``[0, 1]`` is the share of a received
    shock that is transmitted onwards (the rest is absorbed).

    Returns a DataFrame indexed by round (0 = initial hit) plus a final
    ``cumulative`` row.  This is an exposure illustration, not a behavioural
    model.
    """
    if origin not in g:
        raise KeyError(f"Unknown node '{origin}'.")
    if direction not in ("downstream", "upstream"):
        raise ValueError("direction must be 'downstream' or 'upstream'.")
    if not 0.0 <= pass_through <= 1.0:
        raise ValueError("pass_through must lie in [0, 1].")
    nodes = list(g.nodes)
    w = adjacency(g).to_numpy()
    mat = w if direction == "downstream" else w.T  # mat[i, j]: transmission i -> j
    row_sums = mat.sum(axis=1, keepdims=True)
    share = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    vec = np.zeros(len(nodes))
    vec[nodes.index(origin)] = amount
    history = [vec.copy()]
    for _ in range(rounds):
        vec = pass_through * (vec @ share)
        history.append(vec.copy())
    out = pd.DataFrame(history, columns=nodes)
    out.index.name = "round"
    out.loc["cumulative"] = out.sum(axis=0)
    return out
