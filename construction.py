"""Weighted directed graphs of sectoral financial flows and positions.

Two kinds of graph are produced and are always labelled as such through
``graph.graph["kind"]``:

* ``"flow"``  - transactions during a period (a *flow*).  Edge ``u -> v``:
  sector ``u`` pays sector ``v``.  When a transaction has several payers and
  receivers the amount is allocated pro rata:
  ``w(u, v) = |payment_u| * receipt_v / total receipts``.
* ``"stock"`` - financial claims at a point in time (a *stock*).  Edge
  ``u -> v``: sector ``u`` holds a claim on sector ``v`` (``u`` is creditor,
  ``v`` debtor), allocated pro rata over the issuers of each instrument.

Pro-rata allocation is a modelling convention (a TFM/BSM does not record
bilateral counterparties); it is exact whenever a row has one payer/issuer
or one receiver/holder.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from ..accounting.bsm import FINANCIAL as BSM_FINANCIAL
from ..accounting.bsm import BalanceSheetMatrix
from ..accounting.tfm import CAPITAL, CURRENT, TransactionFlowMatrix


def _allocate(g: nx.DiGraph, payers: dict[str, float], receivers: dict[str, float], label: str):
    total = sum(receivers.values())
    if total <= 0:
        return
    for u, pay in payers.items():
        for v, rec in receivers.items():
            if u == v:
                continue
            w = pay * rec / total
            if w <= 0:
                continue
            if g.has_edge(u, v):
                g[u][v]["weight"] += w
                g[u][v]["components"][label] = g[u][v]["components"].get(label, 0.0) + w
            else:
                g.add_edge(u, v, weight=w, components={label: w})


def flow_network(
    tfm: TransactionFlowMatrix, kinds: Iterable[str] = (CURRENT, CAPITAL), tol: float = 1e-12
) -> nx.DiGraph:
    """Sector-to-sector payment network from the TFM rows of the given kinds.

    Use ``kinds=("financial",)`` for the network of financing flows (funds
    move from acquirers of assets to issuers of liabilities).
    """
    kinds = tuple(kinds)
    g = nx.DiGraph(kind="flow", unit="currency per period")
    g.add_nodes_from(tfm.sectors)
    for row in tfm.rows_of_kind(*kinds):
        payers = {s: -v for s, v in row.entries.items() if v < -tol}
        receivers = {s: v for s, v in row.entries.items() if v > tol}
        _allocate(g, payers, receivers, row.name)
    nl = tfm.net_lending()
    for s in tfm.sectors:
        g.nodes[s]["net_lending"] = float(nl[s])
    return g


def exposure_network(bsm: BalanceSheetMatrix, tol: float = 1e-12) -> nx.DiGraph:
    """Creditor -> debtor network of financial claims from the BSM."""
    g = nx.DiGraph(kind="stock", unit="currency")
    g.add_nodes_from(bsm.sectors)
    stocks = bsm.stocks(BSM_FINANCIAL)
    for inst, row in stocks.iterrows():
        creditors = {s: v for s, v in row.items() if v > tol}
        debtors = {s: -v for s, v in row.items() if v < -tol}
        _allocate(g, creditors, debtors, str(inst))
    nfw = bsm.net_financial_wealth()
    for s in bsm.sectors:
        g.nodes[s]["net_financial_wealth"] = float(nfw[s])
    return g
