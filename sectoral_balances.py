"""Sectoral balances over a simulation."""

from __future__ import annotations

import pandas as pd

from ..simulation.engine import SimulationResult


def _series(result: SimulationResult, getter, first: int) -> pd.DataFrame:
    rows = {t: getter(result.matrices(t)) for t in range(first, result.periods + 1)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "period"
    return df


def net_lending(result: SimulationResult) -> pd.DataFrame:
    """Net lending (+) / borrowing (-) by sector, periods 1..T."""
    return _series(result, lambda m: m[0].net_lending(), 1)


def saving(result: SimulationResult) -> pd.DataFrame:
    return _series(result, lambda m: m[0].saving(), 1)


def net_worth(result: SimulationResult) -> pd.DataFrame:
    return _series(result, lambda m: m[1].net_worth(), 0)


def net_financial_wealth(result: SimulationResult) -> pd.DataFrame:
    return _series(result, lambda m: m[1].net_financial_wealth(), 0)


def balance_identity_residual(balances: pd.DataFrame) -> pd.Series:
    """Sum over sectors per period; zero for net lending and net financial wealth."""
    return balances.sum(axis=1)
