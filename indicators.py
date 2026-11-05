"""Macroeconomic indicators computed from simulation output."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from ..simulation.engine import SimulationResult


def growth_rate(series: pd.Series) -> pd.Series:
    """Period-on-period growth; undefined (NaN) where the base is zero."""
    base = series.shift(1)
    return (series / base - 1.0).where(base != 0)


def ratio(df: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    den = df[denominator]
    return (df[numerator] / den).where(den != 0)


def summary_table(
    result: SimulationResult, variables: Iterable[str], periods: Optional[Iterable[int]] = None
) -> pd.DataFrame:
    """Selected variables at selected periods (default: first, middle, last)."""
    variables = list(variables)
    if periods is None:
        T = result.periods
        periods = sorted({0, 1, max(T // 2, 1), T})
    return result.data.loc[list(periods), variables]


def convergence_period(series: pd.Series, tol: float = 1e-3) -> Optional[int]:
    """First period after which ``|x_t - x_T| <= tol * max(1, |x_T|)`` always holds."""
    final = float(series.iloc[-1])
    ok = (series - final).abs() <= tol * max(1.0, abs(final))
    for t in series.index:
        if ok.loc[t:].all():
            return int(t)
    return None
