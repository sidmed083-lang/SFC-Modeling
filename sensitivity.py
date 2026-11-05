"""Sensitivity analysis and Monte Carlo simulation (seeded, reproducible)."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..model.specification import ModelSpecification
from ..solver.diagnostics import ConvergenceError
from .engine import SimulationResult, simulate
from .scenarios import Scenario, Shock

Metric = Callable[[SimulationResult], float]


def final_value(variable: str) -> Metric:
    return lambda r: float(r.data[variable].iloc[-1])


def value_at(variable: str, period: int) -> Metric:
    return lambda r: float(r.data.loc[period, variable])


def one_at_a_time(
    spec: ModelSpecification,
    target: str,
    values: Sequence[float],
    metric: Metric,
    periods: int,
) -> pd.DataFrame:
    """Sweep one parameter/exogenous input over ``values`` (permanent from t=1)."""
    model = spec.compile()
    rows = []
    for v in values:
        res = simulate(model, periods, Scenario(f"{target}={v}", [Shock(target, 1, value=float(v))]))
        rows.append({target: float(v), "metric": metric(res)})
    return pd.DataFrame(rows)


def _draw(rng: np.random.Generator, dist: tuple, n: int) -> np.ndarray:
    kind, *args = dist
    if kind == "normal":
        return rng.normal(args[0], args[1], n)
    if kind == "uniform":
        return rng.uniform(args[0], args[1], n)
    if kind == "triangular":
        return rng.triangular(args[0], args[1], args[2], n)
    raise ValueError(f"Unknown distribution '{kind}' (use normal, uniform, triangular).")


def monte_carlo(
    spec: ModelSpecification,
    distributions: Mapping[str, tuple],
    metric: Metric,
    periods: int,
    n: int = 200,
    seed: int | None = None,
) -> pd.DataFrame:
    """Draw inputs, simulate, collect ``metric``.

    ``distributions`` maps a parameter/exogenous name to ``("normal", mean, sd)``,
    ``("uniform", lo, hi)`` or ``("triangular", lo, mode, hi)``.  The seed
    defaults to ``spec.random_seed``.  Draws whose simulation fails to
    converge are kept with ``metric = NaN`` and ``converged = False``.
    """
    seed = spec.random_seed if seed is None else seed
    rng = np.random.default_rng(seed)
    draws = {k: _draw(rng, d, n) for k, d in distributions.items()}
    model = spec.compile()
    rows = []
    for i in range(n):
        shocks = [Shock(k, 1, value=float(draws[k][i])) for k in draws]
        row = {k: float(draws[k][i]) for k in draws}
        try:
            row["metric"] = metric(simulate(model, periods, Scenario(f"draw{i}", shocks)))
            row["converged"] = True
        except ConvergenceError:
            row["metric"], row["converged"] = float("nan"), False
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["seed"] = seed
    return out


def summarize(draws: pd.DataFrame, quantiles: Iterable[float] = (0.05, 0.5, 0.95)) -> pd.Series:
    m = draws["metric"].dropna()
    s = {"mean": m.mean(), "std": m.std(ddof=1) if len(m) > 1 else 0.0}
    s.update({f"q{int(q * 100):02d}": m.quantile(q) for q in quantiles})
    s["n_failed"] = int((~draws["converged"]).sum())
    return pd.Series(s)
