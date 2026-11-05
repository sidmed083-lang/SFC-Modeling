"""Period-by-period dynamic solution of a compiled model."""

from __future__ import annotations

from typing import Mapping, MutableMapping

import numpy as np
import pandas as pd
from scipy.optimize import root

from ..model.specification import CompiledModel
from .diagnostics import BlockDiagnostic, ConvergenceError, DiagnosticsLog


def build_context(
    model: CompiledModel,
    params: Mapping[str, float],
    exog_row: Mapping[str, float] | pd.DataFrame,
    history: Mapping[int, Mapping[str, float]],
    t: int,
    exog_history: pd.DataFrame | None = None,
    extra_lags: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, float]:
    """Numeric context for period ``t``: parameters, exogenous values, lags.

    ``history[p]`` holds endogenous values of period ``p`` (period 0 is the
    initial condition).  Lags reaching before period 0 reuse period 0.
    ``exog_row`` may be a dict of current values or, with ``exog_history``
    (a DataFrame indexed by period), the exogenous path used for lags.
    """
    ctx: dict[str, float] = dict(params)
    if exog_history is not None:
        for name in model.exogenous:
            ctx[name] = float(exog_history.loc[t, name])
    else:
        ctx.update({k: float(v) for k, v in dict(exog_row).items()})
    lags = dict(model.lag_symbols)
    if extra_lags:
        lags.update(extra_lags)
    for sym, (base, k) in lags.items():
        p = max(t - k, 0)
        if base in model.exogenous:
            ctx[sym] = (
                float(exog_history.loc[p, base]) if exog_history is not None else ctx[base]
            )
        else:
            ctx[sym] = float(history[p][base])
    return ctx


def _solve_block(model, block, ctx, guess, tol, expressions):
    vars_ = block.variables
    if not block.simultaneous:
        v = vars_[0]
        ctx[v] = expressions[v](ctx)
        return "direct", True, 1, 0.0
    exprs = [expressions[v] for v in vars_]

    def resid(x: np.ndarray) -> np.ndarray:
        for v, xv in zip(vars_, x):
            ctx[v] = float(xv)
        return np.array([xv - e(ctx) for xv, e in zip(x, exprs)])

    x0 = np.array([guess.get(v, 0.0) for v in vars_], dtype=float)
    best = None
    for method in ("hybr", "lm"):
        sol = root(resid, x0, method=method, tol=1e-14)
        r = float(np.max(np.abs(resid(sol.x))))
        scale = max(1.0, float(np.max(np.abs(sol.x))))
        ok = bool(np.all(np.isfinite(sol.x))) and r <= tol * scale
        best = (method, ok, int(getattr(sol, "nfev", 0)), r, sol.x)
        if ok:
            break
    method, ok, nfev, r, x = best
    for v, xv in zip(vars_, x):
        ctx[v] = float(xv)
    return method, ok, nfev, r


def solve_period(
    model: CompiledModel,
    ctx: MutableMapping[str, float],
    guess: Mapping[str, float],
    tol: float,
    log: DiagnosticsLog,
    period: int,
    raise_on_failure: bool = True,
    expressions=None,
) -> None:
    """Solve all blocks for one period, writing the solution into ``ctx``."""
    expressions = expressions or model.expressions
    for block in model.blocks:
        method, ok, nfev, resid = _solve_block(model, block, ctx, guess, tol, expressions)
        if not block.simultaneous:
            resid = 0.0
        log.add(
            BlockDiagnostic(period, block.index, ", ".join(block.variables),
                            block.simultaneous, method, ok, nfev, resid)
        )
        if not ok and raise_on_failure:
            raise ConvergenceError(
                f"Period {period}: block {block.variables} did not converge "
                f"(max residual {resid:.3e}, tol {tol:g})."
            )
    log.period_residuals[period] = max(
        abs(ctx[v] - expressions[v](ctx)) for v in model.endogenous
    )


def solve_dynamic(
    model: CompiledModel,
    params: pd.DataFrame,
    exog: pd.DataFrame,
    initial: Mapping[str, float],
    tol: float = 1e-9,
    raise_on_failure: bool = True,
) -> tuple[pd.DataFrame, DiagnosticsLog]:
    """Solve periods ``1..T``.  ``params``/``exog`` are indexed ``0..T``.

    Returns a DataFrame indexed ``0..T`` (endogenous then exogenous columns)
    whose row 0 contains the initial conditions.
    """
    periods = list(params.index)
    history: dict[int, dict[str, float]] = {0: {v: float(initial.get(v, 0.0)) for v in model.endogenous}}
    log = DiagnosticsLog()
    for t in periods[1:]:
        ctx = build_context(model, params.loc[t].to_dict(), None, history, t, exog_history=exog)
        solve_period(model, ctx, history[t - 1], tol, log, t, raise_on_failure)
        history[t] = {v: ctx[v] for v in model.endogenous}
    data = pd.DataFrame.from_dict(history, orient="index")[model.endogenous]
    data = data.join(exog[model.exogenous])
    data.index.name = "period"
    return data, log


def context_from_frame(
    model: CompiledModel,
    data: pd.DataFrame,
    params: pd.DataFrame,
    t: int,
    extra_lags: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, float]:
    """Rebuild the numeric context of period ``t`` from stored results.

    ``extra_lags`` adds lag symbols used outside the equations (accounting templates).
    """
    depth = max([model.max_lag] + [k for _, k in (extra_lags or {}).values()])
    history = {p: data.loc[p, model.endogenous].to_dict() for p in range(max(t - depth, 0), t + 1)}
    history.setdefault(0, data.loc[0, model.endogenous].to_dict())
    ctx = build_context(
        model, params.loc[t].to_dict(), None, history, t, exog_history=data, extra_lags=extra_lags
    )
    for v in model.endogenous:
        ctx[v] = float(data.loc[t, v])
    return ctx
