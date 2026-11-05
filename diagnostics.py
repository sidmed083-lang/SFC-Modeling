"""Convergence diagnostics and numerical error reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class ConvergenceError(RuntimeError):
    """Raised when a block of simultaneous equations cannot be solved."""


@dataclass
class BlockDiagnostic:
    period: int
    block: int
    variables: str
    simultaneous: bool
    method: str
    success: bool
    nfev: int
    max_abs_residual: float


@dataclass
class DiagnosticsLog:
    records: list[BlockDiagnostic] = field(default_factory=list)
    period_residuals: dict[int, float] = field(default_factory=dict)

    def add(self, rec: BlockDiagnostic) -> None:
        self.records.append(rec)

    def to_frame(self) -> pd.DataFrame:
        cols = [
            "period", "block", "variables", "simultaneous", "method",
            "success", "nfev", "max_abs_residual",
        ]
        return pd.DataFrame([[getattr(r, c) for c in cols] for r in self.records], columns=cols)

    @property
    def all_converged(self) -> bool:
        return all(r.success for r in self.records)

    def summary(self) -> dict[str, Any]:
        df = self.to_frame()
        if df.empty:
            return {"blocks_solved": 0, "all_converged": True, "max_abs_residual": 0.0}
        return {
            "blocks_solved": int(len(df)),
            "simultaneous_blocks_solved": int(df["simultaneous"].sum()),
            "all_converged": bool(df["success"].all()),
            "max_abs_residual": float(df["max_abs_residual"].max()),
            "max_period_residual": float(max(self.period_residuals.values(), default=0.0)),
            "total_function_evaluations": int(df["nfev"].sum()),
        }


def stability_analysis(model, params: dict, exog_values: dict, steady: dict, eps: float = 1e-6):
    """Local stability of the period map around ``steady``.

    The state is the set of endogenous variables that appear lagged.  The
    Jacobian of ``x_t = F(x_{t-1})`` is obtained by central differences and its
    eigenvalues are returned.  Accounting-redundant stocks (e.g. the government's
    money liability mirroring households' money holdings) generate eigenvalues
    equal to one; they are flagged in ``n_unit_roots``.
    """
    from .dynamic import build_context, solve_period

    states = model.state_variables
    if not states:
        return {"states": [], "eigenvalues": np.array([]), "max_modulus_non_unit": 0.0,
                "n_unit_roots": 0, "stable": True}

    def step(prev: dict) -> dict:
        history = {0: dict(prev)}
        ctx = build_context(model, params, exog_values, history, 1)
        guess = dict(prev)
        solve_period(model, ctx, guess, tol=1e-13, log=DiagnosticsLog(), period=1)
        return {v: ctx[v] for v in model.endogenous}

    jac = np.zeros((len(states), len(states)))
    for j, sj in enumerate(states):
        hi, lo = dict(steady), dict(steady)
        h = eps * max(1.0, abs(steady[sj]))
        hi[sj] += h
        lo[sj] -= h
        fh, fl = step(hi), step(lo)
        for i, si in enumerate(states):
            jac[i, j] = (fh[si] - fl[si]) / (2 * h)
    eig = np.linalg.eigvals(jac)
    unit = np.isclose(np.abs(eig), 1.0, atol=1e-6)
    non_unit = np.abs(eig[~unit])
    return {
        "states": states,
        "jacobian": jac,
        "eigenvalues": eig,
        "n_unit_roots": int(unit.sum()),
        "max_modulus_non_unit": float(non_unit.max()) if len(non_unit) else 0.0,
        "stable": bool(np.all(np.abs(eig[~unit]) < 1.0 - 1e-9)),
    }
