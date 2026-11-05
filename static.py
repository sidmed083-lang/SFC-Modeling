"""Static equilibrium (steady-state) solution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..model.specification import CompiledModel


@dataclass
class SteadyStateResult:
    values: pd.Series
    converged: bool
    residual_norm: float
    jacobian_rank: int
    n_unknowns: int
    singular_values: np.ndarray
    message: str

    @property
    def unique(self) -> bool:
        """True when the Jacobian has full column rank (locally unique solution)."""
        return self.jacobian_rank == self.n_unknowns


def solve_steady_state(
    model: CompiledModel,
    params: Optional[Mapping[str, float]] = None,
    exog: Optional[Mapping[str, float]] = None,
    guess: Optional[Mapping[str, float]] = None,
    tol: float = 1e-10,
    use_consistency_checks: bool = True,
) -> SteadyStateResult:
    """Solve for the stationary state where every lag equals the current value.

    Stock-accumulation equations (``H = H(-1) + YD - C``) become flow conditions
    (``YD = C``) and no longer pin down the stock.  The model's *consistency
    checks* (the redundant identity, e.g. ``Hs - Hh``) are therefore appended as
    extra residuals, which restores uniqueness in properly closed SFC models.
    The result reports the Jacobian rank so non-uniqueness is visible.
    """
    p = dict(model.parameters if params is None else params)
    spec = model.spec
    e = {n: spec.exogenous_value(n, 10**6) for n in model.exogenous}
    if exog:
        e.update({k: float(v) for k, v in exog.items()})
    endo = model.endogenous
    guess = guess or {}
    # Start from the supplied guess / initial value; zeros are replaced by 1.0
    # because behavioural equations often divide by stocks (e.g. ``YD/V``).
    x0 = np.array(
        [guess.get(v, spec.initial.get(v, 0.0)) or 1.0 for v in endo], dtype=float
    )
    base = {**p, **e}
    exprs = [model.steady_expressions[v] for v in endo]
    checks = model.steady_checks if use_consistency_checks else []

    def resid(x: np.ndarray) -> np.ndarray:
        ctx = dict(base)
        ctx.update(zip(endo, map(float, x)))
        r = [xv - ex(ctx) for xv, ex in zip(x, exprs)]
        r += [c(ctx) for c in checks]
        return np.array(r)

    sol = least_squares(resid, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    r = resid(sol.x)
    # SciPy's finite-difference Jacobian is too noisy for a rank decision, so the
    # Jacobian is re-estimated here by central differences at the solution.
    jac = np.empty((len(r), len(sol.x)))
    for j in range(len(sol.x)):
        h = 1e-6 * max(1.0, abs(sol.x[j]))
        up, dn = sol.x.copy(), sol.x.copy()
        up[j] += h
        dn[j] -= h
        jac[:, j] = (resid(up) - resid(dn)) / (2 * h)
    sv = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(sv > 1e-7 * max(sv.max(), 1.0)))
    rnorm = float(np.max(np.abs(r)))
    return SteadyStateResult(
        values=pd.Series(sol.x, index=endo),
        converged=bool(rnorm <= tol * max(1.0, float(np.max(np.abs(sol.x))))),
        residual_norm=rnorm,
        jacobian_rank=rank,
        n_unknowns=len(endo),
        singular_values=sv,
        message=str(sol.message),
    )
