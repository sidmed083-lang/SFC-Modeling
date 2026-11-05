"""Simulation engine: solve, build accounting matrices, validate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..accounting.bsm import BalanceSheetMatrix
from ..accounting.validation import (
    ValidationReport,
    check_bsm,
    check_stock_flow,
    check_tfm,
)
from ..model.specification import CompiledModel, ModelSpecification
from ..solver.diagnostics import DiagnosticsLog
from ..solver.dynamic import context_from_frame, solve_dynamic
from .scenarios import Scenario, baseline, build_input_paths


@dataclass
class SimulationResult:
    model: CompiledModel
    scenario: Scenario
    data: pd.DataFrame  # periods 0..T; endogenous + exogenous
    parameters: pd.DataFrame  # periods 0..T
    log: DiagnosticsLog
    tol: float
    _accounting: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def periods(self) -> int:
        return int(self.data.index[-1])

    @property
    def diagnostics(self) -> pd.DataFrame:
        return self.log.to_frame()

    # ------------------------------------------------------- consistency
    def consistency_residuals(self) -> pd.DataFrame:
        """Redundant-identity residuals (e.g. ``Hs - Hh``) for periods 1..T."""
        rows = {}
        for t in range(1, self.periods + 1):
            ctx = context_from_frame(self.model, self.data, self.parameters, t)
            rows[t] = {c.text: c(ctx) for c in self.model.checks}
        return pd.DataFrame.from_dict(rows, orient="index", columns=[c.text for c in self.model.checks])

    # --------------------------------------------------------- accounting
    @property
    def has_accounting(self) -> bool:
        return self.model.spec.accounting is not None

    def _compiled_accounting(self):
        if self._accounting is None:
            m = self.model
            self._accounting = m.spec.accounting.compile(
                set(m.endogenous) | set(m.exogenous) | set(m.parameters),
                set(m.endogenous) | set(m.exogenous),
            )
        return self._accounting

    def matrices(self, t: int):
        """``(tfm, bsm, revaluations, nonfinancial_acquisitions)`` at period ``t``."""
        if not self.has_accounting:
            raise ValueError("The model has no accounting specification.")
        if not 0 <= t <= self.periods:
            raise IndexError(f"Period {t} outside 0..{self.periods}.")
        acc = self._compiled_accounting()
        ctx = context_from_frame(self.model, self.data, self.parameters, t, acc.lag_symbols)
        return acc.build(ctx)

    def validate(self, atol: float = 1e-8, rtol: float = 1e-9) -> ValidationReport:
        """Validate every accounting identity in every period."""
        rep = ValidationReport(atol, rtol)
        if not self.has_accounting:
            rep.fail("accounting.missing", "-", "No accounting specification to validate.")
            return rep
        prev_bsm: Optional[BalanceSheetMatrix] = None
        for t in range(0, self.periods + 1):
            tfm, bsm, reval, nfa = self.matrices(t)
            rep.merge(check_bsm(bsm, atol, rtol), f"t={t}: ")
            if t >= 1:
                rep.merge(check_tfm(tfm, atol, rtol), f"t={t}: ")
                rep.merge(check_stock_flow(prev_bsm, bsm, tfm, reval, nfa, atol, rtol), f"t={t}: ")
            prev_bsm = bsm
        return rep


def simulate(
    spec: ModelSpecification | CompiledModel,
    periods: int,
    scenario: Optional[Scenario] = None,
    tol: float = 1e-9,
    raise_on_failure: bool = True,
) -> SimulationResult:
    """Run a dynamic simulation of ``periods`` periods under ``scenario``."""
    model = spec if isinstance(spec, CompiledModel) else spec.compile()
    scenario = scenario or baseline()
    params, exog = build_input_paths(model, periods, scenario)
    data, log = solve_dynamic(model, params, exog, model.spec.initial, tol, raise_on_failure)
    return SimulationResult(model, scenario, data, params, log, tol)


def compare(results: dict[str, SimulationResult], variables: list[str],
            reference: Optional[str] = None) -> pd.DataFrame:
    """Wide table (period x (scenario, variable)); optional differences vs ``reference``."""
    frames = {name: r.data[variables] for name, r in results.items()}
    if reference is not None:
        ref = frames[reference]
        frames = {name: f - ref for name, f in frames.items() if name != reference}
    return pd.concat(frames, axis=1)
