"""Automated accounting validation.

Nothing here assumes consistency: every identity is *evaluated* and any
discrepancy is recorded in a :class:`ValidationReport`.  Tolerances are
``atol + rtol * scale`` where *scale* is the magnitude of the quantities
involved, so that large and small models are treated transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .bsm import BalanceSheetMatrix
from .bsm import FINANCIAL as BSM_FINANCIAL
from .bsm import TANGIBLE
from .errors import AccountingError
from .tfm import TransactionFlowMatrix


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    location: str
    expected: float
    actual: float
    difference: float
    message: str = ""


@dataclass
class ValidationReport:
    atol: float = 1e-8
    rtol: float = 1e-9
    checks_run: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def record(
        self,
        check: str,
        location: str,
        expected: float,
        actual: float,
        scale: Optional[float] = None,
        message: str = "",
    ) -> bool:
        self.checks_run += 1
        if scale is None:
            scale = max(abs(expected), abs(actual))
        diff = actual - expected
        if not abs(diff) <= self.atol + self.rtol * scale:  # also catches NaN
            self.issues.append(
                ValidationIssue(check, location, expected, actual, diff, message)
            )
            return False
        return True

    def fail(self, check: str, location: str, message: str) -> None:
        self.checks_run += 1
        nan = float("nan")
        self.issues.append(ValidationIssue(check, location, nan, nan, nan, message))

    def merge(self, other: "ValidationReport", prefix: str = "") -> None:
        self.checks_run += other.checks_run
        for i in other.issues:
            loc = f"{prefix}{i.location}" if prefix else i.location
            self.issues.append(
                ValidationIssue(i.check, loc, i.expected, i.actual, i.difference, i.message)
            )

    def to_frame(self) -> pd.DataFrame:
        cols = ["check", "location", "expected", "actual", "difference", "message"]
        return pd.DataFrame([[getattr(i, c) for c in cols] for i in self.issues], columns=cols)

    def summary(self) -> str:
        status = "PASSED" if self.ok else f"FAILED ({len(self.issues)} issues)"
        return (
            f"Accounting validation {status}: {self.checks_run} checks run "
            f"(atol={self.atol:g}, rtol={self.rtol:g})."
        )

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AccountingError(self.summary() + "\n" + self.to_frame().head(10).to_string())


def check_tfm(tfm: TransactionFlowMatrix, atol: float = 1e-8, rtol: float = 1e-9) -> ValidationReport:
    rep = ValidationReport(atol, rtol)
    if not tfm.transactions:
        rep.fail("tfm.empty", "-", "The TFM has no transactions.")
        return rep
    frame = tfm.to_frame(include_totals=False)
    # every transaction has a counterpart
    for name, row in frame.iterrows():
        rep.record("tfm.row_sum", str(name), 0.0, float(row.sum()), float(row.abs().sum()))
    # each sector's budget constraint
    for sector in tfm.sectors:
        col = frame[sector]
        rep.record("tfm.column_sum", sector, 0.0, float(col.sum()), float(col.abs().sum()))
    # net lending equals net acquisition of financial assets
    nl = tfm.net_lending()
    fin = tfm.financial_flows()
    for sector in tfm.sectors:
        financing = float(fin[sector].sum()) if len(fin) else 0.0
        rep.record(
            "tfm.net_lending_equals_financial_acquisition",
            sector,
            float(nl[sector]),
            -financing,
            float(frame[sector].abs().sum()),
        )
    rep.record("tfm.aggregate_net_lending", "all", 0.0, float(nl.sum()), float(nl.abs().sum()))
    return rep


def check_bsm(bsm: BalanceSheetMatrix, atol: float = 1e-8, rtol: float = 1e-9) -> ValidationReport:
    rep = ValidationReport(atol, rtol)
    if not bsm.instruments:
        rep.fail("bsm.empty", "-", "The BSM has no instruments.")
        return rep
    fin = bsm.stocks(BSM_FINANCIAL)
    for name, row in fin.iterrows():
        rep.record(
            "bsm.financial_assets_equal_liabilities",
            str(name),
            0.0,
            float(row.sum()),
            float(row.abs().sum()),
        )
    assets, liabs = bsm.financial_assets(), bsm.financial_liabilities()
    rep.record(
        "bsm.aggregate_financial_assets_equal_liabilities",
        "all",
        float(liabs.sum()),
        float(assets.sum()),
    )
    tang = bsm.stocks(TANGIBLE)
    tang_total = float(tang.to_numpy().sum()) if len(tang) else 0.0
    rep.record(
        "bsm.aggregate_net_worth_equals_tangible_assets",
        "all",
        tang_total,
        float(bsm.net_worth().sum()),
        float(bsm.stocks().abs().to_numpy().sum()),
    )
    full = bsm.to_frame(include_balance=True, include_totals=False)
    for sector in bsm.sectors:
        rep.record(
            "bsm.column_sum", sector, 0.0, float(full[sector].sum()), float(full[sector].abs().sum())
        )
    return rep


def _aligned(frame: Optional[pd.DataFrame], index, columns) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(0.0, index=index, columns=columns)
    return frame.reindex(index=index, columns=columns).fillna(0.0)


def check_stock_flow(
    opening: BalanceSheetMatrix,
    closing: BalanceSheetMatrix,
    tfm: TransactionFlowMatrix,
    revaluations: Optional[pd.DataFrame] = None,
    nonfinancial_acquisitions: Optional[pd.Series] = None,
    atol: float = 1e-8,
    rtol: float = 1e-9,
) -> ValidationReport:
    """Closing stock = opening stock + transactions + valuation changes.

    * Financial instrument ``i`` held by sector ``s``:
      ``BSM_close = BSM_open - TFM_fin + reval`` (the TFM financial row has the
      opposite sign to the BSM entry: acquisition of an asset is negative in the
      TFM and positive in the BSM).
    * Tangible assets: ``dK_s = net acquisition_s + reval_s``.
    * Net worth: ``dNW_s = net lending_s + net acquisition of non-financial
      assets_s + total revaluation_s``.
    """
    rep = ValidationReport(atol, rtol)
    if opening.sectors != closing.sectors or opening.instruments != closing.instruments:
        rep.fail("stock_flow.structure", "-", "Opening and closing BSM differ in structure.")
        return rep
    if tfm.sectors != closing.sectors:
        rep.fail("stock_flow.structure", "-", "TFM and BSM sectors differ.")
        return rep
    sectors, instruments = closing.sectors, closing.instruments
    reval = _aligned(revaluations, instruments, sectors)
    nfa = (
        pd.Series(0.0, index=sectors)
        if nonfinancial_acquisitions is None
        else nonfinancial_acquisitions.reindex(sectors).fillna(0.0)
    )
    flows = tfm.financial_flows()
    for name in sorted(set(flows.index) - set(instruments)):
        rep.fail(
            "stock_flow.unmatched_financial_row", name, "TFM financial row has no BSM instrument."
        )
    fin_names = [i for i in instruments if closing.kind_of(i) == BSM_FINANCIAL]
    flows = flows.reindex(index=fin_names, columns=sectors).fillna(0.0)
    o, c = opening.stocks(), closing.stocks()
    for inst in fin_names:
        for s in sectors:
            expected = float(c.loc[inst, s])
            actual = float(o.loc[inst, s] - flows.loc[inst, s] + reval.loc[inst, s])
            rep.record(
                "stock_flow.financial_stock",
                f"{inst}/{s}",
                expected,
                actual,
                abs(float(o.loc[inst, s])) + abs(expected),
            )
        rep.record(
            "stock_flow.revaluation_nets_to_zero",
            inst,
            0.0,
            float(reval.loc[inst].sum()),
            float(reval.loc[inst].abs().sum()),
        )
    tang_names = [i for i in instruments if closing.kind_of(i) == TANGIBLE]
    for s in sectors:
        if tang_names:
            d_tang = float((c.loc[tang_names, s] - o.loc[tang_names, s]).sum())
            exp = float(nfa[s] + reval.loc[tang_names, s].sum())
            rep.record("stock_flow.tangible_stock", s, d_tang, exp, abs(d_tang) + abs(exp))
    nl = tfm.net_lending()
    d_nw = closing.net_worth() - opening.net_worth()
    for s in sectors:
        exp = float(nl[s] + nfa[s] + reval[s].sum())
        rep.record(
            "stock_flow.net_worth",
            s,
            float(d_nw[s]),
            exp,
            abs(float(d_nw[s])) + abs(exp) + float(abs(closing.net_worth()[s])),
        )
    return rep
