"""Transactions-flow matrix (TFM).

Sign convention (Godley & Lavoie): a **positive** entry is a receipt / source
of funds, a **negative** entry is a payment / use of funds.  Rows are
transactions, columns are sectors.  Row kinds:

* ``current``   - income and outlay transactions; column sum = saving.
* ``capital``   - capital transfers; current + capital sum = net lending.
* ``financial`` - change in a financial instrument.  Acquisition of an asset
  is negative (use of funds), issuance of a liability is positive (source).

Every row must sum to zero (each transaction has a counterpart) and every
column must sum to zero (each sector's budget constraint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .errors import AccountingError

CURRENT = "current"
CAPITAL = "capital"
FINANCIAL = "financial"
ROW_KINDS = (CURRENT, CAPITAL, FINANCIAL)
FLOW_KINDS = (CURRENT, CAPITAL)
TOTAL = "Σ"


@dataclass
class TFMRow:
    name: str
    kind: str
    entries: dict[str, float]


class TransactionFlowMatrix:
    def __init__(self, sectors: Sequence[str]):
        sectors = list(sectors)
        if not sectors:
            raise AccountingError("A TFM needs at least one sector.")
        if len(set(sectors)) != len(sectors):
            raise AccountingError(f"Duplicate sectors: {sectors}")
        self.sectors: list[str] = sectors
        self._rows: dict[str, TFMRow] = {}

    # ------------------------------------------------------------ building
    def add_transaction(
        self, name: str, kind: str, entries: Mapping[str, float]
    ) -> None:
        if name in self._rows:
            raise AccountingError(f"Duplicate transaction '{name}'.")
        if kind not in ROW_KINDS:
            raise AccountingError(f"Invalid kind '{kind}' for '{name}'; use {ROW_KINDS}.")
        clean = {s: 0.0 for s in self.sectors}
        for sector, value in entries.items():
            if sector not in clean:
                raise AccountingError(f"Unknown sector '{sector}' in '{name}'.")
            value = float(value)
            if not math.isfinite(value):
                raise AccountingError(f"Non-finite entry for '{name}'/'{sector}'.")
            clean[sector] = value
        self._rows[name] = TFMRow(name, kind, clean)

    @property
    def transactions(self) -> list[str]:
        return list(self._rows)

    def kind_of(self, name: str) -> str:
        return self._rows[name].kind

    def rows_of_kind(self, *kinds: str) -> list[TFMRow]:
        return [r for r in self._rows.values() if r.kind in kinds]

    # ------------------------------------------------------------- frames
    def _frame(self, rows: list[TFMRow]) -> pd.DataFrame:
        return pd.DataFrame(
            [[r.entries[s] for s in self.sectors] for r in rows],
            index=[r.name for r in rows],
            columns=self.sectors,
            dtype=float,
        )

    def to_frame(self, include_totals: bool = True) -> pd.DataFrame:
        frame = self._frame(list(self._rows.values()))
        if include_totals:
            frame[TOTAL] = frame.sum(axis=1)
            frame.loc[TOTAL] = frame.sum(axis=0)
        return frame

    def kinds(self) -> pd.Series:
        return pd.Series({r.name: r.kind for r in self._rows.values()}, dtype=object)

    def financial_flows(self) -> pd.DataFrame:
        return self._frame(self.rows_of_kind(FINANCIAL))

    def _sum(self, kinds: tuple[str, ...]) -> pd.Series:
        total = pd.Series(0.0, index=self.sectors)
        for row in self.rows_of_kind(*kinds):
            total = total + pd.Series(row.entries)
        return total

    def saving(self) -> pd.Series:
        """Sum of current-account rows by sector."""
        return self._sum((CURRENT,))

    def net_lending(self) -> pd.Series:
        """Sum of current and capital rows by sector (net lending / borrowing)."""
        return self._sum(FLOW_KINDS)

    # --------------------------------------------------------- persistence
    @classmethod
    def from_frame(cls, frame: pd.DataFrame, kinds: Mapping[str, str]) -> "TransactionFlowMatrix":
        data = frame.drop(columns=[TOTAL], errors="ignore").drop(index=[TOTAL], errors="ignore")
        tfm = cls(list(data.columns))
        for name, row in data.iterrows():
            tfm.add_transaction(str(name), kinds[str(name)], row.to_dict())
        return tfm

    def validate(self, atol: float = 1e-8, rtol: float = 1e-9):
        from .validation import check_tfm

        return check_tfm(self, atol=atol, rtol=rtol)
