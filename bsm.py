"""Balance-sheet matrix (BSM).

Sign convention: **positive** = asset, **negative** = liability.  Rows are
instruments, columns are sectors.  Financial instruments must sum to zero
across sectors (financial assets = liabilities).  Tangible instruments
(capital, inventories) have no counterpart.  The balancing item is net worth;
following Godley & Lavoie it is displayed with the opposite sign so that every
column sums to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .errors import AccountingError

FINANCIAL = "financial"
TANGIBLE = "tangible"
INSTRUMENT_KINDS = (FINANCIAL, TANGIBLE)
TOTAL = "Σ"
BALANCE = "Net worth (balancing item)"


@dataclass
class BSMRow:
    name: str
    kind: str
    entries: dict[str, float]


class BalanceSheetMatrix:
    def __init__(self, sectors: Sequence[str]):
        sectors = list(sectors)
        if not sectors:
            raise AccountingError("A BSM needs at least one sector.")
        if len(set(sectors)) != len(sectors):
            raise AccountingError(f"Duplicate sectors: {sectors}")
        self.sectors: list[str] = sectors
        self._rows: dict[str, BSMRow] = {}

    def add_instrument(self, name: str, kind: str, entries: Mapping[str, float]) -> None:
        if name in self._rows:
            raise AccountingError(f"Duplicate instrument '{name}'.")
        if kind not in INSTRUMENT_KINDS:
            raise AccountingError(f"Invalid kind '{kind}' for '{name}'.")
        clean = {s: 0.0 for s in self.sectors}
        for sector, value in entries.items():
            if sector not in clean:
                raise AccountingError(f"Unknown sector '{sector}' in '{name}'.")
            value = float(value)
            if not math.isfinite(value):
                raise AccountingError(f"Non-finite entry for '{name}'/'{sector}'.")
            clean[sector] = value
        self._rows[name] = BSMRow(name, kind, clean)

    @property
    def instruments(self) -> list[str]:
        return list(self._rows)

    def kind_of(self, name: str) -> str:
        return self._rows[name].kind

    def _frame(self, kind: str | None = None) -> pd.DataFrame:
        rows = [r for r in self._rows.values() if kind is None or r.kind == kind]
        return pd.DataFrame(
            [[r.entries[s] for s in self.sectors] for r in rows],
            index=[r.name for r in rows],
            columns=self.sectors,
            dtype=float,
        )

    def stocks(self, kind: str | None = None) -> pd.DataFrame:
        """Raw instrument x sector matrix (no balancing row)."""
        return self._frame(kind)

    def net_worth(self) -> pd.Series:
        """Assets minus liabilities, by sector."""
        return self._frame().sum(axis=0).reindex(self.sectors).fillna(0.0)

    def financial_assets(self) -> pd.Series:
        f = self._frame(FINANCIAL)
        return f.clip(lower=0).sum(axis=0).reindex(self.sectors).fillna(0.0)

    def financial_liabilities(self) -> pd.Series:
        f = self._frame(FINANCIAL)
        return (-f.clip(upper=0)).sum(axis=0).reindex(self.sectors).fillna(0.0)

    def net_financial_wealth(self) -> pd.Series:
        return self.financial_assets() - self.financial_liabilities()

    def to_frame(self, include_balance: bool = True, include_totals: bool = True) -> pd.DataFrame:
        frame = self._frame()
        if include_balance:
            frame.loc[BALANCE] = -self.net_worth()
        if include_totals:
            frame[TOTAL] = frame.sum(axis=1)
        return frame

    def kinds(self) -> pd.Series:
        return pd.Series({r.name: r.kind for r in self._rows.values()}, dtype=object)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, kinds: Mapping[str, str]) -> "BalanceSheetMatrix":
        data = frame.drop(columns=[TOTAL], errors="ignore").drop(index=[BALANCE], errors="ignore")
        bsm = cls(list(data.columns))
        for name, row in data.iterrows():
            bsm.add_instrument(str(name), kinds[str(name)], row.to_dict())
        return bsm

    def validate(self, atol: float = 1e-8, rtol: float = 1e-9):
        from .validation import check_bsm

        return check_bsm(self, atol=atol, rtol=rtol)
