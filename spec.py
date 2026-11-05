"""Accounting specification: TFM / BSM templates written as expressions.

Each cell of the transactions-flow matrix and balance-sheet matrix is an
expression in the model variables (``-C``, ``r(-1)*Bh(-1)``, ``Hh - Hh(-1)``).
At every simulated period the engine evaluates the templates against the
solution, builds a :class:`TransactionFlowMatrix` and a
:class:`BalanceSheetMatrix`, and validates them.  Consistency is therefore
*tested*, never assumed.

Sign conventions are those of :mod:`sfc_modeling.accounting.tfm` and
:mod:`sfc_modeling.accounting.bsm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from ..expressions import CompiledExpression, parse_expression, split_lag
from .bsm import INSTRUMENT_KINDS, BalanceSheetMatrix
from .classification import Classification
from .errors import AccountingError
from .tfm import ROW_KINDS, TransactionFlowMatrix


@dataclass
class TransactionSpec:
    name: str
    kind: str
    entries: dict[str, str] = field(default_factory=dict)


@dataclass
class InstrumentSpec:
    name: str
    kind: str
    entries: dict[str, str] = field(default_factory=dict)


@dataclass
class AccountingSpec:
    sectors: list[str]
    transactions: list[TransactionSpec] = field(default_factory=list)
    instruments: list[InstrumentSpec] = field(default_factory=list)
    revaluations: dict[str, dict[str, str]] = field(default_factory=dict)
    nonfinancial_acquisitions: dict[str, str] = field(default_factory=dict)
    classification: Optional[Classification] = None

    # ------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "sectors": list(self.sectors),
            "transactions": [
                {"name": t.name, "kind": t.kind, "entries": dict(t.entries)}
                for t in self.transactions
            ],
            "instruments": [
                {"name": i.name, "kind": i.kind, "entries": dict(i.entries)}
                for i in self.instruments
            ],
            "revaluations": {k: dict(v) for k, v in self.revaluations.items()},
            "nonfinancial_acquisitions": dict(self.nonfinancial_acquisitions),
            "classification": self.classification.to_dict() if self.classification else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AccountingSpec":
        def ents(x: Mapping[str, Any]) -> dict[str, str]:
            return {k: str(v) for k, v in x.items()}

        return cls(
            sectors=list(d["sectors"]),
            transactions=[
                TransactionSpec(t["name"], t["kind"], ents(t.get("entries", {})))
                for t in d.get("transactions", [])
            ],
            instruments=[
                InstrumentSpec(i["name"], i["kind"], ents(i.get("entries", {})))
                for i in d.get("instruments", [])
            ],
            revaluations={k: ents(v) for k, v in d.get("revaluations", {}).items()},
            nonfinancial_acquisitions=ents(d.get("nonfinancial_acquisitions", {})),
            classification=Classification.from_dict(d["classification"])
            if d.get("classification")
            else None,
        )

    # ------------------------------------------------- editable table views
    def transactions_frame(self) -> pd.DataFrame:
        rows = [
            {"transaction": t.name, "kind": t.kind, **{s: t.entries.get(s, "") for s in self.sectors}}
            for t in self.transactions
        ]
        return pd.DataFrame(rows, columns=["transaction", "kind", *self.sectors])

    def instruments_frame(self) -> pd.DataFrame:
        rows = [
            {"instrument": i.name, "kind": i.kind, **{s: i.entries.get(s, "") for s in self.sectors}}
            for i in self.instruments
        ]
        return pd.DataFrame(rows, columns=["instrument", "kind", *self.sectors])

    def with_frames(self, transactions: pd.DataFrame, instruments: pd.DataFrame) -> "AccountingSpec":
        def clean(v: Any) -> str:
            return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()

        tx = [
            TransactionSpec(
                clean(r["transaction"]),
                clean(r["kind"]),
                {s: clean(r[s]) for s in self.sectors if clean(r[s])},
            )
            for _, r in transactions.iterrows()
            if clean(r["transaction"])
        ]
        ins = [
            InstrumentSpec(
                clean(r["instrument"]),
                clean(r["kind"]),
                {s: clean(r[s]) for s in self.sectors if clean(r[s])},
            )
            for _, r in instruments.iterrows()
            if clean(r["instrument"])
        ]
        return AccountingSpec(
            list(self.sectors), tx, ins, dict(self.revaluations),
            dict(self.nonfinancial_acquisitions), self.classification,
        )

    # ------------------------------------------------------------ compile
    def compile(self, known: Iterable[str], lagged: Iterable[str]) -> "CompiledAccounting":
        return CompiledAccounting(self, set(known), set(lagged))


class CompiledAccounting:
    def __init__(self, spec: AccountingSpec, known: set[str], lagged: set[str]):
        self.spec = spec
        errors: list[str] = []
        if spec.classification is not None:
            try:
                spec.classification.require_sectors(spec.sectors)
                spec.classification.require_instruments(i.name for i in spec.instruments)
            except AccountingError as exc:
                errors.append(str(exc))
        for t in spec.transactions:
            if t.kind not in ROW_KINDS:
                errors.append(f"Transaction '{t.name}': invalid kind '{t.kind}'.")
        for i in spec.instruments:
            if i.kind not in INSTRUMENT_KINDS:
                errors.append(f"Instrument '{i.name}': invalid kind '{i.kind}'.")

        def comp(text: str, where: str) -> Optional[CompiledExpression]:
            try:
                return parse_expression(text, known, lagged)
            except ValueError as exc:
                errors.append(f"{where}: {exc}")
                return None

        def entries(item: Any, label: str) -> dict[str, CompiledExpression]:
            out = {}
            for s, text in item.entries.items():
                if s not in spec.sectors:
                    errors.append(f"{label} '{item.name}': unknown sector '{s}'.")
                    continue
                c = comp(text, f"{label} '{item.name}'/{s}")
                if c is not None:
                    out[s] = c
            return out

        self.tx = [(t, entries(t, "Transaction")) for t in spec.transactions]
        self.ins = [(i, entries(i, "Instrument")) for i in spec.instruments]
        inst_names = {i.name for i in spec.instruments}
        self.reval: dict[str, dict[str, CompiledExpression]] = {}
        for inst, row in spec.revaluations.items():
            if inst not in inst_names:
                errors.append(f"Revaluation for unknown instrument '{inst}'.")
                continue
            self.reval[inst] = {
                s: c for s, t in row.items() if (c := comp(t, f"Revaluation {inst}/{s}")) is not None
            }
        self.nfa = {
            s: c
            for s, t in spec.nonfinancial_acquisitions.items()
            if (c := comp(t, f"Non-financial acquisition {s}")) is not None
        }
        if errors:
            raise AccountingError("\n".join(errors))
        self.lag_symbols: dict[str, tuple[str, int]] = {}
        exprs = [c for _, e in self.tx for c in e.values()] + [c for _, e in self.ins for c in e.values()]
        exprs += [c for row in self.reval.values() for c in row.values()] + list(self.nfa.values())
        for c in exprs:
            for n in c.names:
                base, k = split_lag(n)
                if k:
                    self.lag_symbols[n] = (base, k)

    def build(self, ctx: Mapping[str, float]):
        """Evaluate the templates -> ``(tfm, bsm, revaluations, nfa)``."""
        sectors = self.spec.sectors
        tfm = TransactionFlowMatrix(sectors)
        for t, ents in self.tx:
            tfm.add_transaction(t.name, t.kind, {s: c(ctx) for s, c in ents.items()})
        bsm = BalanceSheetMatrix(sectors)
        for i, ents in self.ins:
            bsm.add_instrument(i.name, i.kind, {s: c(ctx) for s, c in ents.items()})
        reval = None
        if self.reval:
            reval = pd.DataFrame(0.0, index=[i.name for i in self.spec.instruments], columns=sectors)
            for inst, row in self.reval.items():
                for s, c in row.items():
                    reval.loc[inst, s] = c(ctx)
        nfa = None
        if self.nfa:
            nfa = pd.Series({s: c(ctx) for s, c in self.nfa.items()}, dtype=float)
        return tfm, bsm, reval, nfa
