"""Sector and instrument classification.

A single, explicit classification keeps sector and instrument names
consistent across the TFM, the BSM and the model equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .errors import AccountingError


class SectorKind(str, Enum):
    HOUSEHOLDS = "households"
    NON_FINANCIAL_CORPORATIONS = "non_financial_corporations"
    FINANCIAL_CORPORATIONS = "financial_corporations"
    GOVERNMENT = "government"
    CENTRAL_BANK = "central_bank"
    FOREIGN = "foreign"
    OTHER = "other"


class InstrumentKind(str, Enum):
    FINANCIAL = "financial"  # asset of one sector, liability of another
    TANGIBLE = "tangible"  # real assets (capital, inventories); no counterpart


@dataclass(frozen=True)
class Sector:
    name: str
    kind: SectorKind = SectorKind.OTHER
    description: str = ""


@dataclass(frozen=True)
class Instrument:
    name: str
    kind: InstrumentKind = InstrumentKind.FINANCIAL
    description: str = ""


@dataclass
class Classification:
    """Registry of the sectors and instruments of a model."""

    sectors: list[Sector] = field(default_factory=list)
    instruments: list[Instrument] = field(default_factory=list)

    def __post_init__(self) -> None:
        for label, items in (("sector", self.sectors), ("instrument", self.instruments)):
            names = [i.name for i in items]
            if len(set(names)) != len(names):
                raise AccountingError(f"Duplicate {label} names: {names}")

    @property
    def sector_names(self) -> list[str]:
        return [s.name for s in self.sectors]

    @property
    def instrument_names(self) -> list[str]:
        return [i.name for i in self.instruments]

    def require_sectors(self, names: Iterable[str]) -> None:
        unknown = sorted(set(names) - set(self.sector_names))
        if unknown:
            raise AccountingError(f"Unknown sectors: {unknown}")

    def require_instruments(self, names: Iterable[str]) -> None:
        unknown = sorted(set(names) - set(self.instrument_names))
        if unknown:
            raise AccountingError(f"Unknown instruments: {unknown}")

    def to_dict(self) -> dict:
        return {
            "sectors": [
                {"name": s.name, "kind": s.kind.value, "description": s.description}
                for s in self.sectors
            ],
            "instruments": [
                {"name": i.name, "kind": i.kind.value, "description": i.description}
                for i in self.instruments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Classification":
        return cls(
            [
                Sector(d["name"], SectorKind(d.get("kind", "other")), d.get("description", ""))
                for d in data.get("sectors", [])
            ],
            [
                Instrument(
                    d["name"],
                    InstrumentKind(d.get("kind", "financial")),
                    d.get("description", ""),
                )
                for d in data.get("instruments", [])
            ],
        )


def standard_classification() -> Classification:
    """The six sectors and a typical instrument set suggested by SFC practice."""
    sectors = [
        Sector("Households", SectorKind.HOUSEHOLDS),
        Sector("Firms", SectorKind.NON_FINANCIAL_CORPORATIONS),
        Sector("Banks", SectorKind.FINANCIAL_CORPORATIONS),
        Sector("Government", SectorKind.GOVERNMENT),
        Sector("CentralBank", SectorKind.CENTRAL_BANK),
        Sector("Foreign", SectorKind.FOREIGN),
    ]
    instruments = [
        Instrument("HighPoweredMoney", description="Cash / reserves"),
        Instrument("Deposits"),
        Instrument("Loans"),
        Instrument("Bills"),
        Instrument("Bonds"),
        Instrument("Equities"),
        Instrument("Capital", InstrumentKind.TANGIBLE, "Fixed capital"),
    ]
    return Classification(sectors, instruments)
