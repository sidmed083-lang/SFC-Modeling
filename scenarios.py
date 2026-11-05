"""Scenarios: policy shocks and parameter changes applied to a baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ..model.specification import CompiledModel


class ScenarioError(ValueError):
    """Raised when a scenario is invalid."""


@dataclass
class Shock:
    """Change to a parameter or exogenous variable.

    Exactly one of ``value`` (new level), ``delta`` (additive change) or
    ``pct`` (relative change, 0.1 = +10 %) must be given.  The shock applies
    from period ``start`` to ``end`` inclusive (``None`` = permanent).
    """

    target: str
    start: int = 1
    value: Optional[float] = None
    delta: Optional[float] = None
    pct: Optional[float] = None
    end: Optional[int] = None

    def __post_init__(self) -> None:
        given = [x for x in (self.value, self.delta, self.pct) if x is not None]
        if len(given) != 1:
            raise ScenarioError(f"Shock on '{self.target}': give exactly one of value/delta/pct.")
        if self.start < 1:
            raise ScenarioError("Shocks must start at period 1 or later.")
        if self.end is not None and self.end < self.start:
            raise ScenarioError("Shock end precedes its start.")

    def apply(self, base: float) -> float:
        if self.value is not None:
            return float(self.value)
        if self.delta is not None:
            return base + float(self.delta)
        return base * (1.0 + float(self.pct))

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Scenario:
    name: str
    shocks: list[Shock] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "shocks": [s.to_dict() for s in self.shocks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scenario":
        return cls(d["name"], [Shock(**s) for s in d.get("shocks", [])], d.get("description", ""))


def baseline() -> Scenario:
    return Scenario("Baseline", [], "No shocks.")


def build_input_paths(
    model: CompiledModel, periods: int, scenario: Optional[Scenario] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parameter and exogenous paths for periods ``0..periods`` under ``scenario``."""
    if periods < 1:
        raise ScenarioError("Simulate at least one period.")
    idx = pd.RangeIndex(0, periods + 1, name="period")
    params = pd.DataFrame({k: float(v) for k, v in model.parameters.items()}, index=idx, dtype=float)
    exog = pd.DataFrame(
        {n: [model.spec.exogenous_value(n, t) for t in idx] for n in model.exogenous},
        index=idx, dtype=float,
    )
    for shock in (scenario.shocks if scenario else []):
        if shock.target in params.columns:
            frame = params
        elif shock.target in exog.columns:
            frame = exog
        else:
            raise ScenarioError(
                f"Shock target '{shock.target}' is neither a parameter nor an exogenous variable."
            )
        last = periods if shock.end is None else min(shock.end, periods)
        for t in range(shock.start, last + 1):
            frame.loc[t, shock.target] = shock.apply(float(frame.loc[t, shock.target]))
    return params, exog
