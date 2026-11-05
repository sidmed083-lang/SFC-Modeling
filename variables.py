"""Variable and parameter metadata."""

from __future__ import annotations

from dataclasses import dataclass

from ..expressions import validate_name

ENDOGENOUS = "endogenous"
EXOGENOUS = "exogenous"


@dataclass
class Variable:
    name: str
    kind: str = ENDOGENOUS
    description: str = ""
    initial: float = 0.0

    def __post_init__(self) -> None:
        validate_name(self.name)
        if self.kind not in (ENDOGENOUS, EXOGENOUS):
            raise ValueError(f"Invalid variable kind '{self.kind}'.")
