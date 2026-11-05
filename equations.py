"""Model equations of the form ``variable = expression``."""

from __future__ import annotations

from dataclasses import dataclass

from ..expressions import ExpressionError, validate_name

BEHAVIORAL = "behavioral"
IDENTITY = "identity"
KINDS = (BEHAVIORAL, IDENTITY)


@dataclass(frozen=True)
class Equation:
    """``lhs = rhs`` where ``lhs`` is a single endogenous variable."""

    lhs: str
    rhs: str
    kind: str = BEHAVIORAL
    description: str = ""

    def __post_init__(self) -> None:
        validate_name(self.lhs)
        if self.kind not in KINDS:
            raise ExpressionError(f"Invalid equation kind '{self.kind}'; use {KINDS}.")
        if not self.rhs.strip():
            raise ExpressionError(f"Equation for '{self.lhs}' has an empty right-hand side.")

    @classmethod
    def from_string(cls, text: str, kind: str = BEHAVIORAL, description: str = "") -> "Equation":
        if text.count("=") != 1 or any(op in text for op in ("==", "<=", ">=", "!=")):
            raise ExpressionError(f"An equation needs exactly one '=': '{text}'.")
        lhs, rhs = text.split("=")
        return cls(lhs.strip(), rhs.strip(), kind, description)

    def __str__(self) -> str:
        return f"{self.lhs} = {self.rhs}"
