"""Model parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..expressions import validate_name


@dataclass
class Parameter:
    name: str
    value: float
    description: str = ""

    def __post_init__(self) -> None:
        validate_name(self.name)
        self.value = float(self.value)
        if not math.isfinite(self.value):
            raise ValueError(f"Parameter '{self.name}' must be finite.")
