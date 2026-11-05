"""Model specification and compilation.

A :class:`ModelSpecification` is plain, JSON-serialisable data.  Compiling it
validates names, parses the equations, analyses dependencies and orders the
equations into *blocks*: recursive blocks (one equation, solved by direct
evaluation) and simultaneous blocks (circular dependencies, solved numerically).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union

import networkx as nx

from ..accounting.spec import AccountingSpec
from ..expressions import (
    CompiledExpression,
    ExpressionError,
    parse_expression,
    split_lag,
    validate_name,
)
from .equations import Equation

Number = Union[float, int]


class ModelSpecificationError(ValueError):
    """Raised when a model specification is inconsistent."""


@dataclass
class ModelSpecification:
    name: str
    equations: list[Equation]
    parameters: dict[str, float] = field(default_factory=dict)
    exogenous: dict[str, Union[Number, list[Number]]] = field(default_factory=dict)
    initial: dict[str, float] = field(default_factory=dict)
    consistency_checks: list[str] = field(default_factory=list)
    accounting: Optional[AccountingSpec] = None
    description: str = ""
    descriptions: dict[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None

    # ------------------------------------------------------------ helpers
    def exogenous_value(self, name: str, period: int) -> float:
        """Value of an exogenous input at ``period`` (lists are indexed by period)."""
        v = self.exogenous[name]
        if isinstance(v, (list, tuple)):
            return float(v[min(max(period, 0), len(v) - 1)])
        return float(v)

    def compile(self) -> "CompiledModel":
        return CompiledModel(self)

    # -------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "equations": [
                {"lhs": e.lhs, "rhs": e.rhs, "kind": e.kind, "description": e.description}
                for e in self.equations
            ],
            "parameters": dict(self.parameters),
            "exogenous": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in self.exogenous.items()},
            "initial": dict(self.initial),
            "consistency_checks": list(self.consistency_checks),
            "accounting": self.accounting.to_dict() if self.accounting else None,
            "descriptions": dict(self.descriptions),
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSpecification":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            equations=[
                Equation(e["lhs"], e["rhs"], e.get("kind", "behavioral"), e.get("description", ""))
                for e in d.get("equations", [])
            ],
            parameters=dict(d.get("parameters", {})),
            exogenous=dict(d.get("exogenous", {})),
            initial=dict(d.get("initial", {})),
            consistency_checks=list(d.get("consistency_checks", [])),
            accounting=AccountingSpec.from_dict(d["accounting"]) if d.get("accounting") else None,
            descriptions=dict(d.get("descriptions", {})),
            random_seed=d.get("random_seed"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ModelSpecification":
        return cls.from_dict(json.loads(text))


@dataclass
class Block:
    index: int
    variables: list[str]
    simultaneous: bool


class CompiledModel:
    """Validated, parsed and block-ordered model."""

    def __init__(self, spec: ModelSpecification):
        self.spec = spec
        errors: list[str] = []
        self.endogenous: list[str] = [e.lhs for e in spec.equations]
        self.parameters: dict[str, float] = dict(spec.parameters)
        self.exogenous: list[str] = list(spec.exogenous)

        if not spec.equations:
            errors.append("The model has no equations.")
        for group, names in (
            ("endogenous", self.endogenous),
            ("parameter", list(self.parameters)),
            ("exogenous", self.exogenous),
        ):
            for n in names:
                try:
                    validate_name(n)
                except ExpressionError as exc:
                    errors.append(f"{group}: {exc}")
        dup = sorted({n for n in self.endogenous if self.endogenous.count(n) > 1})
        if dup:
            errors.append(f"Variables defined by more than one equation: {dup}")
        clash = (
            set(self.endogenous) & set(self.parameters)
            | set(self.endogenous) & set(self.exogenous)
            | set(self.parameters) & set(self.exogenous)
        )
        if clash:
            errors.append(f"Names used in more than one role (endogenous/parameter/exogenous): {sorted(clash)}")
        for n, v in spec.exogenous.items():
            if isinstance(v, (list, tuple)) and not v:
                errors.append(f"Exogenous series '{n}' is empty.")
        unknown_init = sorted(set(spec.initial) - set(self.endogenous) - set(self.exogenous))
        if unknown_init:
            errors.append(f"Initial values given for unknown variables: {unknown_init}")

        known = set(self.endogenous) | set(self.parameters) | set(self.exogenous)
        lagged = set(self.endogenous) | set(self.exogenous)
        self.expressions: dict[str, CompiledExpression] = {}
        for eq in spec.equations:
            try:
                self.expressions[eq.lhs] = parse_expression(eq.rhs, known, lagged)
            except ExpressionError as exc:
                errors.append(f"Equation '{eq}': {exc}")
        self.checks: list[CompiledExpression] = []
        for text in spec.consistency_checks:
            try:
                self.checks.append(parse_expression(text, known, lagged))
            except ExpressionError as exc:
                errors.append(f"Consistency check '{text}': {exc}")
        if errors:
            raise ModelSpecificationError("\n".join(errors))

        self.max_lag = 0
        self.lag_symbols: dict[str, tuple[str, int]] = {}
        for ce in list(self.expressions.values()) + self.checks:
            for sym in ce.names:
                base, k = split_lag(sym)
                if k:
                    self.lag_symbols[sym] = (base, k)
                    self.max_lag = max(self.max_lag, k)
        self.state_variables = sorted({b for b, _ in self.lag_symbols.values() if b in set(self.endogenous)})
        self._build_blocks()
        self.steady_expressions = {v: e.steady_state() for v, e in self.expressions.items()}
        self.steady_checks = [c.steady_state() for c in self.checks]

    # ----------------------------------------------------- dependency analysis
    def _build_blocks(self) -> None:
        endo = set(self.endogenous)
        g = nx.DiGraph()
        g.add_nodes_from(self.endogenous)
        for var, ce in self.expressions.items():
            for name in ce.names:
                if name in endo:  # contemporaneous dependency
                    g.add_edge(name, var)
        self.dependency_graph = g
        cond = nx.condensation(g)
        order_idx = {v: i for i, v in enumerate(self.endogenous)}
        topo = nx.lexicographical_topological_sort(
            cond, key=lambda n: min(order_idx[v] for v in cond.nodes[n]["members"])
        )
        self.blocks: list[Block] = []
        for i, node in enumerate(topo):
            members = sorted(cond.nodes[node]["members"], key=order_idx.get)
            sim = len(members) > 1 or g.has_edge(members[0], members[0])
            self.blocks.append(Block(i, members, sim))

    def dependency_table(self):
        """Equation dependency analysis as a DataFrame."""
        import pandas as pd

        rows = []
        block_of = {v: b.index for b in self.blocks for v in b.variables}
        sim_of = {v: b.simultaneous for b in self.blocks for v in b.variables}
        for var, ce in self.expressions.items():
            contemporaneous, lagged = [], []
            for n in ce.names:
                base, k = split_lag(n)
                (lagged if k else contemporaneous).append(f"{base}({-k})" if k else base)
            rows.append(
                {
                    "variable": var,
                    "equation": str(ce.text),
                    "block": block_of[var],
                    "simultaneous": sim_of[var],
                    "depends_on": ", ".join(contemporaneous),
                    "lags": ", ".join(lagged),
                }
            )
        return pd.DataFrame(rows)
