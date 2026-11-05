"""Safe parsing and compilation of algebraic expressions.

Expressions are written in plain algebra, e.g. ``alpha1*YD + alpha2*H(-1)``.

* ``X(-k)`` denotes the value of variable ``X`` ``k`` periods ago.
* ``^`` is accepted for powers.
* Allowed functions: exp, log, sqrt, Abs, Max, Min, tanh.

Parsing goes through SymPy's ``parse_expr``, which evaluates Python code.  To
reduce that risk, the input is first restricted to a whitelist of characters,
every identifier must be a known name or an allowed function, and attribute
access is rejected.  This is defence in depth, *not* a sandbox: do not expose
the equation editor to untrusted users on the public internet.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

ALLOWED_FUNCTIONS: dict[str, Any] = {
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "Max": sp.Max,
    "Min": sp.Min,
    "tanh": sp.tanh,
}

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_CHARS_RE = re.compile(r"[A-Za-z0-9_+\-*/^().,\s]*")
_IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*")
_ATTR_RE = re.compile(r"[\w)]\s*\.\s*[A-Za-z_]")
_LAG_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(\s*([+-])\s*(\d+)\s*\)")
LAG_TAG = "__lag"


class ExpressionError(ValueError):
    """Raised when an expression cannot be parsed or evaluated."""


def validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ExpressionError(
            f"Invalid name {name!r}: use ASCII letters, digits and single underscores, "
            "starting with a letter."
        )
    if "__" in name:
        raise ExpressionError(f"Invalid name {name!r}: double underscores are reserved.")
    if keyword.iskeyword(name) or name in ALLOWED_FUNCTIONS:
        raise ExpressionError(f"Invalid name {name!r}: reserved word.")


def lag_name(name: str, k: int) -> str:
    return f"{name}{LAG_TAG}{k}"


def split_lag(symbol_name: str) -> tuple[str, int]:
    """Return ``(base, k)``; ``k == 0`` for contemporaneous symbols."""
    if LAG_TAG in symbol_name:
        base, k = symbol_name.split(LAG_TAG)
        return base, int(k)
    return symbol_name, 0


@dataclass
class CompiledExpression:
    text: str
    expr: sp.Expr
    symbols: tuple[sp.Symbol, ...]
    _fn: Optional[Callable[..., Any]]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.symbols]

    def __call__(self, ctx: Mapping[str, float]) -> float:
        if self._fn is None:
            return float(self.expr)
        try:
            args = [ctx[s.name] for s in self.symbols]
        except KeyError as exc:
            raise ExpressionError(f"Missing value for {exc.args[0]!r} in '{self.text}'.") from exc
        return float(self._fn(*args))

    def steady_state(self) -> "CompiledExpression":
        """Same expression with every lagged symbol replaced by its current value."""
        sub = {}
        for s in self.symbols:
            base, k = split_lag(s.name)
            if k:
                sub[s] = sp.Symbol(base)
        return _build(self.expr.subs(sub), self.text) if sub else self


def _build(expr: sp.Expr, text: str) -> CompiledExpression:
    symbols = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
    fn = sp.lambdify(symbols, expr, modules="numpy") if symbols else None
    return CompiledExpression(text, expr, symbols, fn)


def parse_expression(
    text: str,
    known: Iterable[str],
    lagged: Iterable[str] = (),
) -> CompiledExpression:
    """Parse ``text``.

    ``known``  - names allowed contemporaneously (variables, parameters).
    ``lagged`` - names that may carry a lag ``X(-k)`` (variables only).
    """
    known, lagged = set(known), set(lagged)
    s = str(text).strip()
    if not s:
        raise ExpressionError("Empty expression.")
    if not _CHARS_RE.fullmatch(s):
        bad = sorted(set(re.sub(_CHARS_RE.pattern.rstrip("*"), "", s)))
        raise ExpressionError(f"Illegal characters in '{text}': {bad}")
    if _ATTR_RE.search(s):
        raise ExpressionError(f"Attribute access is not allowed in '{text}'.")
    s = s.replace("^", "**")
    lag_symbols: set[str] = set()

    def _sub(m: re.Match) -> str:
        name, sign, k = m.group(1), m.group(2), int(m.group(3))
        if name in ALLOWED_FUNCTIONS:
            return m.group(0)
        if sign == "+":
            raise ExpressionError(f"Leads such as {name}(+{k}) are not supported.")
        if k == 0:
            raise ExpressionError(f"Use {name} instead of {name}(-0).")
        if name not in lagged:
            raise ExpressionError(
                f"'{name}' cannot be lagged in '{text}' (only endogenous/exogenous variables can)."
            )
        sym = lag_name(name, k)
        lag_symbols.add(sym)
        return sym

    s = _LAG_RE.sub(_sub, s)
    for ident in _IDENT_RE.findall(s):
        if ident in ALLOWED_FUNCTIONS or ident in known or ident in lag_symbols:
            continue
        raise ExpressionError(f"Unknown name '{ident}' in '{text}'.")
    local = {n: sp.Symbol(n) for n in known | lag_symbols}
    local.update(ALLOWED_FUNCTIONS)
    try:
        expr = parse_expr(s, local_dict=local, evaluate=True)
    except Exception as exc:  # sympy raises many exception types
        raise ExpressionError(f"Cannot parse '{text}': {exc}") from exc
    if not isinstance(expr, sp.Basic):
        raise ExpressionError(f"'{text}' is not an algebraic expression.")
    return _build(expr, str(text).strip())
