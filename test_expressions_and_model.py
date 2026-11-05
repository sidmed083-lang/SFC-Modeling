import pytest

from sfc_modeling.expressions import ExpressionError, parse_expression, validate_name
from sfc_modeling.model import Equation, ModelSpecification, ModelSpecificationError

KNOWN = {"x", "y", "a", "b"}


def test_parse_and_evaluate_with_lags():
    e = parse_expression("a*x + b*y(-1) + 2^2", KNOWN, {"x", "y"})
    assert set(e.names) == {"a", "x", "b", "y__lag1"}
    assert e({"a": 2.0, "x": 3.0, "b": 1.0, "y__lag1": 5.0}) == 15.0


def test_functions_and_negative_literals_are_not_lags():
    e = parse_expression("exp(-1) + Max(x, 2)", KNOWN, {"x"})
    assert abs(e({"x": 3.0}) - (2.718281828459045 ** -1 + 3.0)) < 1e-12


def test_constants_only():
    assert parse_expression("1.5e1 + 1", KNOWN)({}) == 16.0


def test_rejects_unknown_names_leads_and_bad_lags():
    with pytest.raises(ExpressionError, match="Unknown name"):
        parse_expression("zzz + 1", KNOWN)
    with pytest.raises(ExpressionError, match="Leads"):
        parse_expression("x(+1)", KNOWN, {"x"})
    with pytest.raises(ExpressionError, match="cannot be lagged"):
        parse_expression("a(-1)", KNOWN, {"x"})
    with pytest.raises(ExpressionError):
        parse_expression("", KNOWN)


def test_rejects_code_injection():
    for bad in [
        "__import__('os').system('echo hi')",
        "x.__class__",
        "x; y",
        "[1,2]",
        "x if y else a",
        "open('f')",
        "x == y",
        "lambda: 1",
    ]:
        with pytest.raises(ExpressionError):
            parse_expression(bad, KNOWN)


def test_steady_state_substitution():
    e = parse_expression("x(-1) + 1", KNOWN, {"x"}).steady_state()
    assert e.names == ["x"]


def test_names():
    for ok in ["Y", "alpha1", "Bh_s"]:
        validate_name(ok)
    for bad in ["1x", "a__b", "lambda", "exp", "x y", ""]:
        with pytest.raises(ExpressionError):
            validate_name(bad)


def test_equation_parsing():
    e = Equation.from_string("Y = C + G")
    assert (e.lhs, e.rhs) == ("Y", "C + G")
    with pytest.raises(ExpressionError):
        Equation.from_string("Y == C")
    with pytest.raises(ExpressionError):
        Equation.from_string("Y + 1 = C = 2")


def spec(eqs, **kw):
    return ModelSpecification("t", [Equation.from_string(e) for e in eqs], **kw)


def test_compile_reports_all_problems():
    s = spec(["x = y", "x = 2", "z = q"], parameters={"x": 1.0})
    with pytest.raises(ModelSpecificationError) as err:
        s.compile()
    msg = str(err.value)
    assert "more than one equation" in msg and "Unknown name" in msg and "more than one role" in msg


def test_block_structure_recursive_and_simultaneous():
    m = spec(["x = 1 + sqrt(y)", "y = 2*x + 1", "z = x + y", "w = z(-1) + 1"]).compile()
    sim = [b for b in m.blocks if b.simultaneous]
    assert len(sim) == 1 and set(sim[0].variables) == {"x", "y"}
    order = [b.variables[0] for b in m.blocks if not b.simultaneous]
    assert order == ["z", "w"] or order == ["w", "z"]
    # z must come after the simultaneous block
    assert m.blocks.index(sim[0]) < [b.variables for b in m.blocks].index(["z"])
    assert m.state_variables == ["z"]
    table = m.dependency_table()
    assert set(table["variable"]) == {"x", "y", "z", "w"}


def test_self_referencing_equation_is_simultaneous():
    m = spec(["x = x/2 + 1"]).compile()
    assert m.blocks[0].simultaneous


def test_json_round_trip():
    from sfc_modeling.library import pc_model

    s = pc_model()
    s2 = ModelSpecification.from_json(s.to_json())
    assert s2.to_dict() == s.to_dict()
    s2.compile()
