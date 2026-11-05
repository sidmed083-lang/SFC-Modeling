import io
import zipfile

import pandas as pd
import pytest

from sfc_modeling.library import pc_model, sim_model
from sfc_modeling.reporting import markdown_report, result_bundle
from sfc_modeling.simulation import (
    Scenario,
    ScenarioError,
    Shock,
    compare,
    final_value,
    monte_carlo,
    one_at_a_time,
    simulate,
    summarize,
)


def test_permanent_policy_shock_reaches_new_steady_state():
    spec = sim_model()
    base = simulate(spec, 300)
    shock = simulate(spec, 300, Scenario("G up", [Shock("G", 1, value=25.0)]))
    assert base.data["Y"].iloc[-1] == pytest.approx(100.0, rel=1e-8)
    assert shock.data["Y"].iloc[-1] == pytest.approx(125.0, rel=1e-8)  # Y* = G / theta
    assert shock.data["Hh"].iloc[-1] > base.data["Hh"].iloc[-1]


def test_parameter_shock_and_delta_and_pct():
    spec = sim_model()
    r = simulate(spec, 300, Scenario("tax", [Shock("theta", 1, value=0.25)]))
    assert r.data["Y"].iloc[-1] == pytest.approx(80.0, rel=1e-8)
    r2 = simulate(spec, 300, Scenario("g", [Shock("G", 1, delta=5.0)]))
    assert r2.data["Y"].iloc[-1] == pytest.approx(125.0, rel=1e-8)
    r3 = simulate(spec, 300, Scenario("g", [Shock("G", 1, pct=0.5)]))
    assert r3.data["Y"].iloc[-1] == pytest.approx(150.0, rel=1e-8)


def test_temporary_shock_window():
    r = simulate(sim_model(), 20, Scenario("tmp", [Shock("G", 5, value=30.0, end=8)]))
    assert list(r.data["G"].loc[[4, 5, 8, 9]]) == [20.0, 30.0, 30.0, 20.0]


def test_scenario_errors():
    with pytest.raises(ScenarioError):
        Shock("G", 1)
    with pytest.raises(ScenarioError):
        Shock("G", 1, value=1.0, delta=1.0)
    with pytest.raises(ScenarioError):
        Shock("G", 0, value=1.0)
    with pytest.raises(ScenarioError):
        simulate(sim_model(), 5, Scenario("x", [Shock("nope", 1, value=1.0)]))
    with pytest.raises(ScenarioError):
        simulate(sim_model(), 0)


def test_accounting_validates_under_scenarios_for_both_models():
    for spec, shocks in (
        (sim_model(), [Shock("G", 3, value=35.0), Shock("theta", 6, value=0.3)]),
        (pc_model(), [Shock("G", 3, value=30.0), Shock("r", 6, value=0.05), Shock("lambda0", 8, delta=-0.05)]),
    ):
        r = simulate(spec, 60, Scenario("mix", shocks))
        rep = r.validate()
        assert rep.ok, rep.to_frame().head()
        assert rep.checks_run > 500
        assert r.consistency_residuals().abs().max().max() < 1e-8
        assert r.log.all_converged


def test_validation_detects_broken_accounting_templates():
    spec = sim_model()
    spec.accounting.transactions[3].entries["Government"] = "T*1.01"  # taxes: no counterpart
    rep = simulate(spec, 5).validate()
    assert not rep.ok and "tfm.row_sum" in set(rep.to_frame()["check"])

    spec = sim_model()
    spec.accounting.instruments[0].entries["Government"] = "-Hs*0.99"
    rep = simulate(spec, 5).validate()
    assert "bsm.financial_assets_equal_liabilities" in set(rep.to_frame()["check"])

    spec = sim_model()  # flow does not match the change in stock
    spec.accounting.transactions[4].entries["Households"] = "-(Hh - Hh(-1))*0.5"
    assert not simulate(spec, 5).validate().ok


def test_validation_detects_non_stock_flow_consistent_equations():
    spec = sim_model()
    spec.equations[5] = type(spec.equations[5]).from_string("Hh = Hh(-1) + 0.9*(YD - C)")
    rep = simulate(spec, 5).validate()
    assert not rep.ok
    checks = set(rep.to_frame()["check"])
    assert {"tfm.column_sum", "stock_flow.net_worth", "bsm.financial_assets_equal_liabilities"} <= checks
    # the redundant identity Hs = Hh also breaks
    assert simulate(spec, 5).consistency_residuals().abs().max().max() > 1e-3


def test_model_without_accounting_reports_missing():
    spec = sim_model()
    spec.accounting = None
    assert not simulate(spec, 2).validate().ok


def test_one_at_a_time_sensitivity():
    df = one_at_a_time(sim_model(), "theta", [0.1, 0.2, 0.25], final_value("Y"), 300)
    for got, want in zip(df["metric"], [200.0, 100.0, 80.0]):
        assert got == pytest.approx(want, rel=1e-7)


def test_monte_carlo_reproducible_with_seed():
    spec = sim_model()
    dist = {"alpha1": ("uniform", 0.5, 0.7), "G": ("normal", 20.0, 2.0)}
    a = monte_carlo(spec, dist, final_value("Y"), 60, n=15, seed=7)
    b = monte_carlo(spec, dist, final_value("Y"), 60, n=15, seed=7)
    c = monte_carlo(spec, dist, final_value("Y"), 60, n=15, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["metric"].equals(c["metric"])
    assert a.attrs["seed"] == 7 and a["converged"].all()
    s = summarize(a)
    assert s["q05"] <= s["q50"] <= s["q95"] and s["n_failed"] == 0
    # default seed comes from the specification
    d = monte_carlo(spec, dist, final_value("Y"), 20, n=3)
    assert d.attrs["seed"] == 12345
    with pytest.raises(ValueError):
        monte_carlo(spec, {"G": ("cauchy", 0, 1)}, final_value("Y"), 5, n=2)


def test_simulation_is_deterministic():
    a = simulate(pc_model(), 30).data
    b = simulate(pc_model(), 30).data
    pd.testing.assert_frame_equal(a, b)


def test_compare_scenarios():
    spec = sim_model()
    res = {"Baseline": simulate(spec, 20), "G up": simulate(spec, 20, Scenario("G up", [Shock("G", 1, delta=5.0)]))}
    wide = compare(res, ["Y", "C"])
    assert wide.shape == (21, 4)
    diff = compare(res, ["Y"], reference="Baseline")
    assert list(diff.columns.get_level_values(0)) == ["G up"]
    assert (diff[("G up", "Y")].iloc[1:] > 0).all()


def test_matrices_period_bounds():
    r = simulate(sim_model(), 3)
    with pytest.raises(IndexError):
        r.matrices(4)
    tfm, bsm, _, _ = r.matrices(3)
    assert tfm.validate().ok and bsm.validate().ok


def test_export_bundle_and_report():
    r = simulate(pc_model(), 12)
    z = zipfile.ZipFile(io.BytesIO(result_bundle(r)))
    names = set(z.namelist())
    assert {"data.csv", "model_spec.json", "report.md", "tfm_t12.csv", "bsm_t12.csv",
            "validation_issues.csv", "scenario.json"} <= names
    data = pd.read_csv(io.BytesIO(z.read("data.csv")), index_col=0)
    pd.testing.assert_frame_equal(data, r.data, check_names=False)
    tfm = pd.read_csv(io.BytesIO(z.read("tfm_t12.csv")), index_col=0)
    assert abs(tfm.loc["Σ"]).max() < 1e-8
    text = markdown_report(r)
    assert "PASSED" in text and "Redundant-identity" in text
