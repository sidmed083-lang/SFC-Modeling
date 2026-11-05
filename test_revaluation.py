import pytest

from sfc_modeling.library import revaluation_model
from sfc_modeling.simulation import simulate


def test_capital_gains_are_stock_flow_consistent():
    spec = revaluation_model()
    r = simulate(spec, 20)
    rep = r.validate()
    assert rep.ok, rep.to_frame().head()
    # price rises 10% at t=3 on 'Bq(-1)' units: households gain, government loses
    _, _, reval, _ = r.matrices(3)
    gain = r.data.loc[2, "Bq"] * 0.1
    assert reval.loc["Bonds", "Households"] == pytest.approx(gain)
    assert reval.loc["Bonds", "Government"] == pytest.approx(-gain)
    assert r.data.loc[3, "Bv"] == pytest.approx(r.data.loc[3, "Bq"] * 1.1)


def test_omitting_the_revaluation_is_detected():
    spec = revaluation_model()
    spec.accounting.revaluations = {}
    rep = simulate(spec, 20).validate()
    assert not rep.ok
    checks = set(rep.to_frame()["check"])
    assert "stock_flow.financial_stock" in checks and "stock_flow.net_worth" in checks
    locations = rep.to_frame()["location"]
    # only the periods with a price change (t=3: +10%, t=4: -9.1%, t=5: -9.1%, t=6: +11%) fail
    failing = {int(x.split(":")[0][2:]) for x in locations}
    assert failing == {3, 4, 5, 6}


def test_wrong_revaluation_sign_is_detected():
    spec = revaluation_model()
    spec.accounting.revaluations["Bonds"]["Government"] = "Bq(-1)*(pb - pb(-1))"  # not -gain
    assert "stock_flow.revaluation_nets_to_zero" in set(simulate(spec, 20).validate().to_frame()["check"])
