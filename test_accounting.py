import pandas as pd
import pytest

from sfc_modeling.accounting import (
    AccountingError,
    BalanceSheetMatrix,
    TransactionFlowMatrix,
    check_stock_flow,
    standard_classification,
)


def make_tfm(tax_gov=5.0):
    t = TransactionFlowMatrix(["H", "G"])
    t.add_transaction("Taxes", "current", {"H": -5.0, "G": tax_gov})
    t.add_transaction("Spending", "current", {"H": 2.0, "G": -2.0})
    # H: net lending = -3 -> must issue 3 of liabilities (+3); G: NL = +3 -> acquires (-3)
    t.add_transaction("Money", "financial", {"H": 3.0, "G": -3.0})
    return t


def test_tfm_construction_and_sums():
    t = make_tfm()
    f = t.to_frame()
    assert f.loc["Taxes", "Σ"] == 0.0
    assert f.loc["Σ", "H"] == 0.0 and f.loc["Σ", "G"] == 0.0
    assert t.saving()["H"] == -3.0
    assert t.net_lending()["G"] == 3.0
    assert t.validate().ok


def test_tfm_detects_missing_counterpart():
    rep = make_tfm(tax_gov=4.0).validate()
    assert not rep.ok
    assert "tfm.row_sum" in set(rep.to_frame()["check"])
    with pytest.raises(AccountingError):
        rep.raise_if_failed()


def test_tfm_input_validation():
    t = TransactionFlowMatrix(["H"])
    with pytest.raises(AccountingError):
        t.add_transaction("x", "bogus", {})
    with pytest.raises(AccountingError):
        t.add_transaction("x", "current", {"Z": 1.0})
    with pytest.raises(AccountingError):
        t.add_transaction("x", "current", {"H": float("nan")})
    t.add_transaction("x", "current", {"H": 0.0})
    with pytest.raises(AccountingError):
        t.add_transaction("x", "current", {})
    with pytest.raises(AccountingError):
        TransactionFlowMatrix(["A", "A"])


def test_tfm_financing_mismatch_detected():
    t = TransactionFlowMatrix(["H", "G"])
    t.add_transaction("Taxes", "current", {"H": -5.0, "G": 5.0})
    t.add_transaction("Money", "financial", {"H": 1.0, "G": -1.0})  # should be H:+5, G:-5
    rep = t.validate()
    assert "tfm.net_lending_equals_financial_acquisition" in set(rep.to_frame()["check"])


def test_empty_tfm_reports_failure():
    assert not TransactionFlowMatrix(["H"]).validate().ok


def make_bsm(h_money=10.0, g_money=-10.0, capital=0.0):
    b = BalanceSheetMatrix(["H", "G"])
    b.add_instrument("Money", "financial", {"H": h_money, "G": g_money})
    if capital:
        b.add_instrument("Capital", "tangible", {"H": capital})
    return b


def test_bsm_balances():
    b = make_bsm()
    assert b.validate().ok
    assert b.net_worth()["H"] == 10.0 and b.net_worth()["G"] == -10.0
    f = b.to_frame()
    assert f.loc["Net worth (balancing item)", "H"] == -10.0


def test_bsm_assets_must_equal_liabilities():
    rep = make_bsm(h_money=10.0, g_money=-9.0).validate()
    assert "bsm.financial_assets_equal_liabilities" in set(rep.to_frame()["check"])


def test_bsm_tangible_assets_allowed_and_aggregate_net_worth():
    b = make_bsm(capital=50.0)
    assert b.validate().ok
    assert b.net_worth().sum() == 50.0


def test_stock_flow_reconciliation_pass_and_fail():
    t = TransactionFlowMatrix(["H", "G"])
    t.add_transaction("Taxes", "current", {"H": -5.0, "G": 5.0})
    t.add_transaction("Money", "financial", {"H": 5.0, "G": -5.0})
    # H issues 5 of "money" liability (-5 in BSM), G acquires 5 (+5)
    opening = BalanceSheetMatrix(["H", "G"])
    opening.add_instrument("Money", "financial", {"H": 0.0, "G": 0.0})
    closing = BalanceSheetMatrix(["H", "G"])
    closing.add_instrument("Money", "financial", {"H": -5.0, "G": 5.0})
    assert check_stock_flow(opening, closing, t).ok
    bad = BalanceSheetMatrix(["H", "G"])
    bad.add_instrument("Money", "financial", {"H": -4.0, "G": 4.0})
    rep = check_stock_flow(opening, bad, t)
    assert not rep.ok


def test_stock_flow_with_revaluation_and_capital():
    t = TransactionFlowMatrix(["H", "G"])
    o = BalanceSheetMatrix(["H", "G"])
    o.add_instrument("Bond", "financial", {"H": 10.0, "G": -10.0})
    o.add_instrument("Capital", "tangible", {"H": 20.0})
    c = BalanceSheetMatrix(["H", "G"])
    c.add_instrument("Bond", "financial", {"H": 12.0, "G": -12.0})  # +2 price gain
    c.add_instrument("Capital", "tangible", {"H": 23.0})  # +1 net investment, +2 gain
    t.add_transaction("Bond", "financial", {"H": 0.0, "G": 0.0})
    reval = pd.DataFrame({"H": [2.0, 2.0], "G": [-2.0, 0.0]}, index=["Bond", "Capital"])
    nfa = pd.Series({"H": 1.0, "G": 0.0})
    rep = check_stock_flow(o, c, t, reval, nfa)
    assert rep.ok, rep.to_frame()
    # claim a wrong investment amount -> failure
    assert not check_stock_flow(o, c, t, reval, pd.Series({"H": 0.0, "G": 0.0})).ok


def test_unmatched_financial_row_flagged():
    t = TransactionFlowMatrix(["H", "G"])
    t.add_transaction("Ghost", "financial", {"H": 1.0, "G": -1.0})
    b = make_bsm()
    rep = check_stock_flow(b, b, t)
    assert "stock_flow.unmatched_financial_row" in set(rep.to_frame()["check"])


def test_standard_classification():
    c = standard_classification()
    assert len(c.sectors) == 6
    assert c.sector_names[0] == "Households"
    c.require_sectors(["Banks"])
    with pytest.raises(AccountingError):
        c.require_sectors(["Mars"])
    assert type(c).from_dict(c.to_dict()).sector_names == c.sector_names
