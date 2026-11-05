import pytest

from sfc_modeling.library import pc_model, sim_model
from sfc_modeling.networks import (
    adjacency,
    edge_table,
    exposure_network,
    flow_network,
    node_summary,
    propagate_shock,
)
from sfc_modeling.simulation import simulate


def test_flow_network_matches_tfm():
    r = simulate(sim_model(), 30)
    tfm, _, _, _ = r.matrices(30)
    g = flow_network(tfm)
    assert g.graph["kind"] == "flow"
    d = r.data.loc[30]
    assert g["Households"]["Production"]["weight"] == pytest.approx(d["C"])
    assert g["Government"]["Production"]["weight"] == pytest.approx(d["G"])
    assert g["Production"]["Households"]["weight"] == pytest.approx(d["Y"])
    assert g["Households"]["Government"]["weight"] == pytest.approx(d["T"])
    # total edge weight equals total payments in the (bilateral) rows
    total = sum(w for _, _, w in g.edges(data="weight"))
    assert total == pytest.approx(d["C"] + d["G"] + d["Y"] + d["T"])


def test_flow_network_pro_rata_allocation_conserves_amounts():
    r = simulate(pc_model(), 30)
    tfm, _, _, _ = r.matrices(30)
    g = flow_network(tfm)
    # interest on bills: Government pays, Households + CentralBank receive
    comp_hh = g["Government"]["Households"]["components"]["Interest on bills"]
    comp_cb = g["Government"]["CentralBank"]["components"]["Interest on bills"]
    d = r.data
    assert comp_hh + comp_cb == pytest.approx(d.loc[29, "r"] * d.loc[29, "Bs"])
    ns = node_summary(g)
    assert ns["pagerank"].sum() == pytest.approx(1.0)
    assert g.nodes["Households"]["net_lending"] == pytest.approx(tfm.net_lending()["Households"])


def test_exposure_network_is_stock_and_consistent_with_bsm():
    r = simulate(pc_model(), 40)
    _, bsm, _, _ = r.matrices(40)
    e = exposure_network(bsm)
    assert e.graph["kind"] == "stock"
    d = r.data.loc[40]
    # bills: households and central bank are creditors of government
    assert e["Households"]["Government"]["components"]["Bills"] == pytest.approx(d["Bh"])
    assert e["CentralBank"]["Government"]["components"]["Bills"] == pytest.approx(d["Bcb"])
    # money: households are creditors of the central bank
    assert e["Households"]["CentralBank"]["components"]["Money"] == pytest.approx(d["Hh"])
    # total claims equal total financial liabilities
    total = sum(w for _, _, w in e.edges(data="weight"))
    assert total == pytest.approx(bsm.financial_liabilities().sum())
    assert not edge_table(e).empty
    assert adjacency(e).shape == (4, 4)


def test_shock_propagation():
    r = simulate(pc_model(), 40)
    _, bsm, _, _ = r.matrices(40)
    e = exposure_network(bsm)
    out = propagate_shock(e, "Government", 10.0, rounds=4, direction="upstream")
    assert out.loc[0, "Government"] == 10.0
    assert out.loc[1].sum() == pytest.approx(10.0)  # all of the loss lands on creditors
    damp = propagate_shock(e, "Government", 10.0, rounds=4, pass_through=0.5, direction="upstream")
    assert damp.loc[1].sum() == pytest.approx(5.0)
    with pytest.raises(KeyError):
        propagate_shock(e, "Mars", 1.0)
    with pytest.raises(ValueError):
        propagate_shock(e, "Government", 1.0, pass_through=2.0)
    with pytest.raises(ValueError):
        propagate_shock(e, "Government", 1.0, direction="sideways")
