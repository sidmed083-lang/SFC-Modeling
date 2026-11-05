"""Benchmark models with known solutions (Godley & Lavoie, 2007)."""

import numpy as np
import pytest

from sfc_modeling.library import (
    pc_analytic_steady_state,
    pc_model,
    sim_analytic_first_period,
    sim_analytic_steady_state,
    sim_model,
)
from sfc_modeling.model import Equation, ModelSpecification
from sfc_modeling.simulation import simulate
from sfc_modeling.solver import ConvergenceError, solve_steady_state, stability_analysis


def test_sim_first_period_matches_textbook():
    r = simulate(sim_model(), 1)
    assert r.data.loc[1, "Y"] == pytest.approx(38.46153846, rel=1e-8)
    assert r.data.loc[1, "Y"] == pytest.approx(sim_analytic_first_period(sim_model()), rel=1e-10)
    assert r.data.loc[1, "Hh"] == pytest.approx(12.3076923, rel=1e-6)


def test_sim_matches_closed_form_recursion_every_period():
    spec = sim_model()
    p, G = spec.parameters, spec.exogenous["G"]
    H = 0.0
    r = simulate(spec, 40)
    for t in range(1, 41):
        Y = (G + p["alpha2"] * H) / (1 - p["alpha1"] * (1 - p["theta"]))
        H = H + (1 - p["theta"]) * Y - (p["alpha1"] * (1 - p["theta"]) * Y + p["alpha2"] * H)
        assert r.data.loc[t, "Y"] == pytest.approx(Y, rel=1e-9)
        assert r.data.loc[t, "Hh"] == pytest.approx(H, rel=1e-9, abs=1e-9)


def test_sim_converges_to_analytic_steady_state():
    spec = sim_model()
    r = simulate(spec, 400)
    ss = sim_analytic_steady_state(spec)
    for k, v in ss.items():
        assert r.data[k].iloc[-1] == pytest.approx(v, rel=1e-8)


def test_sim_static_solver_matches_analytic():
    spec = sim_model()
    res = solve_steady_state(spec.compile())
    assert res.converged and res.unique
    for k, v in sim_analytic_steady_state(spec).items():
        assert res.values[k] == pytest.approx(v, rel=1e-8)


def test_pc_static_solver_and_dynamics_match_analytic():
    spec = pc_model()
    ana = pc_analytic_steady_state(spec)
    res = solve_steady_state(spec.compile())
    assert res.converged and res.unique
    r = simulate(spec, 400)
    for k, v in ana.items():
        assert res.values[k] == pytest.approx(v, rel=1e-7)
        assert r.data[k].iloc[-1] == pytest.approx(v, rel=1e-6)


def test_steady_state_flags_non_uniqueness_without_closure_identity():
    spec = pc_model()
    spec.consistency_checks = ["Hs - Hh"]  # central-bank balance sheet identity removed
    res = solve_steady_state(spec.compile())
    assert not res.unique


def test_sim_stability_eigenvalue_matches_hand_derivation():
    spec = sim_model()
    m = spec.compile()
    ss = solve_steady_state(m)
    st = stability_analysis(m, m.parameters, {"G": 20.0, "W": 1.0}, ss.values.to_dict())
    p = spec.parameters
    lam = 1 - p["alpha2"] * p["theta"] / (1 - p["alpha1"] * (1 - p["theta"]))
    eig = sorted(np.real(st["eigenvalues"]))
    assert eig[0] == pytest.approx(lam, rel=1e-5)
    assert eig[1] == pytest.approx(1.0, abs=1e-6)
    assert st["n_unit_roots"] == 1 and st["stable"]


def test_nonlinear_simultaneous_block():
    # x = 1 + sqrt(y), y = 2x + 1  =>  x = 4, y = 9
    s = ModelSpecification(
        "nl", [Equation.from_string("x = 1 + sqrt(y)"), Equation.from_string("y = 2*x + 1")],
        initial={"x": 1.0, "y": 1.0},
    )
    r = simulate(s, 2)
    assert r.data.loc[1, "x"] == pytest.approx(4.0, rel=1e-9)
    assert r.data.loc[2, "y"] == pytest.approx(9.0, rel=1e-9)
    assert r.diagnostics["simultaneous"].any()


def test_dynamic_lag_matches_closed_form():
    s = ModelSpecification("ar", [Equation.from_string("x = 0.5*x(-1) + 1")])
    r = simulate(s, 10)
    for t in range(1, 11):
        assert r.data.loc[t, "x"] == pytest.approx(2 * (1 - 0.5**t), rel=1e-12)


def test_unsolvable_block_raises_convergence_error():
    s = ModelSpecification("bad", [Equation.from_string("x = x + 1")])
    with pytest.raises(ConvergenceError):
        simulate(s, 1)
    r = simulate(s, 1, raise_on_failure=False)
    assert not r.log.all_converged
