"""Streamlit session-state helpers shared by all dashboard pages."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ..library import pc_model, revaluation_model, sim_model
from ..model.specification import CompiledModel, ModelSpecification, ModelSpecificationError
from ..simulation import Scenario, SimulationResult, simulate
from ..solver.diagnostics import ConvergenceError

EXAMPLES = {
    "SIM (Godley & Lavoie, Ch. 3)": sim_model,
    "PC (Godley & Lavoie, Ch. 4)": pc_model,
    "BOND-REVAL (capital gains demo)": revaluation_model,
}


def get_spec() -> ModelSpecification:
    if "spec" not in st.session_state:
        set_spec(sim_model())
    return st.session_state["spec"]


def set_spec(spec: ModelSpecification) -> None:
    st.session_state["spec"] = spec
    st.session_state["results"] = {}
    st.session_state["scenarios"] = []
    st.session_state["steady"] = None


def get_results() -> dict[str, SimulationResult]:
    return st.session_state.setdefault("results", {})


def get_scenarios() -> list[Scenario]:
    return st.session_state.setdefault("scenarios", [])


def compiled(spec: Optional[ModelSpecification] = None, show_error: bool = True) -> Optional[CompiledModel]:
    spec = spec or get_spec()
    try:
        return spec.compile()
    except ModelSpecificationError as exc:
        if show_error:
            st.error(f"Model specification error:\n\n{exc}")
        return None
    except Exception as exc:  # accounting errors etc.
        if show_error:
            st.error(f"Model error: {exc}")
        return None


def run_all(periods: int, tol: float) -> None:
    """Run the baseline and every defined scenario; store the results."""
    spec = get_spec()
    model = compiled(spec)
    if model is None:
        return
    results: dict[str, SimulationResult] = {}
    try:
        results["Baseline"] = simulate(model, periods, None, tol)
        for sc in get_scenarios():
            results[sc.name] = simulate(model, periods, sc, tol)
    except (ConvergenceError, ValueError) as exc:
        st.error(f"Simulation failed: {exc}")
        return
    st.session_state["results"] = results
