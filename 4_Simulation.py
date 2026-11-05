"""Simulation: baseline, policy scenarios, steady state, sensitivity, Monte Carlo."""

import pandas as pd
import streamlit as st

from sfc_modeling.dashboard import state
from sfc_modeling.simulation import Scenario, ScenarioError, Shock, final_value, monte_carlo, one_at_a_time, summarize
from sfc_modeling.solver import solve_steady_state

st.set_page_config(page_title="Simulation", layout="wide")
st.title("Simulation")
spec = state.get_spec()
model = state.compiled(spec)
if model is None:
    st.stop()

c1, c2 = st.columns(2)
periods = c1.slider("Periods", 5, 400, 60)
tol = c2.select_slider("Solver tolerance", [1e-6, 1e-8, 1e-9, 1e-10, 1e-12], value=1e-9, format_func=lambda x: f"{x:g}")

st.subheader("Scenarios")
st.caption("Shocks change a parameter or exogenous variable from `start` to `end` (blank = permanent). "
           "Use exactly one of value / delta / pct (pct: 0.1 = +10 %).")
default = pd.DataFrame({"scenario": ["Policy"], "target": [(model.exogenous or list(model.parameters))[0]],
                        "start": [10], "end": [None], "value": [None], "delta": [5.0], "pct": [None]})
sc_edit = st.data_editor(default, num_rows="dynamic", use_container_width=True, key="shocks")

if st.button("Run baseline + scenarios", type="primary"):
    try:
        scenarios: dict[str, Scenario] = {}
        for _, r in sc_edit.iterrows():
            if not str(r["scenario"]).strip() or str(r["scenario"]) == "nan":
                continue
            nn = lambda v: None if pd.isna(v) else float(v)  # noqa: E731
            shock = Shock(str(r["target"]), int(r["start"]), nn(r["value"]), nn(r["delta"]), nn(r["pct"]),
                          None if pd.isna(r["end"]) else int(r["end"]))
            scenarios.setdefault(r["scenario"], Scenario(r["scenario"])).shocks.append(shock)
        st.session_state["scenarios"] = list(scenarios.values())
        state.run_all(periods, tol)
        if state.get_results():
            st.success(f"Ran {len(state.get_results())} simulation(s). See **Results** and **Diagnostics**.")
    except (ScenarioError, ValueError, KeyError) as exc:
        st.error(f"Invalid scenario: {exc}")

st.divider()
st.subheader("Steady state")
if st.button("Solve static steady state"):
    ss = solve_steady_state(model)
    st.session_state["steady"] = ss
    (st.success if ss.converged else st.error)(f"Converged: {ss.converged} · residual {ss.residual_norm:.2e}")
    if not ss.unique:
        st.warning("The steady state is not locally unique (Jacobian rank "
                   f"{ss.jacobian_rank} < {ss.n_unknowns}): some stocks depend on history. "
                   "Add the model's redundant accounting identities as consistency checks.")
    st.dataframe(ss.values.rename("steady state").to_frame(), use_container_width=True)

st.divider()
st.subheader("Sensitivity and Monte Carlo")
inputs = list(model.parameters) + model.exogenous
metric_var = st.selectbox("Metric: final value of", model.endogenous)
tab_a, tab_b = st.tabs(["One-at-a-time", "Monte Carlo"])
with tab_a:
    target = st.selectbox("Input", inputs, key="sa_t")
    base = model.parameters.get(target, spec.exogenous_value(target, 0) if target in model.exogenous else 0.0)
    lo, hi = st.columns(2)
    lo_v = lo.number_input("From", value=float(base) * 0.5)
    hi_v = hi.number_input("To", value=float(base) * 1.5)
    n = st.slider("Points", 3, 25, 9)
    if st.button("Run sweep"):
        import numpy as np
        out = one_at_a_time(spec, target, list(np.linspace(lo_v, hi_v, n)), final_value(metric_var), periods)
        st.line_chart(out.set_index(target)["metric"])
        st.dataframe(out, use_container_width=True)
with tab_b:
    target = st.selectbox("Input", inputs, key="mc_t")
    base = model.parameters.get(target, spec.exogenous_value(target, 0) if target in model.exogenous else 0.0)
    sd = st.number_input("Std. deviation (normal draw around baseline)", value=abs(float(base)) * 0.1 or 0.1)
    draws = st.slider("Draws", 20, 500, 100)
    seed = st.number_input("Seed", value=int(spec.random_seed or 0), step=1)
    if st.button("Run Monte Carlo"):
        mc = monte_carlo(spec, {target: ("normal", float(base), float(sd))}, final_value(metric_var), periods, draws, int(seed))
        st.write(summarize(mc))
        st.bar_chart(mc["metric"].dropna().round(2).value_counts().sort_index())
        st.caption(f"Seed used: {mc.attrs['seed']} (reproducible).")
