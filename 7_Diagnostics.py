"""Diagnostics: accounting discrepancies, convergence and stability."""

import numpy as np
import pandas as pd
import streamlit as st

from sfc_modeling.dashboard import state
from sfc_modeling.solver import solve_steady_state, stability_analysis

st.set_page_config(page_title="Diagnostics", layout="wide")
st.title("Diagnostics")
results = state.get_results()
if not results:
    st.info("Run a simulation first.")
    st.stop()
name = st.selectbox("Scenario", list(results))
res = results[name]

st.subheader("Accounting validation")
c1, c2 = st.columns(2)
atol = c1.select_slider("Absolute tolerance", [1e-12, 1e-10, 1e-8, 1e-6], value=1e-8, format_func=lambda x: f"{x:g}")
rtol = c2.select_slider("Relative tolerance", [1e-12, 1e-10, 1e-9, 1e-6], value=1e-9, format_func=lambda x: f"{x:g}")
rep = res.validate(atol, rtol)
(st.success if rep.ok else st.error)(rep.summary())
if not rep.ok:
    st.dataframe(rep.to_frame(), use_container_width=True)

st.subheader("Redundant identities")
if res.model.checks:
    cr = res.consistency_residuals()
    st.dataframe(cr.abs().max().rename("max |residual|").to_frame(), use_container_width=True)
    st.line_chart(cr)
else:
    st.info("No consistency checks defined (add them in the Equation Editor).")

st.subheader("Solver convergence")
st.json(res.log.summary())
diag = res.diagnostics
st.dataframe(diag[diag["simultaneous"]].head(200), use_container_width=True)

st.subheader("Local stability at the steady state")
if st.button("Compute eigenvalues"):
    model = res.model
    ss = solve_steady_state(model)
    if not ss.converged:
        st.error("Steady state did not converge; stability analysis skipped.")
    else:
        exog = {n: float(res.data[n].iloc[0]) for n in model.exogenous}
        st_res = stability_analysis(model, model.parameters, exog, ss.values.to_dict())
        eig = st_res["eigenvalues"]
        st.dataframe(pd.DataFrame({"eigenvalue": eig, "modulus": np.abs(eig)}), use_container_width=True)
        st.write(f"States: {st_res['states']}  ·  unit roots: {st_res['n_unit_roots']}  ·  "
                 f"largest non-unit modulus: {st_res['max_modulus_non_unit']:.4f}")
        (st.success if st_res["stable"] else st.error)("Stable (excluding accounting unit roots)" if st_res["stable"] else "Unstable")
        st.caption("Unit roots usually reflect redundant stock-flow equations (e.g. Hs and Hh); "
                   "they are not a sign of instability by themselves. Exogenous values are taken from period 0.")
