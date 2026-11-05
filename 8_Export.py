"""Export: matrices, simulation outputs, model specification and reports."""

import streamlit as st

from sfc_modeling.dashboard import state
from sfc_modeling.reporting import frame_to_csv_bytes, markdown_report, result_bundle

st.set_page_config(page_title="Export", layout="wide")
st.title("Export")
spec = state.get_spec()
st.download_button("Model specification (JSON)", spec.to_json(), f"{spec.name}_model.json", "application/json")
results = state.get_results()
if not results:
    st.info("Run a simulation to enable result exports.")
    st.stop()
name = st.selectbox("Scenario", list(results))
res = results[name]
st.download_button("Simulation data (CSV)", frame_to_csv_bytes(res.data), f"{spec.name}_{name}_data.csv", "text/csv")
st.download_button("Solver diagnostics (CSV)", frame_to_csv_bytes(res.diagnostics.set_index("period")),
                   f"{spec.name}_{name}_diagnostics.csv", "text/csv")
if res.has_accounting:
    t = st.slider("Period for matrices", 1, res.periods, res.periods)
    tfm, bsm, _, _ = res.matrices(t)
    st.download_button(f"TFM t={t} (CSV)", frame_to_csv_bytes(tfm.to_frame()), f"{spec.name}_tfm_t{t}.csv", "text/csv")
    st.download_button(f"BSM t={t} (CSV)", frame_to_csv_bytes(bsm.to_frame()), f"{spec.name}_bsm_t{t}.csv", "text/csv")
st.download_button("Report (Markdown)", markdown_report(res), f"{spec.name}_{name}_report.md", "text/markdown")
st.download_button("Full reproducible bundle (ZIP)", result_bundle(res), f"{spec.name}_{name}_bundle.zip", "application/zip")
