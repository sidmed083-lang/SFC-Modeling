"""Accounting matrices: edit TFM / BSM templates and inspect evaluated matrices."""

import streamlit as st

from sfc_modeling.dashboard import state
from sfc_modeling.visualization import matrix_heatmap

st.set_page_config(page_title="Accounting Matrices", layout="wide")
st.title("Accounting Matrices")
spec = state.get_spec()
if spec.accounting is None:
    st.warning("This model has no accounting specification.")
    st.stop()

st.markdown(
    "**Sign convention** – TFM: `+` receipt/source, `−` payment/use; financial rows: asset acquisition `−`, "
    "liability issuance `+`. BSM: `+` asset, `−` liability. TFM financial rows must be named after "
    "their BSM instrument."
)
tab_t, tab_e = st.tabs(["Templates (expressions)", "Evaluated matrices"])
with tab_t:
    acc = spec.accounting
    st.caption("Each cell is an expression in the model variables, e.g. `-C` or `r(-1)*Bh(-1)`.")
    tx = st.data_editor(acc.transactions_frame(), num_rows="dynamic", use_container_width=True, key="tfm_t")
    ins = st.data_editor(acc.instruments_frame(), num_rows="dynamic", use_container_width=True, key="bsm_t")
    if st.button("Apply templates"):
        new = acc.with_frames(tx, ins)
        old, spec.accounting = spec.accounting, new
        if state.compiled(spec) is None:
            spec.accounting = old
        else:
            st.session_state["results"] = {}
            st.success("Templates applied. Re-run the simulation.")

with tab_e:
    results = state.get_results()
    if not results:
        st.info("Run a simulation first.")
    else:
        name = st.selectbox("Scenario", list(results), key="acc_sc")
        res = results[name]
        t = st.slider("Period", 0, res.periods, res.periods)
        tfm, bsm, _, _ = res.matrices(t)
        if t >= 1:
            st.subheader("Transactions-flow matrix")
            f = tfm.to_frame()
            st.dataframe(f.style.format("{:.4f}"), use_container_width=True)
            st.plotly_chart(matrix_heatmap(f, f"TFM, t={t}"), use_container_width=True)
            rep = tfm.validate()
            (st.success if rep.ok else st.error)("TFM: " + rep.summary())
        st.subheader("Balance-sheet matrix")
        b = bsm.to_frame()
        st.dataframe(b.style.format("{:.4f}"), use_container_width=True)
        st.plotly_chart(matrix_heatmap(b, f"BSM, t={t}"), use_container_width=True)
        rep = bsm.validate()
        (st.success if rep.ok else st.error)("BSM: " + rep.summary())
