"""SFC-Modeling dashboard: Overview.  Launch with ``streamlit run app.py``."""

import streamlit as st

from sfc_modeling import __version__
from sfc_modeling.dashboard import state
from sfc_modeling.model import ModelSpecification

st.set_page_config(page_title="SFC-Modeling", layout="wide")
st.title("SFC-Modeling")
st.caption(f"Stock-flow consistent macroeconomic modelling · v{__version__}")

with st.sidebar:
    st.header("Model")
    choice = st.selectbox("Load an example", list(state.EXAMPLES))
    if st.button("Load example"):
        state.set_spec(state.EXAMPLES[choice]())
        st.rerun()
    up = st.file_uploader("…or upload a model JSON", type="json")
    if up is not None and st.button("Load uploaded model"):
        try:
            state.set_spec(ModelSpecification.from_json(up.read().decode("utf-8")))
            st.rerun()
        except Exception as exc:
            st.error(f"Could not read model: {exc}")

spec = state.get_spec()
model = state.compiled(spec)
st.subheader(f"Model: {spec.name}")
st.write(spec.description)

if model is not None:
    c = st.columns(5)
    c[0].metric("Equations", len(model.endogenous))
    c[1].metric("Parameters", len(model.parameters))
    c[2].metric("Exogenous", len(model.exogenous))
    c[3].metric("Blocks", len(model.blocks))
    c[4].metric("Simultaneous blocks", sum(b.simultaneous for b in model.blocks))

results = state.get_results()
st.subheader("Accounting consistency and key indicators")
if not results:
    st.info("No simulation yet. Open **Simulation** in the sidebar and press *Run*.")
else:
    name = st.selectbox("Scenario", list(results))
    res = results[name]
    if res.has_accounting:
        rep = res.validate()
        (st.success if rep.ok else st.error)(rep.summary())
    else:
        st.warning("This model has no accounting specification, so consistency cannot be validated.")
    st.write("Solver:", res.log.summary())
    st.dataframe(res.data.iloc[[0, 1, len(res.data) // 2, -1]].T, use_container_width=True)
