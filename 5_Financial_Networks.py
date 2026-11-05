"""Financial networks: flows and exposures between sectors (NetworkX + Plotly)."""

import streamlit as st

from sfc_modeling.dashboard import state
from sfc_modeling.networks import edge_table, exposure_network, flow_network, node_summary, propagate_shock
from sfc_modeling.visualization import network_figure

st.set_page_config(page_title="Financial Networks", layout="wide")
st.title("Financial Networks")
results = state.get_results()
if not results:
    st.info("Run a simulation first.")
    st.stop()
name = st.selectbox("Scenario", list(results))
res = results[name]
if not res.has_accounting:
    st.warning("The model has no accounting specification.")
    st.stop()
t = st.slider("Period", 1, res.periods, res.periods)
kind = st.radio("Network", ["Payment flows (current + capital)", "Financing flows (financial account)",
                            "Financial claims (stock exposures)"], horizontal=True)
tfm, bsm, _, _ = res.matrices(t)
if kind.startswith("Payment"):
    g, direction = flow_network(tfm), "downstream"
elif kind.startswith("Financing"):
    g, direction = flow_network(tfm, kinds=("financial",)), "downstream"
else:
    g, direction = exposure_network(bsm), "upstream"
st.caption("Flows are per-period amounts, claims are stocks. Edges are allocated pro rata when a "
           "transaction/instrument has several counterparties (the TFM/BSM do not record bilateral links).")
if g.number_of_edges() == 0:
    st.info("No edges in this network for the selected period.")
else:
    st.plotly_chart(network_figure(g, f"{res.model.spec.name} – {name}, t={t}"), use_container_width=True)
a, b = st.columns(2)
a.subheader("Nodes")
a.dataframe(node_summary(g).style.format("{:.4f}"), use_container_width=True)
b.subheader("Edges")
b.dataframe(edge_table(g), use_container_width=True)

st.subheader("Exposure / propagation (stylised)")
c1, c2, c3 = st.columns(3)
origin = c1.selectbox("Origin of shock", list(g.nodes))
amount = c2.number_input("Shock amount", value=10.0)
pt = c3.slider("Pass-through", 0.0, 1.0, 1.0)
if g.number_of_edges():
    st.dataframe(propagate_shock(g, origin, amount, rounds=5, pass_through=pt, direction=direction),
                 use_container_width=True)
