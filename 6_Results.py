"""Results: compare variables, sectoral balances and outcomes across scenarios."""

import streamlit as st

from sfc_modeling.analysis import net_financial_wealth, net_lending
from sfc_modeling.dashboard import state
from sfc_modeling.simulation import compare
from sfc_modeling.visualization import bar_figure, line_figure

st.set_page_config(page_title="Results", layout="wide")
st.title("Results")
results = state.get_results()
if not results:
    st.info("Run a simulation first.")
    st.stop()
first = next(iter(results.values()))
cols = st.multiselect("Variables", list(first.data.columns), default=list(first.data.columns[:3]))
chosen = st.multiselect("Scenarios", list(results), default=list(results))
mode = st.radio("Show", ["Levels", "Difference from Baseline"], horizontal=True)
if cols and chosen:
    sel = {k: results[k] for k in chosen}
    if mode.startswith("Diff") and "Baseline" in sel and len(sel) > 1:
        df = compare(sel, cols, reference="Baseline")
        title = "Difference from Baseline"
    else:
        df, title = compare(sel, cols), "Levels"
    for var in cols:
        sub = df.xs(var, axis=1, level=1)
        st.plotly_chart(line_figure(sub, f"{var} – {title}"), use_container_width=True)
    with st.expander("Data table"):
        st.dataframe(df, use_container_width=True)

st.subheader("Sectoral balances")
name = st.selectbox("Scenario for balances", list(results))
res = results[name]
if res.has_accounting:
    nl = net_lending(res)
    st.plotly_chart(bar_figure(nl, "Net lending (+) / borrowing (−) by sector"), use_container_width=True)
    st.plotly_chart(line_figure(net_financial_wealth(res), "Net financial wealth by sector"),
                    use_container_width=True)
    st.caption(f"Max |Σ sectors| of net lending: {nl.sum(axis=1).abs().max():.2e} (must be ≈ 0).")
else:
    st.info("No accounting specification: sectoral balances unavailable.")
