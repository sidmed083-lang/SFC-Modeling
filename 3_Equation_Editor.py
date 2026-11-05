"""Equation editor: equations, parameters, exogenous inputs, initial values."""

import pandas as pd
import streamlit as st

from sfc_modeling.dashboard import state
from sfc_modeling.model import Equation, ModelSpecification

st.set_page_config(page_title="Equation Editor", layout="wide")
st.title("Equation Editor")
spec = state.get_spec()
st.markdown(
    "Write `variable = expression`. Use `X(-1)` for lags and `^` or `**` for powers; functions: "
    "`exp, log, sqrt, Abs, Max, Min, tanh`. Each endogenous variable has exactly one equation."
)

eq_df = pd.DataFrame(
    [{"variable": e.lhs, "expression": e.rhs, "kind": e.kind, "description": e.description} for e in spec.equations]
)
eq_edit = st.data_editor(eq_df, num_rows="dynamic", use_container_width=True, key="eqs",
                         column_config={"kind": st.column_config.SelectboxColumn(options=["behavioral", "identity"])})
c1, c2 = st.columns(2)
with c1:
    st.subheader("Parameters")
    p_edit = st.data_editor(
        pd.DataFrame({"name": list(spec.parameters), "value": list(spec.parameters.values())}),
        num_rows="dynamic", use_container_width=True, key="params")
    st.subheader("Initial values (period 0)")
    i_edit = st.data_editor(
        pd.DataFrame({"name": list(spec.initial), "value": list(spec.initial.values())}),
        num_rows="dynamic", use_container_width=True, key="init")
with c2:
    st.subheader("Exogenous variables")
    st.caption("A number, or a comma-separated list indexed by period (last value is held).")
    x_edit = st.data_editor(
        pd.DataFrame({"name": list(spec.exogenous),
                      "value": [", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
                                for v in spec.exogenous.values()]}),
        num_rows="dynamic", use_container_width=True, key="exog")
    st.subheader("Consistency checks")
    st.caption("Redundant identities that must equal zero, one per line (e.g. `Hs - Hh`).")
    chk = st.text_area("checks", "\n".join(spec.consistency_checks), label_visibility="collapsed")


def _num_list(text: str):
    parts = [float(x) for x in str(text).split(",") if x.strip()]
    return parts[0] if len(parts) == 1 else parts


if st.button("Validate and apply", type="primary"):
    try:
        new = ModelSpecification(
            name=spec.name, description=spec.description,
            equations=[Equation(r["variable"].strip(), r["expression"].strip(), r["kind"] or "behavioral",
                                r["description"] or "") for _, r in eq_edit.iterrows() if str(r["variable"]).strip()],
            parameters={r["name"].strip(): float(r["value"]) for _, r in p_edit.iterrows() if str(r["name"]).strip()},
            exogenous={r["name"].strip(): _num_list(r["value"]) for _, r in x_edit.iterrows() if str(r["name"]).strip()},
            initial={r["name"].strip(): float(r["value"]) for _, r in i_edit.iterrows() if str(r["name"]).strip()},
            consistency_checks=[c.strip() for c in chk.splitlines() if c.strip()],
            accounting=spec.accounting, descriptions=spec.descriptions, random_seed=spec.random_seed,
        )
        model = state.compiled(new)
        if model is not None:
            state.set_spec(new)
            st.success("Model is valid and has been applied. Previous results were cleared.")
            spec = new
    except Exception as exc:
        st.error(f"Could not apply: {exc}")

model = state.compiled(spec, show_error=False)
if model is not None:
    st.subheader("Dependency analysis")
    st.caption("Blocks are solved in this order; *simultaneous* blocks contain circular dependencies "
               "and are solved numerically (SciPy `root`).")
    st.dataframe(model.dependency_table(), use_container_width=True)
