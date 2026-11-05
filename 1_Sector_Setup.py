"""Sector setup: sectors, financial instruments, behavioural rules."""

import pandas as pd
import streamlit as st

from sfc_modeling.accounting import Classification, Instrument, InstrumentKind, Sector, SectorKind
from sfc_modeling.dashboard import state

st.set_page_config(page_title="Sector Setup", layout="wide")
st.title("Sector Setup")
spec = state.get_spec()
if spec.accounting is None:
    st.warning("This model has no accounting specification.")
    st.stop()
acc = spec.accounting
cls = acc.classification or Classification(
    [Sector(s) for s in acc.sectors], [Instrument(i.name) for i in acc.instruments]
)

st.subheader("Sectors")
st.caption("Available kinds: " + ", ".join(k.value for k in SectorKind))
sec_df = pd.DataFrame(
    [{"name": s.name, "kind": s.kind.value, "description": s.description} for s in cls.sectors]
)
sec_edit = st.data_editor(sec_df, num_rows="dynamic", use_container_width=True, key="sectors")

st.subheader("Financial instruments")
st.caption("Instrument kinds: " + ", ".join(k.value for k in InstrumentKind))
ins_df = pd.DataFrame(
    [{"name": i.name, "kind": i.kind.value, "description": i.description} for i in cls.instruments]
)
ins_edit = st.data_editor(ins_df, num_rows="dynamic", use_container_width=True, key="instruments")

if st.button("Apply classification"):
    try:
        new_cls = Classification(
            [Sector(r["name"], SectorKind(r["kind"]), r["description"] or "") for _, r in sec_edit.iterrows() if r["name"]],
            [Instrument(r["name"], InstrumentKind(r["kind"]), r["description"] or "") for _, r in ins_edit.iterrows() if r["name"]],
        )
        new_sectors = new_cls.sector_names
        tx, ins = acc.transactions_frame(), acc.instruments_frame()
        for frame in (tx, ins):
            for s in new_sectors:
                if s not in frame.columns:
                    frame[s] = ""
        keep = lambda f, lead: f[[*lead, *new_sectors]]  # noqa: E731
        acc2 = type(acc)(new_sectors, [], [], acc.revaluations, acc.nonfinancial_acquisitions, new_cls)
        acc2 = acc2.with_frames(keep(tx, ["transaction", "kind"]), keep(ins, ["instrument", "kind"]))
        spec.accounting = acc2
        if state.compiled(spec) is not None:
            st.success("Classification applied. Re-run the simulation to refresh results.")
            st.session_state["results"] = {}
    except Exception as exc:
        st.error(f"Invalid classification: {exc}")

st.subheader("Behavioural rules")
st.caption("Edit equations on the **Equation Editor** page.")
st.dataframe(
    pd.DataFrame([{"variable": e.lhs, "equation": str(e), "kind": e.kind, "description": e.description}
                  for e in spec.equations]),
    use_container_width=True,
)
