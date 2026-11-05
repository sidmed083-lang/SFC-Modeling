"""Exports: CSV, JSON, Markdown report and a reproducible ZIP bundle."""

from __future__ import annotations

import io
import platform
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

from .. import __version__
from ..accounting.validation import ValidationReport
from ..simulation.engine import SimulationResult


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv().encode("utf-8")


def markdown_report(result: SimulationResult, report: Optional[ValidationReport] = None) -> str:
    spec = result.model.spec
    report = report if report is not None else (result.validate() if result.has_accounting else None)
    lines = [
        f"# SFC-Modeling report: {spec.name} / {result.name}",
        "",
        f"- Periods simulated: {result.periods}",
        f"- Solver tolerance: {result.tol:g}",
        f"- Random seed (spec): {spec.random_seed}",
        f"- sfc_modeling {__version__}, Python {platform.python_version()}, NumPy {np.__version__}",
        "",
        "## Solver diagnostics",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in result.log.summary().items()]
    lines += ["", "## Accounting validation", ""]
    lines.append(report.summary() if report is not None else "No accounting specification.")
    if report is not None and not report.ok:
        lines += ["", report.to_frame().head(20).to_markdown(index=False)
                  if _has_tabulate() else report.to_frame().head(20).to_string(index=False)]
    if result.model.checks:
        res = result.consistency_residuals()
        lines += ["", "## Redundant-identity residuals (max abs)", ""]
        lines += [f"- {c}: {res[c].abs().max():.3e}" for c in res.columns]
    lines += ["", "## Final period values", "", "```", result.data.iloc[-1].to_string(), "```", ""]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


def result_bundle(result: SimulationResult, report: Optional[ValidationReport] = None) -> bytes:
    """ZIP with data, parameters, diagnostics, matrices, validation, spec and report."""
    if result.has_accounting and report is None:
        report = result.validate()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.csv", frame_to_csv_bytes(result.data))
        z.writestr("parameters.csv", frame_to_csv_bytes(result.parameters))
        z.writestr("diagnostics.csv", frame_to_csv_bytes(result.diagnostics.set_index("period")))
        z.writestr("model_spec.json", result.model.spec.to_json())
        z.writestr("scenario.json", __import__("json").dumps(result.scenario.to_dict(), indent=2))
        if result.has_accounting:
            for t in sorted({0, 1, result.periods}):
                if t == 0:
                    _, bsm, _, _ = result.matrices(0)
                    z.writestr("bsm_t0.csv", frame_to_csv_bytes(bsm.to_frame()))
                    continue
                tfm, bsm, _, _ = result.matrices(t)
                z.writestr(f"tfm_t{t}.csv", frame_to_csv_bytes(tfm.to_frame()))
                z.writestr(f"bsm_t{t}.csv", frame_to_csv_bytes(bsm.to_frame()))
            z.writestr("validation_issues.csv", frame_to_csv_bytes(report.to_frame()))
        z.writestr("report.md", markdown_report(result, report))
    return buf.getvalue()
