"""Validate accounting identities and benchmark solutions of the bundled models.

Exit status is non-zero on any failure.  Used by CI.
"""

import sys

from sfc_modeling.library import (
    pc_analytic_steady_state,
    pc_model,
    sim_analytic_first_period,
    sim_analytic_steady_state,
    sim_model,
)
from sfc_modeling.simulation import Scenario, Shock, simulate
from sfc_modeling.solver import solve_steady_state


def main() -> int:
    failures = []
    for spec, analytic in ((sim_model(), sim_analytic_steady_state), (pc_model(), pc_analytic_steady_state)):
        scenario = Scenario("shock", [Shock("G", 5, delta=5.0)])
        for sc in (None, scenario):
            res = simulate(spec, 200, sc)
            report = res.validate()
            print(f"[{spec.name}/{res.name}] {report.summary()}")
            if not report.ok:
                failures.append(f"{spec.name}/{res.name}: accounting")
        ss = solve_steady_state(spec.compile())
        expected = analytic(spec)
        err = max(abs(float(ss.values[k]) - v) for k, v in expected.items())
        print(f"[{spec.name}] steady-state max abs error vs analytic: {err:.2e}")
        if not (ss.converged and ss.unique and err < 1e-7):
            failures.append(f"{spec.name}: steady state")
    y1 = simulate(sim_model(), 1).data.loc[1, "Y"]
    if abs(y1 - sim_analytic_first_period(sim_model())) > 1e-9:
        failures.append("SIM first period")
    print("FAILURES:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
