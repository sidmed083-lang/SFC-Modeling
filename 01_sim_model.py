"""Model SIM: baseline run, closed-form check and accounting validation."""

from sfc_modeling.library import sim_analytic_steady_state, sim_model
from sfc_modeling.simulation import simulate
from sfc_modeling.solver import solve_steady_state

spec = sim_model()
result = simulate(spec, periods=60)

print(result.data[["Y", "YD", "C", "Hh", "Hs"]].iloc[[0, 1, 2, 10, 60]].round(4))
print("\nSteady state (analytic):", sim_analytic_steady_state(spec))
print("Steady state (solver):  ", solve_steady_state(spec.compile()).values.round(6).to_dict())

tfm, bsm, _, _ = result.matrices(60)
print("\nTransactions-flow matrix, t=60\n", tfm.to_frame().round(4))
print("\nBalance-sheet matrix, t=60\n", bsm.to_frame().round(4))
print("\n", result.validate().summary())
