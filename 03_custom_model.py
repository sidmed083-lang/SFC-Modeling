"""Define a model from scratch: SIM with adaptive income expectations (SIMEX-style).

Consumption depends on *expected* disposable income, YDe = YD(-1).  The
accounting templates are reused from SIM, and validated every period.
"""

from sfc_modeling.library import sim_model
from sfc_modeling.model import Equation
from sfc_modeling.simulation import Scenario, Shock, simulate
from sfc_modeling.solver import solve_steady_state

spec = sim_model()
spec.name = "SIM-expectations"
spec.equations = [
    Equation.from_string("Y = C + G", "identity"),
    Equation.from_string("YD = Y - T", "identity"),
    Equation.from_string("T = theta*Y"),
    Equation.from_string("YDe = YD(-1)"),
    Equation.from_string("C = alpha1*YDe + alpha2*Hh(-1)"),
    Equation.from_string("N = Y/W"),
    Equation.from_string("Hh = Hh(-1) + YD - C", "identity"),
    Equation.from_string("Hs = Hs(-1) + G - T", "identity"),
]
model = spec.compile()
print(model.dependency_table().to_string(index=False))  # no simultaneous block now

result = simulate(model, 80, Scenario("G +5", [Shock("G", 20, delta=5.0)]))
print(result.validate().summary())
print("Steady state:", solve_steady_state(model).values.round(4).to_dict())
print(result.data[["Y", "Hh"]].iloc[[19, 20, 21, 40, 80]].round(3))
