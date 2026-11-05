"""Model PC: compare a fiscal expansion with an interest-rate shock."""

from sfc_modeling.analysis import net_lending
from sfc_modeling.library import pc_model
from sfc_modeling.networks import exposure_network, node_summary
from sfc_modeling.simulation import Scenario, Shock, compare, simulate

spec = pc_model()
T = 80
runs = {
    "Baseline": simulate(spec, T),
    "G +10": simulate(spec, T, Scenario("G +10", [Shock("G", 10, delta=10.0)])),
    "r 2.5%->5%": simulate(spec, T, Scenario("r 2.5%->5%", [Shock("r", 10, value=0.05)])),
}
for name, res in runs.items():
    print(name, "->", res.validate().summary())

print(compare(runs, ["Y", "Bs"], reference="Baseline").iloc[[10, 20, T]].round(3))
print("\nNet lending (G +10, last period):\n", net_lending(runs["G +10"]).iloc[-1].round(4))

_, bsm, _, _ = runs["G +10"].matrices(T)
print("\nClaims network:\n", node_summary(exposure_network(bsm)).round(3))
