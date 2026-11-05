"""Write the bundled models to JSON (reproducible, shareable configurations)."""

from pathlib import Path

from sfc_modeling.library import pc_model, sim_model

here = Path(__file__).parent
for spec in (sim_model(), pc_model()):
    (here / f"{spec.name.lower()}_model.json").write_text(spec.to_json(), encoding="utf-8")
    print("wrote", f"{spec.name.lower()}_model.json")
