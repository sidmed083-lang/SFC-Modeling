# SFC-Modeling

Open-source platform for building, solving, simulating and analysing **stock-flow consistent (SFC)**
macroeconomic models in Python.

Every simulated period is checked against the accounting identities of the model's
transactions-flow matrix (TFM) and balance-sheet matrix (BSM). Consistency is **evaluated and
reported, never assumed**: if an equation or template breaks an identity, the validator says where.

```bash
streamlit run app.py      # launch the dashboard
```

## Contents
- [Motivation](#motivation) · [Features](#features) · [Installation](#installation)
- [Quick start](#quick-start) · [Methodology and conventions](#methodology-and-conventions)
- [Model construction](#model-construction) · [TFM and BSM](#tfm-and-bsm-examples)
- [Simulation and scenarios](#simulation-and-scenarios) · [Financial networks](#financial-networks)
- [Validation and diagnostics](#accounting-validation-and-diagnostics) · [Dashboard](#dashboard)
- [Testing](#testing) · [Limitations](#known-limitations-and-future-improvements) · [License and citation](#license-and-citation)

## Motivation
SFC models (Godley & Lavoie 2007) force every flow to have a source and a use and every financial
asset to have a matching liability. That discipline makes it hard to build a model that "leaks".
Most implementations are one-off scripts. SFC-Modeling provides the reusable parts: an accounting
engine, a model specification language, a block-recursive solver, scenario tooling, network analysis
and a dashboard, all reproducible and tested against analytically known solutions.

## Features
| Area | What is implemented |
|---|---|
| Sectors and instruments | `Sector` / `Instrument` classification (households, non-financial corporations, financial corporations/banks, government, central bank, foreign; financial and tangible instruments) |
| Accounting | TFM and BSM classes with explicit sign conventions; saving, net lending, net financial wealth; opening/closing stocks; revaluation (capital gains); net acquisition of non-financial assets |
| Validation | Row/column sums, assets = liabilities, aggregate net worth = tangible assets, net lending = financial acquisition, stock-flow reconciliation (`close = open + flows + revaluation`) |
| Model language | `Y = C + G`, lags `X(-1)`, nonlinear functions, JSON round-trip, redundant-identity checks |
| Solver | SymPy parsing → dependency graph → strongly connected components → recursive blocks evaluated directly, simultaneous blocks solved with SciPy `root` (`hybr`, fallback `lm`) |
| Simulation | Dynamic simulation, baseline and scenarios (value / delta / % shocks, temporary or permanent, parameters or exogenous), steady state, local stability (eigenvalues), one-at-a-time sensitivity, seeded Monte Carlo |
| Networks | NetworkX weighted digraphs of payment flows, financing flows and financial claims; strengths, PageRank, shock propagation |
| Dashboard | Streamlit pages: Overview, Sector Setup, Accounting Matrices, Equation Editor, Simulation, Financial Networks, Results, Diagnostics, Export |
| Reproducibility | Models and scenarios serialise to JSON; seeds recorded; exports include a ZIP bundle with spec, data, matrices, validation and report |

## Installation
Python ≥ 3.10.
```bash
git clone https://github.com/<your-user>/SFC-Modeling.git
cd SFC-Modeling
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dashboard,dev]"                      # core + Streamlit/Plotly + test tools
```
Core only (no dashboard): `pip install -e .`

## Quick start
```python
from sfc_modeling.library import sim_model
from sfc_modeling.simulation import simulate, Scenario, Shock

spec = sim_model()                                   # Godley & Lavoie, Model SIM
base = simulate(spec, periods=60)
print(base.data[["Y", "Hh"]].iloc[[1, 60]])          # Y_1 = 38.46..., Y_60 -> 100
print(base.validate().summary())                     # every identity, every period

shock = simulate(spec, 60, Scenario("G up", [Shock("G", start=10, delta=5.0)]))
```
More in [`examples/`](examples) (`01_sim_model.py`, `02_pc_policy_scenarios.py`, `03_custom_model.py`).

## Methodology and conventions
Full details in [`docs/methodology.md`](docs/methodology.md). In short:

* **TFM**: rows = transactions, columns = sectors. `+` receipt / source of funds, `−` payment /
  use of funds. Row kinds: `current`, `capital`, `financial`. Financial rows: acquiring an asset is
  `−`, issuing a liability is `+`. Every row and every column sums to zero.
* **BSM**: `+` asset, `−` liability, by instrument and sector. Financial instruments sum to zero;
  tangible assets do not. The balancing item is net worth (shown with the opposite sign so columns
  sum to zero).
* **Stock-flow link**: `BSM_close = BSM_open − TFM_financial + revaluation`;
  `ΔNW = net lending + net non-financial acquisition + revaluation`.
* **Timing**: period `t` is solved given period `t−1`; `X(-k)` is `k` periods back; lags before
  period 0 reuse the initial values.
* **Tolerances**: a check passes if `|actual − expected| ≤ atol + rtol·scale` (defaults
  `1e-8`, `1e-9`); the report states the tolerances used.

## Model construction
A model is a `ModelSpecification`: equations (`variable = expression`), parameters, exogenous
inputs (constant, or a list indexed by period), initial values, redundant-identity checks, and an
optional accounting specification.

```python
from sfc_modeling.model import Equation, ModelSpecification

spec = ModelSpecification(
    name="demo",
    equations=[
        Equation.from_string("Y = C + G", "identity"),
        Equation.from_string("C = alpha1*YD + alpha2*H(-1)"),
        Equation.from_string("YD = Y - T", "identity"),
        Equation.from_string("T = theta*Y"),
        Equation.from_string("H = H(-1) + YD - C", "identity"),
    ],
    parameters={"alpha1": 0.6, "alpha2": 0.4, "theta": 0.2},
    exogenous={"G": 20.0},
    initial={"H": 0.0},
)
model = spec.compile()
print(model.dependency_table())     # blocks, simultaneous or recursive, lags
```
Rules: one equation per endogenous variable; names are ASCII, start with a letter and contain no
double underscore; functions `exp log sqrt Abs Max Min tanh`; no leads. Compilation reports every
problem at once. Circular dependencies are detected automatically and solved simultaneously.

> **Security note.** Equations are parsed by SymPy's `parse_expr` after a character/identifier
> whitelist. That is defence in depth, not a sandbox. Do not expose the equation editor to untrusted
> users on the public internet.

## TFM and BSM examples
Accounting templates are expressions evaluated every period (see `sfc_modeling/library.py`):
```python
TransactionSpec("Consumption", "current", {"Households": "-C", "Production": "C"})
TransactionSpec("Money", "financial", {"Households": "-(Hh - Hh(-1))", "Government": "Hs - Hs(-1)"})
InstrumentSpec("Money", "financial", {"Households": "Hh", "Government": "-Hs"})
```
TFM financial rows must be named after their BSM instrument. Output for SIM at `t = 60`:
```
                         Households  Production  Government    Σ
Consumption                -79.9968     79.9968      0.0000  0.0
Government expenditure       0.0000     20.0000    -20.0000  0.0
Income (wages)              99.9968    -99.9968      0.0000  0.0
Taxes                      -19.9994      0.0000     19.9994  0.0
Money                       -0.0006      0.0000      0.0006  0.0
Σ                            0.0000      0.0000      0.0000  0.0
```

## Simulation and scenarios
```python
from sfc_modeling.simulation import simulate, Scenario, Shock, compare, one_at_a_time, monte_carlo, final_value
from sfc_modeling.solver import solve_steady_state, stability_analysis

runs = {"Baseline": simulate(spec, 80),
        "Tax cut": simulate(spec, 80, Scenario("Tax cut", [Shock("theta", 10, value=0.15)]))}
compare(runs, ["Y"], reference="Baseline")                 # differences vs baseline
solve_steady_state(spec.compile())                          # static equilibrium (+ uniqueness check)
one_at_a_time(spec, "theta", [0.1, 0.2, 0.3], final_value("Y"), periods=300)
monte_carlo(spec, {"alpha1": ("uniform", 0.5, 0.7)}, final_value("Hh"), periods=100, n=200, seed=1)
```
A `Shock` takes exactly one of `value`, `delta`, `pct`, applied from `start` to `end` (default
permanent). Static steady states use the model's redundant identities as extra equations; the
result reports Jacobian rank and flags non-uniqueness.

## Financial networks
```python
from sfc_modeling.networks import flow_network, exposure_network, node_summary, propagate_shock

tfm, bsm, _, _ = base.matrices(60)
flows = flow_network(tfm)            # kind="flow": who pays whom this period
claims = exposure_network(bsm)       # kind="stock": creditor -> debtor claims
propagate_shock(claims, "Government", 10.0, direction="upstream")
```
Flows and stocks are never mixed (`graph.graph["kind"]`). Multi-party rows are allocated **pro rata**,
a convention (the TFM/BSM do not record bilateral links).

## Accounting validation and diagnostics
```python
report = base.validate()          # all periods
report.ok, report.summary()
report.to_frame()                 # check, location (period/sector/instrument), expected, actual, difference
base.consistency_residuals()      # redundant identities such as Hs - Hh
base.diagnostics                  # per block, per period: method, success, evaluations, residual
```
The test-suite deliberately breaks models (missing counterparts, mis-signed or omitted revaluations,
stock-flow-inconsistent equations) and asserts that validation fails.

## Dashboard
`streamlit run app.py`. Pick an example (SIM, PC, BOND-REVAL) or upload a model JSON in the sidebar,
edit sectors/equations/matrix templates, run scenarios, inspect networks and diagnostics, and download
CSV / JSON / Markdown / ZIP exports.

## Testing
```bash
pytest -q                          # unit, accounting, benchmark and scenario tests
python examples/validate_examples.py   # accounting identities + analytic benchmarks (used in CI)
black --check . && ruff check .    # formatting and lint
```
Benchmarks with known solutions: SIM first-period output `G/(1−α₁(1−θ)) = 38.46`, SIM and PC
closed-form steady states, SIM stability eigenvalue `1 − α₂θ/(1−α₁(1−θ))`, a nonlinear simultaneous
system (`x = 1+√y, y = 2x+1 ⇒ x=4, y=9`) and an AR(1) closed form. See [`docs/QA_REPORT.md`](docs/QA_REPORT.md).

## Known limitations and future improvements
* No forward-looking (model-consistent) expectations: leads `X(+1)` are rejected.
* Equations must be explicit (`variable = expression`); implicit equations are not supported.
* Simultaneous blocks use SciPy's finite-difference Jacobians; analytic (SymPy) Jacobians would help
  large models.
* Steady states of stock-accumulation models need their redundant identity supplied as a consistency check.
* Network edges are pro-rata allocations; `propagate_shock` is an exposure illustration, not a behavioural contagion model.
* The dashboard is exercised by a scripted smoke run but has no automated UI tests yet
  (candidate: `streamlit.testing.v1.AppTest`).
* Bundled models are textbook examples (SIM, PC) plus a capital-gains demo; no calibration to data,
  and no foreign-sector or bank-credit reference model yet (the sector classification supports them).
* Time-varying parameters exist only through scenarios.

## License and citation
MIT, see [LICENSE](LICENSE). Citation metadata in [CITATION.cff](CITATION.cff). Please also cite:

> Godley, W. and Lavoie, M. (2007). *Monetary Economics: An Integrated Approach to Credit, Money,
> Income, Production and Wealth.* Palgrave Macmillan.
