# Quality-assurance report (v0.1.0)

This report states what was **actually run** in the development environment and what was **not**.

## Environment
Python 3.12.3, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, SymPy 1.14.0, NetworkX 3.7.
The environment had **no network access**: pytest, Black, Ruff, mypy, Streamlit and Plotly were not available.

## What was run
| Check | Result |
|---|---|
| Test suite: 55 tests (accounting, expressions, model compilation, benchmarks, scenarios, validation-failure detection, sensitivity/Monte Carlo, networks, exports, revaluation) | **55 passed, 0 failed**, executed with a minimal pytest-compatible runner (supports `raises`, `approx`, plain asserts; no fixtures are used in the tests) |
| `examples/validate_examples.py` (both reference models, baseline and shocked) | passed, no failures |
| Package install (`pip install -e . --no-deps`) and import | passed |
| Dashboard: all 8 pages + Overview executed end-to-end for SIM and PC with a *fake* `streamlit`/`plotly` (all buttons "pressed", all radio options, all downloads non-empty) | 0 exceptions, 0 `st.error` calls |
| Static check: unused imports | none (one intentional `tabulate` availability probe) |
| Examples 01-03 | ran to completion |
| README code snippets | executed; output as documented |

## Accounting results
| Model | Periods | Accounting checks | Issues |
|---|---|---|---|
| SIM, baseline and with shock | 200 | 5 006 each | 0 |
| PC, baseline and with shock | 200 | 7 808 each | 0 |
| BOND-REVAL (capital gains) | 20 | validated in tests | 0; omitting or mis-signing the revaluation is detected |
Redundant identities (`Hs − Hh`, `Hs − Bcb`) stay below 5e-13 in absolute value.

## Benchmark solutions
| Benchmark | Expected | Obtained |
|---|---|---|
| SIM `Y₁` | `G/(1−α₁(1−θ)) = 38.4615…` | matches to 1e-8 relative |
| SIM dynamic path | closed-form recursion, 40 periods | matches to 1e-9 |
| SIM steady state (solver / long run) | `Y=100, Hh=80` | error 1.4e-14 / < 1e-8 |
| PC steady state (closed form, 10 variables) | derivation in `docs/methodology.md` | solver error 4.3e-14; long-run simulation < 1e-6 |
| SIM stability eigenvalue | `1 − α₂θ/(1−α₁(1−θ)) = 0.846154` | matches; one unit root from `Hs` |
| Nonlinear simultaneous block | `x = 4, y = 9` | matches to 1e-9 |
| AR(1) with lag | `2(1 − 0.5ᵗ)` | matches to 1e-12 |

## Defects found and fixed during development
1. TFM financial rows must carry the BSM instrument name; a naming mismatch in the SIM template was
   caught by the validator itself (now documented).
2. Lagged variables used only in accounting templates (not in equations) were missing from the
   evaluation context (found by PC); fixed.
3. Steady-state solver divided by zero from a zero initial stock (fixed); its rank diagnostic used a noisy
   Jacobian and missed a genuine non-uniqueness (fixed with central differences). The missing closure
   (`Hs = Bcb`, the central bank has no net worth) is now part of PC.

## NOT verified (please do before relying on the project)
- **Real pytest, Black, Ruff, mypy**: not run. The code is not guaranteed Black-formatted; CI's
  `black --check` will probably fail on the first push until `black .` is run (see `PUBLISHING.md`).
  mypy is configured as advisory.
- **Streamlit and Plotly behaviour**: pages ran against fakes, so my own logic is exercised but
  real widget behaviour, rendering, deprecations (e.g. `use_container_width` in newest Streamlit) and
  layout are untested. No automated UI tests exist.
- **Python 3.10 / 3.11** and the declared minimum dependency versions: untested (CI matrix will exercise them).
- **Test coverage** was not measured.
- **Scale**: performance on large models (hundreds of equations) and robustness of the numerical
  solver on stiff/non-convex systems were not tested.
- Economic content is limited to textbook benchmarks; no empirical calibration or peer review.
