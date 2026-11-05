# Methodology

## 1. Accounting conventions
**Sectors** are columns, **transactions / instruments** are rows.

### Transactions-flow matrix
| Row kind | Meaning | Constraint |
|---|---|---|
| `current` | income and outlay | row sum = 0; column sum = saving |
| `capital` | capital transfers | row sum = 0; current+capital column sum = net lending |
| `financial` | change in a financial instrument | row sum = 0; column sum of *all* rows = 0 |

Signs: `+` receipt/source, `−` payment/use. Financial rows: asset acquisition `−`, liability issuance `+`.
Hence **net lending = − Σ financial rows** per sector, and Σ sectors of net lending = 0.

### Balance-sheet matrix
`+` asset, `−` liability. Financial instrument rows sum to zero. Tangible rows have no counterpart, so
`Σ_s NW_s = Σ tangible assets`. Net worth `NW_s = Σ_i BSM[i, s]`; the displayed balancing item is `−NW_s`.

### Stock-flow reconciliation
For a financial instrument `i` and sector `s`:
`BSM_t[i,s] = BSM_{t−1}[i,s] − TFM_t^{fin}[i,s] + Reval_t[i,s]`.
For tangible assets: `ΔK_s = NFA_s + Reval_s`.
For net worth: `ΔNW_s = NL_s + NFA_s + Reval_s`, where `NFA` is net acquisition of non-financial
assets at cost. Revaluations of a financial instrument must net to zero across sectors
(a gain for the holder is a loss for the issuer).

### Tolerances
`|actual − expected| ≤ atol + rtol·scale`, `scale` = magnitude of the quantities in the check
(e.g. the sum of absolute entries of a row). Defaults `atol = 1e-8`, `rtol = 1e-9`.
NaN always fails.

## 2. Model equations and timing
* Equation form `v = f(...)`. Parameters are constant within a scenario; exogenous inputs are
  constants or per-period lists; scenarios turn both into time paths.
* Period 0 holds initial conditions (unlisted endogenous variables start at 0). Periods `1..T` are solved.
  `X(-k)` at period `t` refers to period `max(t−k, 0)`.
* **Block solution.** Build the contemporaneous dependency graph among endogenous variables, take
  strongly connected components, order the condensation topologically. Single-variable acyclic blocks
  are evaluated directly; others are solved by `scipy.optimize.root` (`hybr`, then `lm`) with the
  previous period as initial guess. A block fails if `max|residual| > tol·max(1,|x|)`; the default
  behaviour is to raise `ConvergenceError`.
* **Redundant equation.** An SFC model with `n` sectors has one redundant stock-flow relation
  (Walras' law). Models list it as a *consistency check* (e.g. `Hs - Hh`). It is evaluated each
  period and used to close the static steady-state problem.

## 3. Steady state
Setting every lag equal to the current value turns accumulation equations into flow conditions
(`H = H(-1) + YD − C` becomes `YD = C`) which no longer fix the stock. The solver stacks the
equation residuals and the consistency checks and applies Levenberg–Marquardt
(`scipy.optimize.least_squares`). The Jacobian is re-estimated by central differences and its rank
is reported; `unique = (rank == number of unknowns)`.

## 4. Stability
The state vector is the set of endogenous variables that appear lagged. The Jacobian of
`x_t = F(x_{t−1})` at the steady state is computed numerically and its eigenvalues returned.
Eigenvalues equal to one that come from accounting-redundant stocks are counted as unit roots and
excluded from the stability verdict.

## 5. Benchmarks
### Model SIM
`Y = C + G, YD = Y − T, T = θY, C = α₁YD + α₂H₋₁, H = H₋₁ + YD − C, Hs = Hs₋₁ + G − T`.
* First period from `H₀ = 0`: `Y₁ = G / (1 − α₁(1 − θ))`.
* Steady state: `Y* = G/θ`, `YD* = (1−θ)Y*`, `H* = YD*(1 − α₁)/α₂`.
* Persistence: `λ = 1 − α₂θ / (1 − α₁(1−θ))` (plus a unit root from `Hs`).

### Model PC
Households hold money `H` and bills `B` with `B = V(λ₀ + λ₁r − λ₂ YD/V)`.
In steady state (`ΔV = 0`, `ΔBs = 0`), with `X = Y + rB_h`:
`YD = (1−θ)X`, `C = YD ⇒ V = kX` with `k = (1−θ)(1−α₁)/α₂`;
`B_h = bX` with `b = k(λ₀ + λ₁ r) − λ₂(1−θ)`; government budget `G + rB_h = θX`
⇒ `X = G/(θ − r b)`, `Y = X − rB_h`. The central bank has no net worth: `Hs = Bcb = V − B_h`, `Bs = V`.

## 6. Networks
*Flow network*: for a row with payers `p` (negative entries) and receivers `q` (positive entries),
`w(p→q) = |x_p|·x_q / Σ_q x_q`. *Exposure network*: creditors (positive BSM entries) to debtors
(negative entries) allocated the same way. These are conventions; actual bilateral links require
additional data.

## 7. Reference
Godley, W. & Lavoie, M. (2007), *Monetary Economics*, Palgrave Macmillan, Ch. 3–4.
