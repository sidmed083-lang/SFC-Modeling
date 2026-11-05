"""Reference models from Godley & Lavoie (2007), *Monetary Economics*.

Both models carry full accounting templates (TFM financial rows are named after
their BSM instrument, which is how flows are matched to stocks), so every simulated period is
checked against the TFM / BSM identities.

* :func:`sim_model` - Model SIM (Ch. 3): households, production, government,
  one asset (money).  Analytic steady state: ``Y* = G/theta``.
* :func:`pc_model`  - Model PC (Ch. 4): adds bills, a central bank and a
  portfolio choice.
"""

from __future__ import annotations

from .accounting.spec import AccountingSpec, InstrumentSpec, TransactionSpec
from .model.equations import IDENTITY, Equation
from .model.specification import ModelSpecification


def sim_model() -> ModelSpecification:
    eq = Equation.from_string
    equations = [
        eq("Y = C + G", IDENTITY, "Output = consumption + government spending"),
        eq("YD = Y - T", IDENTITY, "Disposable income"),
        eq("T = theta*Y", description="Taxes (wages = output)"),
        eq("C = alpha1*YD + alpha2*Hh(-1)", description="Consumption out of income and wealth"),
        eq("N = Y/W", description="Employment"),
        eq("Hh = Hh(-1) + YD - C", IDENTITY, "Households' money holdings"),
        eq("Hs = Hs(-1) + G - T", IDENTITY, "Money supplied by government deficits"),
    ]
    sectors = ["Households", "Production", "Government"]
    accounting = AccountingSpec(
        sectors=sectors,
        transactions=[
            TransactionSpec("Consumption", "current", {"Households": "-C", "Production": "C"}),
            TransactionSpec("Government expenditure", "current", {"Government": "-G", "Production": "G"}),
            TransactionSpec("Income (wages)", "current", {"Households": "Y", "Production": "-Y"}),
            TransactionSpec("Taxes", "current", {"Households": "-T", "Government": "T"}),
            TransactionSpec("Money", "financial",
                            {"Households": "-(Hh - Hh(-1))", "Government": "Hs - Hs(-1)"}),
        ],
        instruments=[InstrumentSpec("Money", "financial", {"Households": "Hh", "Government": "-Hs"})],
    )
    return ModelSpecification(
        name="SIM",
        description="Godley & Lavoie Model SIM: simplest SFC model with government money.",
        equations=equations,
        parameters={"alpha1": 0.6, "alpha2": 0.4, "theta": 0.2},
        exogenous={"G": 20.0, "W": 1.0},
        initial={"Hh": 0.0, "Hs": 0.0},
        consistency_checks=["Hs - Hh"],
        accounting=accounting,
        descriptions={"alpha1": "Propensity to consume out of income",
                      "alpha2": "Propensity to consume out of wealth",
                      "theta": "Tax rate", "G": "Government expenditure", "W": "Wage rate"},
        random_seed=12345,
    )


def pc_model() -> ModelSpecification:
    eq = Equation.from_string
    equations = [
        eq("Y = C + G", IDENTITY),
        eq("YD = Y - T + r(-1)*Bh(-1)", IDENTITY, "Disposable income incl. interest"),
        eq("T = theta*(Y + r(-1)*Bh(-1))", description="Income tax"),
        eq("V = V(-1) + YD - C", IDENTITY, "Household wealth"),
        eq("C = alpha1*YD + alpha2*V(-1)"),
        eq("Hh = V - Bh", IDENTITY, "Money = wealth not held in bills"),
        eq("Bh = V*(lambda0 + lambda1*r - lambda2*YD/V)", description="Bill demand (portfolio choice)"),
        eq("Bs = Bs(-1) + G + r(-1)*Bs(-1) - T - r(-1)*Bcb(-1)", IDENTITY, "Government borrowing"),
        eq("Bcb = Bs - Bh", IDENTITY, "Central bank takes up the residual"),
        eq("Hs = Hs(-1) + Bcb - Bcb(-1)", IDENTITY, "Money issued against bills"),
    ]
    sectors = ["Households", "Production", "Government", "CentralBank"]
    accounting = AccountingSpec(
        sectors=sectors,
        transactions=[
            TransactionSpec("Consumption", "current", {"Households": "-C", "Production": "C"}),
            TransactionSpec("Government expenditure", "current", {"Government": "-G", "Production": "G"}),
            TransactionSpec("Income (wages)", "current", {"Households": "Y", "Production": "-Y"}),
            TransactionSpec("Interest on bills", "current", {
                "Households": "r(-1)*Bh(-1)", "Government": "-r(-1)*Bs(-1)",
                "CentralBank": "r(-1)*Bcb(-1)"}),
            TransactionSpec("Central bank profits", "current",
                            {"Government": "r(-1)*Bcb(-1)", "CentralBank": "-r(-1)*Bcb(-1)"}),
            TransactionSpec("Taxes", "current", {"Households": "-T", "Government": "T"}),
            TransactionSpec("Money", "financial",
                            {"Households": "-(Hh - Hh(-1))", "CentralBank": "Hs - Hs(-1)"}),
            TransactionSpec("Bills", "financial", {
                "Households": "-(Bh - Bh(-1))", "Government": "Bs - Bs(-1)",
                "CentralBank": "-(Bcb - Bcb(-1))"}),
        ],
        instruments=[
            InstrumentSpec("Money", "financial", {"Households": "Hh", "CentralBank": "-Hs"}),
            InstrumentSpec("Bills", "financial", {"Households": "Bh", "Government": "-Bs", "CentralBank": "Bcb"}),
        ],
    )
    return ModelSpecification(
        name="PC",
        description="Godley & Lavoie Model PC: portfolio choice between money and bills.",
        equations=equations,
        parameters={"alpha1": 0.6, "alpha2": 0.4, "theta": 0.2,
                    "lambda0": 0.635, "lambda1": 0.05, "lambda2": 0.01},
        exogenous={"G": 20.0, "r": 0.025},
        initial={"V": 0.0, "Bh": 0.0, "Bs": 0.0, "Bcb": 0.0, "Hs": 0.0, "Hh": 0.0},
        consistency_checks=["Hs - Hh", "Hs - Bcb"],
        accounting=accounting,
        random_seed=12345,
    )


def sim_analytic_steady_state(spec: ModelSpecification) -> dict[str, float]:
    p, G = spec.parameters, spec.exogenous["G"]
    Y = G / p["theta"]
    YD = (1 - p["theta"]) * Y
    H = YD * (1 - p["alpha1"]) / p["alpha2"]
    return {"Y": Y, "YD": YD, "T": p["theta"] * Y, "C": YD, "Hh": H, "Hs": H}


def sim_analytic_first_period(spec: ModelSpecification) -> float:
    """Y_1 with zero initial money: ``G / (1 - alpha1 (1 - theta))``."""
    p = spec.parameters
    return spec.exogenous["G"] / (1 - p["alpha1"] * (1 - p["theta"]))


def pc_analytic_steady_state(spec: ModelSpecification) -> dict[str, float]:
    """Closed-form steady state of PC (derivation in docs/methodology.md)."""
    p, G, r = spec.parameters, spec.exogenous["G"], spec.exogenous["r"]
    th, a1, a2 = p["theta"], p["alpha1"], p["alpha2"]
    k = (1 - th) * (1 - a1) / a2  # V = k * X, X = Y + r*Bh
    b = k * (p["lambda0"] + p["lambda1"] * r) - p["lambda2"] * (1 - th)  # Bh = b * X
    X = G / (th - r * b)
    Bh = b * X
    V = k * X
    # Central bank has no net worth: Hs = Bcb = Hh = V - Bh, and Bs = Bh + Bcb = V.
    return {"Y": X - r * Bh, "YD": (1 - th) * X, "T": th * X, "V": V, "Bh": Bh,
            "Hh": V - Bh, "Hs": V - Bh, "Bcb": V - Bh, "Bs": V}


def revaluation_model() -> ModelSpecification:
    """SIM-like model where government bonds trade at an exogenous price ``pb``.

    Not from Godley & Lavoie: a minimal demonstration of capital gains.  Bond
    holdings are revalued every period (``Bq(-1) * dpb``): households gain what
    the government loses, so financial assets still equal liabilities and
    ``dNW = net lending + revaluation`` holds for every sector.
    """
    eq = Equation.from_string
    equations = [
        eq("Y = C + G", IDENTITY),
        eq("YD = Y - T", IDENTITY),
        eq("T = theta*Y"),
        eq("C = c*YD"),
        eq("Bq = Bq(-1) + (G - T)/pb", IDENTITY, "Bond quantity issued to fund the deficit"),
        eq("Bv = Bq*pb", IDENTITY, "Market value of bonds"),
    ]
    sectors = ["Households", "Production", "Government"]
    gain = "Bq(-1)*(pb - pb(-1))"
    accounting = AccountingSpec(
        sectors=sectors,
        transactions=[
            TransactionSpec("Consumption", "current", {"Households": "-C", "Production": "C"}),
            TransactionSpec("Government expenditure", "current", {"Government": "-G", "Production": "G"}),
            TransactionSpec("Income (wages)", "current", {"Households": "Y", "Production": "-Y"}),
            TransactionSpec("Taxes", "current", {"Households": "-T", "Government": "T"}),
            TransactionSpec("Bonds", "financial", {
                "Households": "-(Bq - Bq(-1))*pb", "Government": "(Bq - Bq(-1))*pb"}),
        ],
        instruments=[InstrumentSpec("Bonds", "financial", {"Households": "Bq*pb", "Government": "-Bq*pb"})],
        revaluations={"Bonds": {"Households": gain, "Government": "-" + gain}},
    )
    return ModelSpecification(
        name="BOND-REVAL",
        description="Model with market-priced government bonds and capital gains.",
        equations=equations,
        parameters={"theta": 0.2, "c": 0.8},
        exogenous={"G": 20.0, "pb": [1.0, 1.0, 1.0, 1.1, 1.0, 0.9, 1.0]},
        initial={"Bq": 10.0},
        accounting=accounting,
        random_seed=12345,
    )
