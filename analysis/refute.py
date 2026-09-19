"""
Stage-A refutation battery (DoWhy) — pre-registered.

Runs the seven DoWhy refutations after the causal estimate is produced, scores
each against a *fixed-in-advance* pass criterion, and writes a tidy report. The
thresholds below are committed BEFORE the real estimate exists so the refutations
can't be reverse-engineered into a pass.

The seven tests (DoWhy method_name in parentheses) and what each attacks in our DAG
(see analysis/dag.html):

  1. Add Random Common Cause  (random_common_cause)
       Inject an independent random covariate into the adjustment set. A correct
       estimator should be UNCHANGED. Guards against estimator/covariate-set
       misspecification.
  2. Placebo Treatment        (placebo_treatment_refuter, placebo_type="permute")
       Permute HDI across users. Effect should collapse to ~0. Guards against the
       estimator manufacturing signal from noise.
  3. Dummy Outcome            (dummy_outcome_refuter, default zero+noise DGP)
       Replace remoteness with an outcome whose TRUE effect is 0. Recovered effect
       should be ~0. Outcome-side placebo.
  4. Simulated Outcome        (dummy_outcome_refuter, known linear DGP + injected effect)
       Replace remoteness with f(confounders) + beta*HDI for KNOWN beta. Recovered
       effect should MATCH beta. Confirms the estimator recovers a known truth.
  5. Add Unobserved Common Cause (add_unobserved_common_cause, linear simulation grid)
       Dial in an unobserved confounder of increasing strength on both HDI and
       remoteness; the estimate should not be too sensitive (sign preserved). This is
       the unmeasured-confounding / Flickr-selection-collider stress test.
       NOTE: dowhy 0.14's partial-R2 (Cinelli-Hazlett RV) path is buggy and rejects
       effect modifiers, so we use the classic linear grid here and compute the
       closed-form Robustness Value separately (see robustness_value()).
  6. Data Subset Validation   (data_subset_refuter, subset_fraction=0.8)
       Re-estimate on random subsets; estimate should be stable. Finite-sample
       fragility / influential-subsample check.
  7. Bootstrap Validation     (bootstrap_refuter)
       Re-estimate on bootstrap resamples; estimate should be stable and the point
       estimate should sit inside the bootstrap spread.

Usage (from Stage A, after you have model / identified_estimand / estimate):

    from analysis.refute import run_all_refutations, RefuteConfig
    report = run_all_refutations(model, identified_estimand, estimate,
                                 out_dir="analysis/outputs")

Self-test on synthetic data (no warehouse needed):

    uv run python analysis/refute.py --selftest
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
logging.getLogger("dowhy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from dowhy import CausalModel


# --------------------------------------------------------------------------- #
# Pre-registered configuration (thresholds fixed before the real estimate)
# --------------------------------------------------------------------------- #
@dataclass
class RefuteConfig:
    num_simulations: int = 100          # bootstrap/subset/random-cause/placebo sims
    random_seed: int = 20240608

    # PASS thresholds -------------------------------------------------------- #
    # "should not change" refuters: relative change in the point estimate
    rel_change_random_cause: float = 0.10
    rel_change_subset: float = 0.15
    rel_change_bootstrap: float = 0.15
    # "should go to zero" refuters. placebo: |new effect| as a fraction of
    # |original| (same outcome scale, ratio is coherent). dummy: STANDARDIZED
    # effect in sd-units (|beta|*sd(T)/sd(dummy_y)) because the dummy outcome
    # is unit-variance noise unrelated to the real outcome's scale.
    placebo_zero_frac: float = 0.15
    dummy_zero_frac: float = 0.15
    # "should match the injected truth" refuter
    simulated_match_tol: float = 0.20   # |recovered - injected| / |injected|
    simulated_injected_effect: Optional[float] = None  # None -> use |original|

    # Unobserved-confounder linear grid. None -> let dowhy auto-calibrate the
    # strengths from the observed covariates (recommended). Override in Stage A
    # to bracket the strength of `region` / `distance` explicitly.
    unobserved_effect_strength_treatment: Optional[list] = None
    unobserved_effect_strength_outcome: Optional[list] = None

    # Estimator used to RE-estimate on the simulated-outcome dataset (#4). Default
    # mirrors the linear baseline; set to the Stage-A DML method to test that estimator.
    reestimate_method_name: str = "backdoor.linear_regression"
    reestimate_method_params: Optional[dict] = None


@dataclass
class RefuterResult:
    name: str
    method: str
    hint: str
    original_effect: float
    observed: float          # new effect (or worst-case across a grid/list)
    statistic: str           # human-readable comparison
    passed: bool
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Model construction helpers (reusable by Stage A)
# --------------------------------------------------------------------------- #
def make_causal_model(df: pd.DataFrame, treatment: str, outcome: str,
                      common_causes: list[str],
                      effect_modifiers: Optional[list[str]] = None) -> CausalModel:
    """Build a DoWhy CausalModel from explicit confounders / effect modifiers.

    Passing common_causes/effect_modifiers lets DoWhy assemble the graph; this is
    less error-prone than hand-writing GML and matches the adjustment set in the DAG.
    """
    return CausalModel(
        data=df, treatment=treatment, outcome=outcome,
        common_causes=common_causes,
        effect_modifiers=effect_modifiers or [],
    )


def estimate_linear(model: CausalModel, identified_estimand):
    """Baseline backdoor linear-regression estimate (refuters are estimator-agnostic,
    but the simulated/dummy outcome DGPs require a covariate-adjustment estimator)."""
    return model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")


def dml_nuisance_models() -> dict:
    """Stage-A DML nuisance models (user decision, 2026-07): both nuisances
    default to econml's 'auto' model selection.

    The treatment stays CONTINUOUS (raw HDI, not binned). With a continuous
    treatment the discrete-treatment DR learners (and any propensity classifier)
    don't apply; the estimator is LinearDML, where model_t is the treatment
    model — the continuous analogue of the propensity — and model_y is the
    outcome model. NOTE: the linear final stage estimates a single slope in
    HDI; the U-shape found in the assumption-audit EDA is handled by a
    binned/spline sensitivity spec, not by the headline."""
    return {"model_y": "auto", "model_t": "auto"}


def estimate_dml(model: CausalModel, identified_estimand, **init_overrides):
    """Debiased-ML estimate via econml's LinearDML (DoWhy-wrapped so the
    refutation battery can re-run it): 'auto' nuisances + linear final stage
    with statsmodels inference. Continuous treatment."""
    init = {**dml_nuisance_models(), "random_state": 20240608}
    init.update(init_overrides)
    return model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        method_params={"init_params": init, "fit_params": {}},
    )


# --------------------------------------------------------------------------- #
# Result parsing
# --------------------------------------------------------------------------- #
def _effects(result) -> list[float]:
    """Flatten a refuter result (CausalRefutation or list thereof) to new-effect floats."""
    items = result if isinstance(result, (list, tuple)) else [result]
    out = []
    for r in items:
        ne = getattr(r, "new_effect", None)
        if ne is None:
            continue
        arr = np.asarray(ne, dtype=float).ravel()
        out.extend([float(x) for x in arr if np.isfinite(x)])
    return out


def _first_effect(result) -> float:
    """First finite new-effect, or NaN when the refuter returned none (a NaN
    re-estimate). NaN fails every threshold comparison, so the test records a
    FAIL row instead of crashing the battery with an IndexError."""
    effs = _effects(result)
    return effs[0] if effs else float("nan")


def _pval(result) -> Optional[float]:
    items = result if isinstance(result, (list, tuple)) else [result]
    for r in items:
        rr = getattr(r, "refutation_result", None)
        if isinstance(rr, dict) and "p_value" in rr:
            try:
                return float(rr["p_value"])
            except Exception:
                return None
    return None


def _rel(new: float, orig: float) -> float:
    return abs(new - orig) / max(abs(orig), 1e-9)


# --------------------------------------------------------------------------- #
# Closed-form Cinelli-Hazlett Robustness Value (version-proof; pooled linear spec)
# --------------------------------------------------------------------------- #
def robustness_value(t_stat: float, dof: int, q: float = 1.0) -> float:
    """RV_q: the minimum partial-R^2 an unobserved confounder must share with BOTH
    treatment and outcome to reduce the |effect| by 100*q% (q=1 -> to zero).
    Cinelli & Hazlett (2020), closed form. Use on the pooled OLS t-stat / dof."""
    f = q * abs(t_stat) / np.sqrt(dof)
    return float(0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2))


def _simulated_outcome_recovery(model: CausalModel, injected: float,
                                config: RefuteConfig) -> float:
    """Build a simulated outcome y = f(W) + beta*(T-meanT) + noise with KNOWN beta,
    then re-run identification + estimation and return the recovered effect."""
    from sklearn.linear_model import LinearRegression

    df = model._data
    treatment = model._treatment[0]
    outcome = model._outcome[0]
    common = list(model.get_common_causes() or [])
    try:
        mods = list(model.get_effect_modifiers() or [])
    except Exception:
        mods = []
    mods = [m for m in mods if m not in common]
    feats = common + mods
    rng = np.random.default_rng(config.random_seed + 99)

    y = df[outcome].to_numpy(float)
    if feats:
        # One-hot encode string/categorical confounders (e.g. origin_region) —
        # DoWhy's own estimators encode internally, but sklearn needs it done here.
        X = pd.get_dummies(df[feats], drop_first=True, dtype=float).to_numpy(float)
        fW = LinearRegression().fit(X, y).predict(X)
    else:
        fW = np.full(len(df), float(y.mean()))
    t = df[treatment].to_numpy(float)
    resid_sd = float(np.std(y - fW)) or 1.0
    y_sim = fW + injected * (t - t.mean()) + rng.normal(0, resid_sd, len(df))

    df2 = df.copy()
    df2[outcome] = y_sim
    m2 = make_causal_model(df2, treatment, outcome, common, mods)
    id2 = m2.identify_effect(proceed_when_unidentifiable=True)
    est2 = m2.estimate_effect(id2, method_name=config.reestimate_method_name,
                              method_params=config.reestimate_method_params)
    return float(est2.value)


# --------------------------------------------------------------------------- #
# The seven refutations
# --------------------------------------------------------------------------- #
def run_all_refutations(model: CausalModel, identified_estimand, estimate,
                        config: Optional[RefuteConfig] = None,
                        out_dir: Optional[str] = None) -> list[RefuterResult]:
    config = config or RefuteConfig()   # fresh default per call, never shared
    np.random.seed(config.random_seed)
    orig = float(estimate.value)
    N = config.num_simulations
    results: list[RefuterResult] = []

    def refute(**kw):
        return model.refute_estimate(identified_estimand, estimate, **kw)

    def stable(name: str, method: str, r, tol: float) -> RefuterResult:
        """Shared shape for the 'estimate should NOT change' refuters (1/6/7)."""
        new = _first_effect(r)
        return RefuterResult(
            name, method, "estimate should NOT change", orig, new,
            f"rel. change {_rel(new, orig):.1%} (<= {tol:.0%})",
            _rel(new, orig) <= tol, {"p_value": _pval(r)})

    # 1. Add Random Common Cause -------------------------------------------- #
    r = refute(method_name="random_common_cause", num_simulations=N)
    results.append(stable("Add Random Common Cause", "random_common_cause",
                          r, config.rel_change_random_cause))

    # 2. Placebo Treatment --------------------------------------------------- #
    r = refute(method_name="placebo_treatment_refuter", placebo_type="permute",
               num_simulations=N)
    new = _first_effect(r)
    frac = abs(new) / max(abs(orig), 1e-9)
    results.append(RefuterResult(
        "Placebo Treatment", "placebo_treatment_refuter",
        "effect should go to ~0", orig, new,
        f"|new|/|orig| {frac:.1%} (<= {config.placebo_zero_frac:.0%})",
        frac <= config.placebo_zero_frac, {"p_value": _pval(r)}))

    # 3. Dummy Outcome (true effect = 0) ------------------------------------ #
    # The dummy outcome is dowhy's zero + N(0,1) noise, so the recovered
    # coefficient lives on a scale UNRELATED to the real outcome — dividing by
    # |orig| would make the verdict depend on the real outcome's units
    # (spurious FAIL for small-scale outcomes like remoteness in [0,1]).
    # Compare the STANDARDIZED placebo effect instead: |beta| * sd(T) / sd(y_dummy=1),
    # i.e. sd-units of dummy outcome per sd of treatment; ~0 for a sound estimator.
    r = refute(method_name="dummy_outcome_refuter", num_simulations=N)
    effs = _effects(r)
    worst = max(effs, key=abs) if effs else float("nan")
    t_sd = float(np.std(model._data[model._treatment[0]].to_numpy(float))) or 1.0
    std_eff = abs(worst) * t_sd
    results.append(RefuterResult(
        "Dummy Outcome", "dummy_outcome_refuter",
        "effect should go to ~0", orig, worst,
        f"worst standardized effect {std_eff:.3f} sd (<= {config.dummy_zero_frac:.2f} sd)",
        std_eff <= config.dummy_zero_frac, {"all_new_effects": effs, "t_sd": t_sd}))

    # 4. Simulated Outcome (known injected effect) -------------------------- #
    # Implemented as a transparent manual known-DGP swap rather than via
    # dummy_outcome_refuter's estimator path: that path bins the continuous
    # treatment and dowhy 0.14 has a bug (preprocess_data_by_treatment line 758
    # takes data.max() over ALL columns -> non-monotonic pd.cut bins). We build
    # y_sim = f(W) + beta*(T - mean T) + noise for KNOWN beta, re-estimate with the
    # same identification + adjustment set, and check beta is recovered.
    injected = (config.simulated_injected_effect
                if config.simulated_injected_effect is not None
                else (orig if abs(orig) > 1e-6 else 1.0))
    recovered = _simulated_outcome_recovery(model, injected, config)
    miss = abs(recovered - injected) / max(abs(injected), 1e-9)
    results.append(RefuterResult(
        "Simulated Outcome", "known-DGP outcome swap (manual)",
        "recovered effect should MATCH injected", orig, recovered,
        f"injected {injected:.4f}; recovered {recovered:.4f}; miss {miss:.1%} "
        f"(<= {config.simulated_match_tol:.0%})",
        miss <= config.simulated_match_tol, {"injected": injected}))

    # 5. Add Unobserved Common Cause (linear simulation grid) --------------- #
    # dowhy 0.14's auto-calibration (_infer_default_kappa_t) crashes on this data
    # shape, so we always pass an explicit strength grid. NOTE for Stage A: calibrate
    # this grid to bracket the effect strength of the observed confounders
    # (region / distance) so the test asks "would a confounder no stronger than the
    # ones we DID measure flip the sign?".
    ke_t = (config.unobserved_effect_strength_treatment
            or [0.01, 0.02, 0.03, 0.04, 0.05])
    ke_y = (config.unobserved_effect_strength_outcome
            or [0.01, 0.02, 0.03, 0.04, 0.05])
    r = refute(method_name="add_unobserved_common_cause",
               confounders_effect_on_treatment="linear",
               confounders_effect_on_outcome="linear",
               effect_strength_on_treatment=ke_t,
               effect_strength_on_outcome=ke_y)
    effs = _effects(r)
    # worst case = the simulated estimate closest to zero / most sign-flipping
    worst = min(effs, key=abs) if effs else float("nan")
    sign_preserved = bool(effs) and all(np.sign(e) == np.sign(orig) for e in effs)
    results.append(RefuterResult(
        "Add Unobserved Common Cause", "add_unobserved_common_cause",
        "estimate should not be too sensitive (sign preserved)", orig, worst,
        f"sign preserved across grid: {sign_preserved}; "
        f"estimate range [{min(effs):.4f}, {max(effs):.4f}]" if effs else "no grid returned",
        sign_preserved, {"grid_new_effects": effs}))

    # 6. Data Subset Validation --------------------------------------------- #
    r = refute(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=N)
    results.append(stable("Data Subset Validation", "data_subset_refuter",
                          r, config.rel_change_subset))

    # 7. Bootstrap Validation ----------------------------------------------- #
    r = refute(method_name="bootstrap_refuter", num_simulations=N)
    results.append(stable("Bootstrap Validation", "bootstrap_refuter",
                          r, config.rel_change_bootstrap))

    print_report(results, f"REFUTATION REPORT   (original effect = {orig:.5f})")
    if out_dir:
        write_report(results, config, out_dir, stem="refutation_report",
                     title="Refutation report",
                     extra={"original_effect": orig},
                     md_header=[f"Original effect estimate: **{orig:.5f}**"])
    return results


def _flag(passed) -> str:
    """Tri-state: True=PASS, False=FAIL, None=PEND (sensitivity.py stores None
    for not-yet-computable checks in the same RefuterResult type)."""
    return "PEND" if passed is None else ("PASS" if passed else "FAIL")


def print_report(results: list[RefuterResult], title: str,
                 width: int = 82, name_w: int = 34) -> None:
    """Console table shared by the DoWhy battery and sensitivity.py's design
    refutations, so the two reports render identically (incl. PEND rows)."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"{'#':<2} {'test':<{name_w}} {'result':<8} detail")
    print("-" * width)
    for i, r in enumerate(results, 1):
        print(f"{i:<2} {r.name:<{name_w}} {_flag(r.passed):<8} {r.statistic}")
    n_pass = sum(r.passed is True for r in results)
    n_pend = sum(r.passed is None for r in results)
    print("-" * width)
    tail = f" ({n_pend} pending)" if n_pend else ""
    print(f"{n_pass}/{len(results)} refutations passed{tail}")
    print("=" * width)


def write_report(results: list[RefuterResult], config, out_dir: str,
                 stem: str, title: str,
                 extra: Optional[dict] = None,
                 md_header: Optional[list[str]] = None) -> None:
    """JSON + markdown report writer shared with sensitivity.py. `extra` merges
    into the JSON payload; `md_header` lines go under the markdown title."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_pass = sum(r.passed is True for r in results)
    n_pend = sum(r.passed is None for r in results)
    payload = {
        **(extra or {}),
        "config": asdict(config),
        "n_passed": n_pass,
        "n_pending": n_pend,
        "n_total": len(results),
        "results": [asdict(r) for r in results],
    }
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str),
                                      encoding="utf-8")
    marks = {True: "✅ PASS", False: "❌ FAIL", None: "⏳ PENDING"}
    pend_note = f" ({n_pend} pending)" if n_pend else ""
    lines = [f"# {title}\n", *[h + "\n" for h in (md_header or [])],
             f"Passed **{n_pass}/{len(results)}**{pend_note}\n",
             "| # | Test | Hint | Result | Detail |", "|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r.name} | {r.hint} | "
                     f"{marks[r.passed]} | {r.statistic} |")
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> wrote {out / (stem + '.json')} and .md")


# --------------------------------------------------------------------------- #
# Self-test on synthetic data
# --------------------------------------------------------------------------- #
def _selftest(num_sims: int = 25) -> int:
    import dowhy.datasets
    np.random.seed(7)
    data = dowhy.datasets.linear_dataset(
        beta=8, num_common_causes=3, num_effect_modifiers=1,
        num_samples=1500, num_instruments=0, treatment_is_binary=False)
    model = CausalModel(data=data["df"], treatment=data["treatment_name"],
                        outcome=data["outcome_name"], graph=data["gml_graph"])
    ident = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(ident, method_name="backdoor.linear_regression")

    cfg = RefuteConfig(num_simulations=num_sims)
    results = run_all_refutations(model, ident, est, config=cfg)
    # On a correctly-specified linear DGP, ALL seven should pass.
    ok = all(r.passed for r in results)
    print(f"\nSELF-TEST {'OK — all refutations passed' if ok else 'had failures (see above)'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="DoWhy refutation battery.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the 7 refuters on a synthetic linear dataset.")
    p.add_argument("--num-sims", type=int, default=25)
    args = p.parse_args()
    if args.selftest:
        return _selftest(args.num_sims)
    print("Nothing to do. Import run_all_refutations() from Stage A, or pass --selftest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
