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
    # "should go to zero" refuters: |new effect| as a fraction of |original|
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


# Module-level (picklable) known-effect function for the simulated-outcome DGP.
class _LinearEffect:
    def __init__(self, beta: float):
        self.beta = float(beta)

    def __call__(self, t):
        return self.beta * np.asarray(t, dtype=float)


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
        X = df[feats].to_numpy(float)
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
                        config: RefuteConfig = RefuteConfig(),
                        out_dir: Optional[str] = None) -> list[RefuterResult]:
    np.random.seed(config.random_seed)
    orig = float(estimate.value)
    N = config.num_simulations
    results: list[RefuterResult] = []

    def refute(**kw):
        return model.refute_estimate(identified_estimand, estimate, **kw)

    # 1. Add Random Common Cause -------------------------------------------- #
    r = refute(method_name="random_common_cause", num_simulations=N)
    new = _effects(r)[0]
    results.append(RefuterResult(
        "Add Random Common Cause", "random_common_cause",
        "estimate should NOT change", orig, new,
        f"rel. change {_rel(new, orig):.1%} (<= {config.rel_change_random_cause:.0%})",
        _rel(new, orig) <= config.rel_change_random_cause,
        {"p_value": _pval(r)}))

    # 2. Placebo Treatment --------------------------------------------------- #
    r = refute(method_name="placebo_treatment_refuter", placebo_type="permute",
               num_simulations=N)
    new = _effects(r)[0]
    frac = abs(new) / max(abs(orig), 1e-9)
    results.append(RefuterResult(
        "Placebo Treatment", "placebo_treatment_refuter",
        "effect should go to ~0", orig, new,
        f"|new|/|orig| {frac:.1%} (<= {config.placebo_zero_frac:.0%})",
        frac <= config.placebo_zero_frac, {"p_value": _pval(r)}))

    # 3. Dummy Outcome (true effect = 0) ------------------------------------ #
    r = refute(method_name="dummy_outcome_refuter", num_simulations=N)
    effs = _effects(r)
    worst = max(effs, key=abs) if effs else float("nan")
    frac = abs(worst) / max(abs(orig), 1e-9)
    results.append(RefuterResult(
        "Dummy Outcome", "dummy_outcome_refuter",
        "effect should go to ~0", orig, worst,
        f"worst |new|/|orig| {frac:.1%} (<= {config.dummy_zero_frac:.0%})",
        frac <= config.dummy_zero_frac, {"all_new_effects": effs}))

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
    new = _effects(r)[0]
    results.append(RefuterResult(
        "Data Subset Validation", "data_subset_refuter",
        "estimate should NOT change", orig, new,
        f"rel. change {_rel(new, orig):.1%} (<= {config.rel_change_subset:.0%})",
        _rel(new, orig) <= config.rel_change_subset, {"p_value": _pval(r)}))

    # 7. Bootstrap Validation ----------------------------------------------- #
    r = refute(method_name="bootstrap_refuter", num_simulations=N)
    new = _effects(r)[0]
    results.append(RefuterResult(
        "Bootstrap Validation", "bootstrap_refuter",
        "estimate should NOT change", orig, new,
        f"rel. change {_rel(new, orig):.1%} (<= {config.rel_change_bootstrap:.0%})",
        _rel(new, orig) <= config.rel_change_bootstrap, {"p_value": _pval(r)}))

    _print_report(orig, results)
    if out_dir:
        _write_report(orig, results, config, out_dir)
    return results


def _print_report(orig: float, results: list[RefuterResult]) -> None:
    print("\n" + "=" * 78)
    print(f"REFUTATION REPORT   (original effect = {orig:.5f})")
    print("=" * 78)
    print(f"{'#':<2} {'test':<30} {'result':<8} detail")
    print("-" * 78)
    for i, r in enumerate(results, 1):
        flag = "PASS" if r.passed else "FAIL"
        print(f"{i:<2} {r.name:<30} {flag:<8} {r.statistic}")
    n_pass = sum(r.passed for r in results)
    print("-" * 78)
    print(f"{n_pass}/{len(results)} refutations passed")
    print("=" * 78)


def _write_report(orig: float, results: list[RefuterResult],
                  config: RefuteConfig, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "original_effect": orig,
        "config": asdict(config),
        "n_passed": sum(r.passed for r in results),
        "n_total": len(results),
        "results": [asdict(r) for r in results],
    }
    (out / "refutation_report.json").write_text(json.dumps(payload, indent=2, default=str),
                                                encoding="utf-8")
    lines = [f"# Refutation report\n", f"Original effect estimate: **{orig:.5f}**\n",
             f"Passed **{payload['n_passed']}/{payload['n_total']}**\n",
             "| # | Test | Hint | Result | Detail |", "|---|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r.name} | {r.hint} | "
                     f"{'✅ PASS' if r.passed else '❌ FAIL'} | {r.statistic} |")
    (out / "refutation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> wrote {out/'refutation_report.json'} and .md")


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
