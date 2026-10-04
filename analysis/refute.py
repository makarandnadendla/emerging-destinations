"""
Stage-A refutations — pre-registered, in one module.

This file merges the former refute.py (DoWhy battery), sensitivity.py (DAG
node-specific design refutations) and run_refutations.py (the live driver).
Every PASS threshold below was committed BEFORE the real estimate existed and
is carried over VERBATIM from those files, so the pre-registration trail is
unbroken. Estimators live in estimate.py; the identification assumptions live
in assumptions_dag.py; this module only attacks.

PART 1 — the seven DoWhy refutations (method_name in parentheses) and what
each attacks in our DAG (see analysis/dag.html):

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
       NOTE 2: dowhy 0.14's GRID path compounds the injected confounders across
       grid cells (the deep-copied frame is mutated in place and never reset),
       wildly exaggerating the reported range — so the driver below runs the
       grid as per-cell SINGLE-value calls, which deep-copy per call.
  6. Data Subset Validation   (data_subset_refuter, subset_fraction=0.8)
       Re-estimate on random subsets; estimate should be stable. Finite-sample
       fragility / influential-subsample check.
  7. Bootstrap Validation     (bootstrap_refuter)
       Re-estimate on bootstrap resamples; estimate should be stable and the point
       estimate should sit inside the bootstrap spread.

PART 2 — the three refutations the DAG and SPEC.md §147-149 name explicitly
but DoWhy's refute_estimate does not provide — one per labelled node:

  A. HOUR  — hour-of-day NEGATIVE-CONTROL outcome           (SPEC §147)
       Re-run the SAME adjusted spec with the placebo outcome = per-user mean
       photo hour-of-day, conditioned on month-of-year. The DAG asserts origin
       HDI has NO path to *when of day* a traveler shoots, so the coefficient
       must be ~0. A significant coefficient exposes residual confounding /
       selection — the  HDI -> Flickr -> [SAMPLE] <- REGION -> Y  collider leg
       the design cannot close. PASS = 95% CI covers 0.
  B. (unmeasured confounding) — VanderWeele/Ding E-VALUE    (SPEC §149)
       The minimum strength, on the risk-ratio scale, an unmeasured confounder
       would need with BOTH HDI and remoteness to explain away the point estimate
       (and to drag the CI limit to the null). Complements the Cinelli-Hazlett
       Robustness Value (partial-R^2 scale): same question, different scale.
  C. HOMERES — stated-vs-modal HOME-RESOLUTION agreement    (SPEC §97, §128)
       Cross-check the two origin-assignment methods. Reports raw agreement +
       Cohen's kappa against pre-registered floors; PENDING (not FAIL) while
       modal_country_iso is unpopulated.

PART 3 — the live driver (--run): executes both batteries against the real
warehouse. The design refutations are composed PER-REGION (one E-value per
in-scope region) because the H3 headline is regional; the DoWhy battery
attacks the pooled Tier-2 LinearDML via estimate.estimate_dml, so every
refuter re-fits ladder rung 3 exactly, and the identified estimand is
embedded in the written reports.

Usage:
    uv run python analysis/refute.py --selftest           # both batteries, synthetic
    uv run python analysis/refute.py --run                # live run (100 sims/refuter)
    uv run python analysis/refute.py --run --num-sims 25  # quick pass

Outputs (with --run): analysis/outputs/design_refutation_report.{json,md}
                      analysis/outputs/refutation_report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
logging.getLogger("dowhy").setLevel(logging.ERROR)

import duckdb
import numpy as np
import pandas as pd
from dowhy import CausalModel

try:
    from analysis.assumptions_dag import build_model, describe_estimand, identify
    from analysis.config import CFG
    from analysis.estimate import (CONTRAST, SEED, WAREHOUSE, adjusted_coef,
                                   estimate_dml, identification_gate,
                                   load_frame, region_estimate)
except ImportError:
    from assumptions_dag import build_model, describe_estimand, identify
    from config import CFG
    from estimate import (CONTRAST, SEED, WAREHOUSE, adjusted_coef,
                          estimate_dml, identification_gate,
                          load_frame, region_estimate)

OUT_DIR = "analysis/outputs"


# --------------------------------------------------------------------------- #
# Pre-registered configuration (thresholds fixed before the real estimate;
# moved VERBATIM from the former refute.py / sensitivity.py)
# --------------------------------------------------------------------------- #
@dataclass
class RefuteConfig:
    num_simulations: int = 100          # bootstrap/subset/random-cause/placebo sims
    random_seed: int = SEED

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

    # Unobserved-confounder linear grid. None -> the default grid below.
    # NOTE for Stage A: calibrate to bracket the strength of the observed
    # confounders (region / distance) so the test asks "would a confounder no
    # stronger than the ones we DID measure flip the sign?".
    unobserved_effect_strength_treatment: Optional[list] = None
    unobserved_effect_strength_outcome: Optional[list] = None

    # Estimator used to RE-estimate on the simulated-outcome dataset (#4). Default
    # mirrors the linear baseline; set to the Stage-A DML method to test that estimator.
    reestimate_method_name: str = "backdoor.linear_regression"
    reestimate_method_params: Optional[dict] = None


@dataclass
class SensitivityConfig:
    # A. Negative control: PASS when the placebo coefficient's 95% CI covers 0
    #    (the DAG predicts no HDI -> hour effect). We additionally flag if the
    #    standardized magnitude is large even when non-significant.
    neg_ci_level: float = 0.95
    neg_std_warn: float = 0.10          # |std. coef| above this is flagged in detail

    # B. E-value: report E for the point estimate and for the CI limit nearest the
    #    null. PASS = the CI-limit E-value clears a pre-registered floor, i.e. a
    #    confounder would need at least this RR-on-both to overturn significance.
    evalue_floor: float = 1.25
    evalue_smd_to_rr: float = 0.91      # VanderWeele 2017 approx: RR ~= exp(0.91*d)

    # C. Home-resolution agreement floors
    agreement_floor: float = 0.70
    kappa_floor: float = 0.60


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
# Report rendering (shared by both batteries, incl. PEND rows)
# --------------------------------------------------------------------------- #
def _flag(passed) -> str:
    """Tri-state: True=PASS, False=FAIL, None=PEND (not-yet-computable or
    not-applicable checks live in the same RefuterResult type)."""
    return "PEND" if passed is None else ("PASS" if passed else "FAIL")


def print_report(results: list[RefuterResult], title: str,
                 width: int = 82, name_w: int = 34) -> None:
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
    """JSON + markdown report writer. `extra` merges into the JSON payload;
    `md_header` lines go under the markdown title."""
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


# =========================================================================== #
# PART 1 — the DoWhy battery
# =========================================================================== #
def estimate_linear(model: CausalModel, identified_estimand):
    """Baseline backdoor linear-regression estimate (refuters are estimator-agnostic,
    but the simulated/dummy outcome DGPs require a covariate-adjustment estimator)."""
    return model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")


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

    try:
        from analysis.assumptions_dag import make_causal_model
    except ImportError:
        from assumptions_dag import make_causal_model

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


def run_all_refutations(model: CausalModel, identified_estimand, estimate,
                        config: Optional[RefuteConfig] = None,
                        out_dir: Optional[str] = None,
                        report_extra: Optional[dict] = None) -> list[RefuterResult]:
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
    # shape, so we always pass an explicit strength grid — and we iterate the
    # grid OURSELVES as single-value calls: dowhy's own array path mutates its
    # working frame in place across cells, COMPOUNDING the injected confounders
    # (verified against the closed-form OVB prediction, 2026-10-02).
    ke_t = (config.unobserved_effect_strength_treatment
            or [0.01, 0.02, 0.03, 0.04, 0.05])
    ke_y = (config.unobserved_effect_strength_outcome
            or [0.01, 0.02, 0.03, 0.04, 0.05])
    effs = []
    for kt in ke_t:
        for ky in ke_y:
            r = refute(method_name="add_unobserved_common_cause",
                       confounders_effect_on_treatment="linear",
                       confounders_effect_on_outcome="linear",
                       effect_strength_on_treatment=float(kt),
                       effect_strength_on_outcome=float(ky))
            effs.extend(_effects(r))
    # worst case = the simulated estimate closest to zero / most sign-flipping
    worst = min(effs, key=abs) if effs else float("nan")
    sign_preserved = bool(effs) and all(np.sign(e) == np.sign(orig) for e in effs)
    results.append(RefuterResult(
        "Add Unobserved Common Cause", "add_unobserved_common_cause (per-cell grid)",
        "estimate should not be too sensitive (sign preserved)", orig, worst,
        f"sign preserved across grid: {sign_preserved}; "
        f"estimate range [{min(effs):.4f}, {max(effs):.4f}]" if effs else "no grid returned",
        sign_preserved, {"grid_new_effects": effs,
                         "grid_kappa_t": list(ke_t), "grid_kappa_y": list(ke_y)}))

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
                     extra={"original_effect": orig, **(report_extra or {})},
                     md_header=[f"Original effect estimate: **{orig:.5f}**",
                                *(f"```\n{v}\n```" for k, v in (report_extra or {}).items()
                                  if k == "estimand")])
    return results


# =========================================================================== #
# PART 2 — the DAG node-specific design refutations
# =========================================================================== #
def negative_control(df: pd.DataFrame, treatment: str, neg_outcome: str,
                     confounders: list[str], categorical: Optional[list[str]] = None,
                     month_col: Optional[str] = None,
                     cluster_col: Optional[str] = None,
                     config: Optional[SensitivityConfig] = None) -> RefuterResult:
    """Estimate the treatment coefficient on the PLACEBO outcome under the SAME
    adjustment set as the headline, conditioned on month-of-year. The DAG predicts
    ~0; a CI that excludes 0 reveals residual confounding / selection.

    Note: y_neg_hour is a circular quantity treated linearly here. Acceptable
    because Japan tourism hours cluster mid-day (mass near the 0/24 seam is
    negligible); if the y_neg_hour histogram ever shows seam mass, switch to
    regressing the sin/cos components instead."""
    config = config or SensitivityConfig()
    cats = list(categorical or [])
    if month_col:
        cats = cats + [month_col]
    cols = [neg_outcome, treatment, *confounders, *cats]
    if cluster_col:
        cols.append(cluster_col)
    d = df.dropna(subset=[c for c in cols if c in df.columns]).copy()

    beta, se, ci, pval, dof = adjusted_coef(
        d, outcome=neg_outcome, treatment=treatment,
        numeric=confounders, categorical=cats, cluster_col=cluster_col,
        ci_level=config.neg_ci_level)

    sd_out = float(d[neg_outcome].std()) or 1.0
    std_coef = beta / sd_out
    covers_zero = ci[0] <= 0.0 <= ci[1]
    return RefuterResult(
        name="Hour-of-day Negative Control",
        method="adjusted placebo-outcome regression (cluster-robust)",
        hint="placebo coefficient should be ~0 (95% CI covers 0)",
        original_effect=float("nan"),
        observed=beta,
        statistic=(f"beta={beta:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                   f"p={pval:.3f}  std.coef={std_coef:+.3f}"),
        passed=bool(covers_zero),
        detail={"se": se, "p_value": pval, "ci": list(ci), "dof": dof,
                "std_coef": std_coef,
                "magnitude_flag": abs(std_coef) > config.neg_std_warn,
                "n": int(len(d))})


def _evalue_from_rr(rr: float) -> float:
    """E-value for a risk ratio (VanderWeele & Ding 2017). RR<1 is mirrored."""
    if rr <= 0 or not math.isfinite(rr):
        return float("nan")
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + math.sqrt(rr * (rr - 1.0))


def evalue(point: float, ci_low: float, ci_high: float, outcome_sd: float,
           contrast: float = 1.0,
           config: Optional[SensitivityConfig] = None) -> RefuterResult:
    """E-value for a continuous-outcome effect. The effect is standardized
    (d = effect*contrast / sd) and mapped to an approximate RR via exp(0.91*d),
    then converted to the E-value. `contrast` lets you ask about a meaningful HDI
    change (e.g. 0.1 of HDI) instead of one full unit."""
    config = config or SensitivityConfig()
    if not outcome_sd or not math.isfinite(outcome_sd):
        outcome_sd = 1.0
    k = config.evalue_smd_to_rr

    def rr(x):
        return math.exp(k * (x * contrast) / outcome_sd)

    rr_pt = rr(point)
    e_point = _evalue_from_rr(rr_pt)
    res = partial(
        RefuterResult,
        name="E-value (unmeasured confounding)",
        method="VanderWeele-Ding E-value (continuous, SMD->RR)",
        hint=f"a confounder would need RR>=this on BOTH to overturn (floor {config.evalue_floor})",
        original_effect=point)
    base = {"e_point": e_point, "rr_point": rr_pt,
            "contrast": contrast, "outcome_sd": outcome_sd}

    rr_lo, rr_hi = rr(ci_low), rr(ci_high)
    if (rr_lo - 1.0) * (rr_hi - 1.0) <= 0:
        # CI crosses the null: there is no significant effect to explain away,
        # so an E-value floor is NOT APPLICABLE — this is the pre-registered H4
        # outcome, not a failed robustness check. Report PENDING/NA, never FAIL.
        return res(
            observed=e_point,
            statistic=(f"not applicable — headline CI crosses the null "
                       f"(H4-consistent); E(point)={e_point:.2f} reported for context"),
            passed=None,
            detail={**base, "e_ci": None, "status": "not_applicable_null_headline"})

    rr_ci = min((rr_lo, rr_hi), key=lambda r: abs(math.log(r)))  # limit nearest 1
    e_ci = _evalue_from_rr(rr_ci)
    return res(
        observed=e_ci,
        statistic=(f"E(point)={e_point:.2f}  E(CI limit)={e_ci:.2f}  "
                   f"floor={config.evalue_floor:.2f}  (approx RR={rr_pt:.3f})"),
        passed=bool(e_ci >= config.evalue_floor),
        detail={**base, "e_ci": e_ci})


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    po = float(np.mean(a == b))
    pa = np.zeros(len(labels)); pb = np.zeros(len(labels))
    for x in a: pa[idx[x]] += 1
    for x in b: pb[idx[x]] += 1
    pa /= n; pb /= n
    pe = float(np.sum(pa * pb))
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0


def home_resolution_agreement(df: pd.DataFrame, stated_col: str, modal_col: str,
                              config: Optional[SensitivityConfig] = None
                              ) -> RefuterResult:
    """Agreement between stated profile country and modal photo-pattern country.
    Returns PENDING (passed=None) when the check cannot be computed — modal
    entirely unpopulated, or zero rows with BOTH sides present — so a
    not-yet-computable check never reads as a silent PASS or a NaN FAIL."""
    config = config or SensitivityConfig()
    res = partial(
        RefuterResult,
        name="Home-Resolution Agreement (stated vs modal)",
        method="raw agreement + Cohen's kappa",
        hint=f"agreement>={config.agreement_floor:.0%} & kappa>={config.kappa_floor:.2f}",
        original_effect=float("nan"))

    if modal_col not in df.columns or df[modal_col].notna().sum() == 0:
        reason = "modal_country_iso unpopulated (needs global photo pull)"
        d = None
    else:
        d = df.dropna(subset=[stated_col, modal_col])
        reason = None if len(d) else "no rows with BOTH stated and modal populated"
    if reason:
        return res(observed=float("nan"), statistic=f"PENDING — {reason}",
                   passed=None, detail={"status": "pending", "reason": reason})

    a = d[stated_col].to_numpy(); b = d[modal_col].to_numpy()
    agree = float(np.mean(a == b))
    kappa = _cohen_kappa(a, b)
    passed = (agree >= config.agreement_floor) and (kappa >= config.kappa_floor)
    return res(
        observed=agree,
        statistic=(f"agreement={agree:.1%} (>={config.agreement_floor:.0%})  "
                   f"kappa={kappa:.3f} (>={config.kappa_floor:.2f})  n={len(d)}"),
        passed=bool(passed),
        detail={"agreement": agree, "kappa": kappa, "n": int(len(d)),
                "n_conflicts": int((a != b).sum())})


def run_design_refutations(df: pd.DataFrame, treatment: str, neg_outcome: str,
                           confounders: list[str],
                           categorical: Optional[list[str]] = None,
                           month_col: Optional[str] = None,
                           cluster_col: Optional[str] = None,
                           stated_col: Optional[str] = None,
                           modal_col: Optional[str] = None,
                           headline: Optional[tuple] = None,
                           outcome_sd: Optional[float] = None,
                           contrast: float = 1.0,
                           config: Optional[SensitivityConfig] = None,
                           out_dir: Optional[str] = None) -> list[RefuterResult]:
    """Single-headline composer (kept for synthetic use / the selftest). The
    live H3 headline is per-region, so the driver below composes the three
    checks itself with one E-value row per in-scope region."""
    config = config or SensitivityConfig()   # fresh default per call, never shared
    results: list[RefuterResult] = []

    results.append(negative_control(
        df, treatment, neg_outcome, confounders, categorical, month_col,
        cluster_col, config))

    if headline is not None:
        pt, lo, hi = headline
        results.append(evalue(pt, lo, hi, outcome_sd or 1.0, contrast, config))

    if stated_col and modal_col:
        results.append(home_resolution_agreement(df, stated_col, modal_col, config))

    print_report(results, "DAG DESIGN-REFUTATION REPORT  "
                          "(node-specific checks beyond the DoWhy battery)")
    if out_dir:
        write_report(results, config, out_dir, stem="design_refutation_report",
                     title="DAG design-refutation report")
    return results


# =========================================================================== #
# PART 3 — the live driver (both batteries against the real warehouse)
# =========================================================================== #
def design_frame():
    """Analysis frame + the design-refutation columns (negative-control hour,
    modal trip month, modal home country) from user_features."""
    df = load_frame()
    con = duckdb.connect(WAREHOUSE, read_only=True)
    extra = con.execute("""
        SELECT user_id_hash, y_neg_hour, trip_month_modal, modal_country_iso
        FROM user_features
    """).df()
    con.close()
    df = df.merge(extra, on="user_id_hash", how="left")
    df["yr"] = df.trip_year.astype(int).astype(str)
    return df


def pooled_ols(d):
    """Rung-1 pooled OLS (per-unit scale) — reused for the E-value and the RV."""
    return adjusted_coef(
        d.dropna(subset=["y_first_trip"]).rename(columns={"y_first_trip": "_y"}),
        outcome="_y", treatment="hdi", numeric=[], categorical=["region", "yr"],
        cluster_col="origin_iso", ci_level=0.95)


def run_design_battery(df, in_scope) -> list[RefuterResult]:
    cfg = SensitivityConfig()
    results = [negative_control(
        df, treatment="hdi", neg_outcome="y_neg_hour", confounders=[],
        categorical=["region", "yr"], month_col="trip_month_modal",
        cluster_col="origin_iso", config=cfg)]

    beta, se, ci, p, dof = pooled_ols(df)
    r = evalue(beta * CONTRAST, ci[0] * CONTRAST, ci[1] * CONTRAST,
               outcome_sd=float(df.y_first_trip.std()), contrast=1.0, config=cfg)
    r.name += " — pooled rung 1"
    results.append(r)
    for reg in in_scope:
        sub = df[df.region == reg].dropna(subset=["y_first_trip"])
        b, bci, bp, bdof = region_estimate(sub, "y_first_trip")
        r = evalue(b, bci[0], bci[1], outcome_sd=float(sub.y_first_trip.std()),
                   contrast=1.0, config=cfg)
        r.name += f" — {reg}"
        results.append(r)

    results.append(home_resolution_agreement(
        df, "origin_iso", "modal_country_iso", config=cfg))

    print_report(results, "DAG DESIGN-REFUTATION REPORT  (live run; one E-value "
                          "per in-scope region — H3 headline)",
                 width=110, name_w=50)
    write_report(results, cfg, OUT_DIR, stem="design_refutation_report",
                 title="DAG design-refutation report (live run)")
    return results


def run_dowhy_battery(df, num_sims: int) -> list[RefuterResult]:
    d = df.dropna(subset=["y_first_trip"])
    model = build_model(d, "y_first_trip")
    ident = identify(model)
    est = estimate_dml(model, ident)
    estimand_txt = describe_estimand(ident)
    print(f"\nDoWhy battery target — pooled LinearDML('auto'), ATE per "
          f"+{CONTRAST:.2f} HDI = {float(est.value) * CONTRAST:+.4f} "
          f"(= estimate.py rung 3); {num_sims} simulations per refuter")
    print("  " + estimand_txt.replace("\n", "\n  "))

    cfg = RefuteConfig(num_simulations=num_sims)
    results = run_all_refutations(model, ident, est, config=cfg, out_dir=OUT_DIR,
                                  report_extra={"estimand": estimand_txt})

    beta, se, ci, p, dof = pooled_ols(df)
    rv = robustness_value(beta / se, dof)
    print(f"\nCinelli-Hazlett RV_1 (pooled rung-1 OLS, t={beta / se:.2f}, "
          f"dof={dof}): {rv:.3f} — a confounder needs partial-R^2 >= {rv:.1%} "
          f"with BOTH treatment and outcome to drive the pooled estimate to 0")
    return results


def run_live(num_sims: int) -> int:
    df = design_frame()
    in_scope = identification_gate(df, verbose=False)
    print(f"-> in-scope regions: {', '.join(in_scope)}")

    design = run_design_battery(df, in_scope)
    battery = run_dowhy_battery(df, num_sims)

    hard_fail = [r.name for r in design + battery if r.passed is False]
    print(f"\n{'ALL CHECKS PASS' if not hard_fail else 'FAILURES: ' + ', '.join(hard_fail)}"
          f" ({sum(r.passed is True for r in design + battery)} pass / "
          f"{sum(r.passed is None for r in design + battery)} pending/NA / "
          f"{len(hard_fail)} fail)")
    return 1 if hard_fail else 0


# =========================================================================== #
# Self-tests on synthetic data (no warehouse needed)
# =========================================================================== #
def _selftest_battery(num_sims: int = 25) -> int:
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
    print(f"\n[battery selftest] {'OK — all refutations passed' if ok else 'FAILURES (see above)'}")
    return 0 if ok else 1


def _selftest_design() -> int:
    rng = np.random.default_rng(7)
    n = 4000
    region = rng.integers(0, 7, n)                       # confounder + modifier
    month = rng.integers(1, 13, n)                       # seasonal control
    # HDI confounded by region; remoteness = region effect + TRUE beta*HDI + noise
    hdi = 0.45 + 0.05 * region + rng.normal(0, 0.05, n)
    log_photos = rng.normal(3, 1, n)
    beta_true = 0.30
    remote = 0.10 * region + beta_true * (hdi - hdi.mean()) + rng.normal(0, 0.1, n)
    # Negative control: hour depends on region + month but NOT on HDI -> must be null
    hour = 12 + 0.3 * region + 0.2 * np.sin(2 * np.pi * month / 12) + rng.normal(0, 1.5, n)
    # Stated vs modal: ~88% agree
    iso_pool = np.array(["USA", "FRA", "CHN", "TWN", "THA", "ESP", "DEU"])
    stated = iso_pool[region]
    flip = rng.random(n) < 0.12
    modal = np.where(flip, iso_pool[rng.integers(0, 7, n)], stated)

    df = pd.DataFrame({
        "hdi": hdi, "y_mean_remoteness": remote, "y_neg_hour": hour,
        "origin_region": region.astype(str), "trip_month_modal": month.astype(str),
        "log_photos": log_photos, "origin_iso": stated,
        "stated_country_iso": stated, "modal_country_iso": modal,
    })

    # A headline estimate for the E-value (regress remoteness on hdi + region)
    pt, se, ci, p, dof = adjusted_coef(
        df, "y_mean_remoteness", "hdi", numeric=["log_photos"],
        categorical=["origin_region"], cluster_col="origin_iso", ci_level=0.95)

    results = run_design_refutations(
        df, treatment="hdi", neg_outcome="y_neg_hour",
        confounders=["log_photos"], categorical=["origin_region"],
        month_col="trip_month_modal", cluster_col="origin_iso",
        stated_col="stated_country_iso", modal_col="modal_country_iso",
        headline=(pt, ci[0], ci[1]), outcome_sd=df["y_mean_remoteness"].std())

    nc = next(r for r in results if r.name.startswith("Hour"))
    ev = next(r for r in results if r.name.startswith("E-value"))
    ag = next(r for r in results if r.name.startswith("Home"))
    print(f"\n[design selftest] headline beta={pt:+.4f} (true {beta_true:+.2f})")
    ok = (nc.passed is True) and (ev.passed is True) and (ag.passed is True)
    # Sanity: also confirm a PENDING modal column is reported, not crashed/failed.
    df2 = df.copy(); df2["modal_country_iso"] = np.nan
    pend = home_resolution_agreement(df2, "stated_country_iso", "modal_country_iso")
    ok = ok and (pend.passed is None)
    print(f"[design selftest] negative-control PASS={nc.passed}  E-value PASS={ev.passed}  "
          f"agreement PASS={ag.passed}  pending-handled={pend.passed is None}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pre-registered refutation batteries (DoWhy 7 + design 3) + live driver.")
    p.add_argument("--selftest", action="store_true",
                   help="Run both batteries on synthetic data.")
    p.add_argument("--run", action="store_true",
                   help="Run both batteries against the live warehouse estimates.")
    p.add_argument("--num-sims", type=int, default=None,
                   help="Simulations per DoWhy refuter "
                        "(default: 25 for --selftest, 100 for --run).")
    args = p.parse_args()
    if args.selftest:
        rc = _selftest_battery(args.num_sims or 25) + _selftest_design()
        print(f"\nSELF-TEST {'OK' if rc == 0 else 'FAILED'}")
        return 1 if rc else 0
    if args.run:
        return run_live(args.num_sims or RefuteConfig().num_simulations)
    print("Nothing to do: pass --selftest or --run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
