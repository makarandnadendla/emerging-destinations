"""
Stage-A DAG node-specific refutations — pre-registered.

`refute.py` holds the seven DoWhy refuters that attack the *confounding* and
*estimator-validity* parts of the DAG (the back-door set + "is the effect real /
stable" claims). This module implements the three refutations the DAG
(analysis/dag.html) and SPEC.md §147-149 name explicitly but that DoWhy's
`refute_estimate` does not provide — one per labelled node:

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
       (and to drag the CI limit to the null). Complements refute.py's
       Cinelli-Hazlett Robustness Value (partial-R^2 scale): same question,
       different, more reportable scale.

  C. HOMERES — stated-vs-modal HOME-RESOLUTION agreement    (SPEC §97, §128)
       Cross-check the two origin-assignment methods (stated profile country vs
       modal photo-pattern country). Reports raw agreement + Cohen's kappa.
       Disagreement is non-differential measurement error on the treatment
       assignment; low agreement => analyze conflicts separately. PASS =
       agreement and kappa above pre-registered floors. Reports PENDING (not
       FAIL) while modal_country_iso is unpopulated (needs the global photo pull).

Thresholds are committed BEFORE the real estimate exists so the checks can't be
reverse-engineered into a pass — same discipline as refute.py.

Usage (from Stage A, after you have the user-level analysis frame `df` and the
headline estimate):

    from analysis.sensitivity import run_design_refutations, SensitivityConfig
    report = run_design_refutations(
        df, treatment="hdi", neg_outcome="y_neg_hour",
        confounders=["gdp_pc_ppp", "log_photos", "dist_capital_km"],
        categorical=["origin_region"], month_col="trip_month_modal",
        cluster_col="origin_iso", stated_col="origin_iso",   # user_features exposes the
        modal_col="modal_country_iso",                       # STATED home as origin_iso
        headline=(beta, ci_low, ci_high), outcome_sd=df["y_mean_remoteness"].std(),
        out_dir="analysis/outputs")

Self-test on synthetic data (no warehouse needed):

    uv run python analysis/sensitivity.py --selftest
"""
from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Reuse the report row type AND the report renderers so the design refutations
# and the DoWhy battery render as one system (works run-as-script or imported).
try:
    from analysis.refute import RefuterResult, print_report, write_report
except ImportError:  # running as `python analysis/sensitivity.py`
    from refute import RefuterResult, print_report, write_report


# --------------------------------------------------------------------------- #
# Pre-registered configuration (committed before the real estimate)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# A. Hour-of-day negative-control outcome
# --------------------------------------------------------------------------- #
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

    beta, se, ci, pval, dof = _adjusted_coef(
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


# --------------------------------------------------------------------------- #
# B. VanderWeele / Ding E-value
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# C. Stated-vs-modal home-resolution agreement
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Adjusted regression helper (statsmodels if available, numpy fallback)
# --------------------------------------------------------------------------- #
def _design(d: pd.DataFrame, treatment: str, numeric: list[str],
            categorical: list[str]):
    """Build [intercept, treatment, numerics, one-hot(categoricals)] design matrix."""
    parts = [np.ones((len(d), 1)), d[[treatment]].to_numpy(float)]
    names = ["const", treatment]
    for c in numeric:
        parts.append(d[[c]].to_numpy(float)); names.append(c)
    for c in categorical:
        dummies = pd.get_dummies(d[c].astype("category"), prefix=c, drop_first=True)
        if dummies.shape[1]:
            parts.append(dummies.to_numpy(float)); names.extend(dummies.columns.tolist())
    return np.hstack(parts), names


def _adjusted_coef(d: pd.DataFrame, outcome: str, treatment: str,
                   numeric: list[str], categorical: list[str],
                   cluster_col: Optional[str], ci_level: float):
    """Return (beta, se, (ci_lo, ci_hi), p, dof) for `treatment`, with cluster-
    robust SEs when cluster_col is given (CR1), else HC1."""
    from scipy import stats

    X, names = _design(d, treatment, numeric, categorical)
    y = d[outcome].to_numpy(float)
    j = names.index(treatment)
    n, kf = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_vec = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_vec
    beta = float(beta_vec[j])

    if cluster_col is not None:
        groups = d[cluster_col].to_numpy()
        uniq = np.unique(groups)
        G = len(uniq)
        meat = np.zeros((kf, kf))
        for g in uniq:
            m = groups == g
            sg = X[m].T @ resid[m]
            meat += np.outer(sg, sg)
        adj = (G / (G - 1.0)) * ((n - 1.0) / (n - kf)) if G > 1 else 1.0
        V = adj * (XtX_inv @ meat @ XtX_inv)
        dof = max(G - 1, 1)
    else:
        meat = (X * (resid ** 2)[:, None]).T @ X
        V = (n / (n - kf)) * (XtX_inv @ meat @ XtX_inv)
        dof = max(n - kf, 1)

    se = float(np.sqrt(max(V[j, j], 0.0)))
    tcrit = float(stats.t.ppf(0.5 + ci_level / 2.0, dof))
    ci = (beta - tcrit * se, beta + tcrit * se)
    tstat = beta / se if se > 0 else 0.0
    pval = float(2 * stats.t.sf(abs(tstat), dof))
    return beta, se, ci, pval, dof


# --------------------------------------------------------------------------- #
# Combined runner + report
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Self-test on synthetic data
# --------------------------------------------------------------------------- #
def _selftest() -> int:
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
    pt, se, ci, p, dof = _adjusted_coef(
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
    print(f"\n[selftest] headline beta={pt:+.4f} (true {beta_true:+.2f})")
    ok = (nc.passed is True) and (ev.passed is True) and (ag.passed is True)
    # Sanity: also confirm a PENDING modal column is reported, not crashed/failed.
    df2 = df.copy(); df2["modal_country_iso"] = np.nan
    pend = home_resolution_agreement(df2, "stated_country_iso", "modal_country_iso")
    ok = ok and (pend.passed is None)
    print(f"[selftest] negative-control PASS={nc.passed}  E-value PASS={ev.passed}  "
          f"agreement PASS={ag.passed}  pending-handled={pend.passed is None}")
    print(f"\nSELF-TEST {'OK — all design refutations behave as designed' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="DAG design-refutation battery.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the three design refutations on synthetic data.")
    args = p.parse_args()
    if args.selftest:
        return _selftest()
    print("Nothing to do. Import run_design_refutations() from Stage A, or pass --selftest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
