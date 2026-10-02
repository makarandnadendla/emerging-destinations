"""
Stage-A LIVE refutation run — both pre-registered batteries against the real
warehouse estimates (first executed 2026-10-02, after the first estimate.py --run).

Part A — design refutations (sensitivity.py; fast, runs first so a crash in the
  slow battery never costs these results). Composed from the component functions
  directly because the H3 headline is PER-REGION — one E-value row per in-scope
  region — which the single-headline run_design_refutations() signature predates:
    * hour-of-day negative control on the pooled adjusted spec
      (region + trip-year dummies + modal trip month, origin-clustered);
    * E-values for the pooled rung-1 OLS and each in-scope region's headline
      beta (per +0.10 HDI, region-specific outcome sd); null-crossing CIs
      report NA/PENDING by design (H4-consistent), never FAIL;
    * stated-vs-modal home-resolution agreement on the analysis frame.

Part B — DoWhy battery (refute.run_all_refutations, 7 tests) attacks the pooled
  Tier-2 secondary: LinearDML('auto', cv from config) of y_first_trip on HDI,
  W = region + trip-year dummies, DoWhy-wrapped via refute.estimate_dml so every
  refuter re-fits the identical estimator (ladder rung 3). The within-region
  headline tier is plain OLS; its estimator checks are the LOO + ladder
  agreement in estimate.py. Also prints the closed-form Cinelli-Hazlett
  robustness value on the pooled rung-1 OLS t-stat (refute.robustness_value;
  dowhy 0.14's partial-R2 path is unusable — see refute.py).

Usage:
    uv run python analysis/run_refutations.py              # full battery (100 sims)
    uv run python analysis/run_refutations.py --num-sims 25  # quick pass

Outputs: analysis/outputs/design_refutation_report.{json,md}
         analysis/outputs/refutation_report.{json,md}
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import duckdb

try:
    from analysis.estimate import (CONTRAST, WAREHOUSE, identification_gate,
                                   load_frame, region_estimate)
    from analysis.refute import (RefuteConfig, estimate_dml, make_causal_model,
                                 print_report, robustness_value,
                                 run_all_refutations, write_report)
    from analysis.sensitivity import (SensitivityConfig, _adjusted_coef, evalue,
                                      home_resolution_agreement, negative_control)
except ImportError:
    from estimate import (CONTRAST, WAREHOUSE, identification_gate,
                          load_frame, region_estimate)
    from refute import (RefuteConfig, estimate_dml, make_causal_model,
                        print_report, robustness_value,
                        run_all_refutations, write_report)
    from sensitivity import (SensitivityConfig, _adjusted_coef, evalue,
                             home_resolution_agreement, negative_control)

OUT_DIR = "analysis/outputs"


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
    return _adjusted_coef(
        d.dropna(subset=["y_first_trip"]).rename(columns={"y_first_trip": "_y"}),
        outcome="_y", treatment="hdi", numeric=[], categorical=["region", "yr"],
        cluster_col="origin_iso", ci_level=0.95)


def run_design_battery(df, in_scope) -> list:
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


def run_dowhy_battery(df, num_sims: int) -> list:
    d = df.dropna(subset=["y_first_trip"])
    model = make_causal_model(
        d[["y_first_trip", "hdi", "region", "yr"]].copy(),
        "hdi", "y_first_trip", common_causes=["region", "yr"])
    ident = model.identify_effect(proceed_when_unidentifiable=True)
    est = estimate_dml(model, ident)
    print(f"\nDoWhy battery target — pooled LinearDML('auto'), ATE per "
          f"+{CONTRAST:.2f} HDI = {float(est.value) * CONTRAST:+.4f} "
          f"(= estimate.py rung 3); {num_sims} simulations per refuter")

    cfg = RefuteConfig(num_simulations=num_sims)
    results = run_all_refutations(model, ident, est, config=cfg, out_dir=OUT_DIR)

    beta, se, ci, p, dof = pooled_ols(df)
    rv = robustness_value(beta / se, dof)
    print(f"\nCinelli-Hazlett RV_1 (pooled rung-1 OLS, t={beta / se:.2f}, "
          f"dof={dof}): {rv:.3f} — a confounder needs partial-R^2 >= {rv:.1%} "
          f"with BOTH treatment and outcome to drive the pooled estimate to 0")
    return results


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run both pre-registered refutation batteries on the live warehouse.")
    p.add_argument("--num-sims", type=int, default=RefuteConfig().num_simulations,
                   help="simulations per DoWhy refuter (pre-registered default: 100)")
    args = p.parse_args()

    df = design_frame()
    in_scope = identification_gate(df, verbose=False)
    print(f"-> in-scope regions: {', '.join(in_scope)}")

    design = run_design_battery(df, in_scope)
    battery = run_dowhy_battery(df, args.num_sims)

    hard_fail = [r.name for r in design + battery if r.passed is False]
    print(f"\n{'ALL CHECKS PASS' if not hard_fail else 'FAILURES: ' + ', '.join(hard_fail)}"
          f" ({sum(r.passed is True for r in design + battery)} pass / "
          f"{sum(r.passed is None for r in design + battery)} pending/NA / "
          f"{len(hard_fail)} fail)")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
