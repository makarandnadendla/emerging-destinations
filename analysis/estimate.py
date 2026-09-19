"""
Stage-A headline estimation — within-region HDI effects under a STATED SCOPE CONDITION.

SCOPE CONDITION (a design element, not a hidden weakness):
  Causal within-region HDI effects are estimated ONLY for regions that pass a
  pre-stated identification gate, computed from the data and printed with every
  run:
      (a) within-region user-level HDI sd >= MIN_HDI_SD   (0.02), and
      (b) region cohort size            >= MIN_REGION_USERS (50).
  Regions failing the gate (the EDA showed North America sd=0.004 and Oceania
  sd=0.003 — HDI is structurally constant there) are reported DESCRIPTIVELY
  ONLY, clearly labeled. Rationale: region ~ 70% of HDI variance (see the
  assumption-audit EDA); the identifying variation is the within-region
  residual, which exists only in some regions. No estimator can conjure
  identification where treatment does not vary — the gate states where it does.

Estimators:
  * HEADLINE (per in-scope region): OLS of Y on HDI + trip-year dummies within
    the region, ORIGIN-CLUSTERED SEs (reuses sensitivity._adjusted_coef so the
    negative control runs on the identical machinery). Fragility is surfaced by
    leave-one-origin-out (LOO): slope range + sign-flip flag. Two-origin regions
    (East Asia = CHN vs KOR) cannot run LOO — stated in the output.
  * SECONDARY (pooled): econml LinearDML, nuisances 'auto'
    (refute.dml_nuisance_models — shared with the DoWhy battery via
    refute.estimate_dml). Continuous HDI; region dummies in W. Pooled ATE is
    dominated by regions with real HDI variation (DML self-localizes: the
    residualized treatment is ~0 where HDI doesn't vary) — labeled as such.

Design decisions carried from the EDA + conversation record:
  * Outcome Y = FIRST-TRIP photo-weighted mean cell remoteness (headline);
    pooled all-trips Y and drop-stated/modal-conflicts are sensitivity specs.
  * Effects reported per +0.10 HDI. Trip gap rule 30d (sensitivities 14/60).
  * U-shape / between-region contrasts are DESCRIPTIVE results, reported
    separately — never folded into the causal coefficients.

Usage:
    uv run python analysis/estimate.py --selftest    # synthetic machinery check
    uv run python analysis/estimate.py --run         # first real estimates
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd

try:
    from analysis.refute import dml_nuisance_models
    from analysis.sensitivity import _adjusted_coef as adjusted_coef
except ImportError:
    from refute import dml_nuisance_models
    from sensitivity import _adjusted_coef as adjusted_coef

WAREHOUSE = "data/warehouse_japan.duckdb"
SEED = 20240608
GAP_DAYS = 30            # trip segmentation rule (sensitivities: 14 / 60)
CONTRAST = 0.10          # effects reported per +0.10 HDI

# --- the stated scope condition (pre-registered thresholds) -----------------
MIN_HDI_SD = 0.02        # below this, within-region HDI is effectively constant
MIN_REGION_USERS = 50    # below this, a region-specific slope is not credible

# Origin -> region (Stage-A pinned artifact; promoted from the assumption EDA).
REGIONS = {}
for _iso in "TWN CHN HKG KOR MAC MNG".split():                      REGIONS[_iso] = "East Asia"
for _iso in "THA SGP MYS IDN PHL VNM KHM LAO MMR BRN TLS".split():  REGIONS[_iso] = "Southeast Asia"
for _iso in "IND PAK BGD LKA NPL BTN MDV AFG KAZ UZB KGZ TJK TKM".split():
    REGIONS[_iso] = "South/Central Asia"
for _iso in ("GBR FRA DEU ITA ESP NLD BEL CHE AUT SWE NOR DNK FIN IRL PRT POL "
             "CZE SVK HUN GRC ROU BGR HRV SVN SRB EST LVA LTU UKR RUS BLR ISL "
             "LUX MLT CYP MKD ALB BIH MNE MDA").split():            REGIONS[_iso] = "Europe"
for _iso in "USA CAN".split():                                       REGIONS[_iso] = "North America"
for _iso in ("MEX BRA ARG CHL COL PER VEN ECU URY PRY BOL CRI PAN GTM HND SLV "
             "NIC DOM CUB JAM TTO BRB BHS GUY SUR").split():         REGIONS[_iso] = "Latin America"
for _iso in "AUS NZL FJI PNG NCL PYF GUM".split():                   REGIONS[_iso] = "Oceania"
for _iso in ("TUR ISR SAU ARE QAT KWT BHR OMN JOR LBN IRQ IRN YEM SYR EGY ZAF "
             "MAR DZA TUN LBY KEN NGA GHA ETH TZA UGA SEN CIV CMR ZWE ZMB MUS "
             "MDG MOZ AGO BWA NAM RWA SDN").split():                 REGIONS[_iso] = "Mid-East & Africa"


def load_frame(gap_days: int = GAP_DAYS) -> pd.DataFrame:
    """Analysis frame: one row per cohort user with first-trip + pooled outcomes."""
    con = duckdb.connect(WAREHOUSE, read_only=True)
    df = con.execute(f"""
        WITH p AS (
            SELECT ud.user_id_hash, ph.taken_ts, cr.remoteness_norm
            FROM photos ph
            JOIN user_destination ud USING (user_id_hash)
            JOIN cell_remoteness cr ON cr.h3_r6 = ph.h3_r6
            WHERE ph.taken_ts IS NOT NULL
              AND EXTRACT(year FROM ph.taken_ts) BETWEEN 2012 AND 2019
        ),
        g AS (
            SELECT *, CASE WHEN date_diff('day', LAG(taken_ts) OVER
                      (PARTITION BY user_id_hash ORDER BY taken_ts), taken_ts)
                      > {gap_days} THEN 1 ELSE 0 END AS new_trip
            FROM p
        ),
        t AS (
            SELECT *, SUM(new_trip) OVER (PARTITION BY user_id_hash
                      ORDER BY taken_ts ROWS UNBOUNDED PRECEDING) AS trip_id
            FROM g
        ),
        trips AS (
            SELECT user_id_hash,
                   MAX(trip_id) + 1 AS n_trips,
                   SUM(remoteness_norm * (trip_id = 0)::INT)
                       / NULLIF(SUM((trip_id = 0)::INT), 0) AS y_first_trip,
                   SUM((trip_id = 0)::INT) AS trip1_photos
            FROM t GROUP BY 1
        )
        SELECT uf.user_id_hash, uf.origin_iso, uf.trip_year,
               uf.y_mean_remoteness AS y_pooled, uf.hdi, uf.agree_flag,
               tr.y_first_trip, tr.n_trips, tr.trip1_photos
        FROM user_features uf
        LEFT JOIN trips tr USING (user_id_hash)
    """).df()
    con.close()

    df["region"] = df.origin_iso.map(REGIONS).fillna("Other")
    n0 = len(df)
    df = df.dropna(subset=["hdi", "y_first_trip"]).copy()
    print(f"-> frame: {n0:,} cohort users; {len(df):,} with HDI + first-trip Y "
          f"(dropped: {n0 - len(df):,}, incl. Taiwan's missing HDI)")
    return df


# --------------------------------------------------------------------------- #
# The identification gate (the stated scope condition, computed + printed)
# --------------------------------------------------------------------------- #
def identification_gate(df: pd.DataFrame, verbose: bool = True) -> list[str]:
    g = (df.groupby("region")
           .agg(users=("hdi", "size"), origins=("origin_iso", "nunique"),
                hdi_sd=("hdi", "std"))
           .fillna(0.0))
    g["sd_ok"] = g.hdi_sd >= MIN_HDI_SD
    g["n_ok"] = g.users >= MIN_REGION_USERS
    g["in_scope"] = g.sd_ok & g.n_ok
    if verbose:
        print("\nSCOPE CONDITION — causal estimation only where HDI varies within region")
        print(f"  gate: HDI sd >= {MIN_HDI_SD}  AND  users >= {MIN_REGION_USERS}")
        print(f"  {'region':<20} {'users':>6} {'origins':>8} {'HDI sd':>8}  verdict")
        for reg, r in g.sort_values("users", ascending=False).iterrows():
            why = ("IN SCOPE" if r.in_scope else
                   "descriptive only — " +
                   ("no within-region HDI variation" if not r.sd_ok else
                    "too few users"))
            print(f"  {reg:<20} {int(r.users):>6} {int(r.origins):>8} "
                  f"{r.hdi_sd:>8.3f}  {why}")
    return sorted(g.index[g.in_scope])


# --------------------------------------------------------------------------- #
# Tier 1 — within-region estimates (headline), origin-clustered SEs, LOO
# --------------------------------------------------------------------------- #
def region_estimate(d: pd.DataFrame, outcome: str) -> tuple:
    """(beta, ci) per +CONTRAST HDI, origin-clustered; d = one region's users."""
    dd = d.rename(columns={outcome: "_y"}).copy()
    dd["yr"] = dd.trip_year.astype(int).astype(str)
    beta, se, ci, p, dof = adjusted_coef(
        dd, outcome="_y", treatment="hdi", numeric=[], categorical=["yr"],
        cluster_col="origin_iso", ci_level=0.95)
    return beta * CONTRAST, (ci[0] * CONTRAST, ci[1] * CONTRAST), p, dof


def loo_origins(d: pd.DataFrame, outcome: str) -> str:
    """Leave-one-origin-out slope range; the fragility statement."""
    origins = d.origin_iso.unique()
    if len(origins) < 3:
        return (f"LOO impossible — contrast rests on {len(origins)} origins "
                f"({'/'.join(origins)}); treat as a two-point contrast")
    betas = []
    for o in origins:
        sub = d[d.origin_iso != o]
        if sub.hdi.std() < 0.005 or sub.origin_iso.nunique() < 2:
            continue
        b, _, _, _ = region_estimate(sub, outcome)
        betas.append((o, b))
    if not betas:
        return "LOO impossible after drops"
    bs = [b for _, b in betas]
    flip = (min(bs) < 0 < max(bs))
    worst = min(betas, key=lambda t: t[1]) if flip else max(
        betas, key=lambda t: abs(t[1] - float(np.median(bs))))
    return (f"LOO range [{min(bs):+.4f}, {max(bs):+.4f}]"
            + (f"  SIGN FLIPS (e.g. without {worst[0]})" if flip else "  sign stable"))


def run_within_region(df: pd.DataFrame, outcome: str, label: str,
                      in_scope: list[str], drop_conflicts: bool = False) -> None:
    d = df.copy()
    if drop_conflicts:
        d = d[d.agree_flag != False]          # noqa: E712 (keep NULL-modal users)
    d = d.dropna(subset=[outcome])
    print(f"\n=== {label}  (n={len(d):,}) — effects per +{CONTRAST:.2f} HDI ===")
    for reg in in_scope:
        sub = d[d.region == reg]
        if len(sub) < 30:
            print(f"  {reg:<18} n={len(sub):>4}  (spec subset too small, skipped)")
            continue
        b, ci, p, dof = region_estimate(sub, outcome)
        print(f"  {reg:<18} n={len(sub):>4}  beta={b:+.4f}  "
              f"[95% CI {ci[0]:+.4f}, {ci[1]:+.4f}]  p={p:.3f} (dof={dof})")
        print(f"  {'':<18} {loo_origins(sub, outcome)}")
    # descriptive tier — out-of-scope regions, labeled
    out = sorted(set(d.region.unique()) - set(in_scope))
    if out:
        means = d[d.region.isin(out)].groupby("region")[outcome].agg(["mean", "size"])
        desc = "  ·  ".join(f"{r} {m['mean']:.3f} (n={int(m['size'])})"
                            for r, m in means.iterrows())
        print(f"  [descriptive only, NO causal estimate] {desc}")


# --------------------------------------------------------------------------- #
# Tier 2 — pooled LinearDML secondary (self-localizing; labeled)
# --------------------------------------------------------------------------- #
def run_pooled_dml(df: pd.DataFrame, outcome: str, label: str) -> None:
    from econml.dml import LinearDML
    d = df.dropna(subset=[outcome])
    Y = d[outcome].to_numpy(float)
    T = d.hdi.to_numpy(float)
    W = pd.get_dummies(d[["region"]].assign(yr=d.trip_year.astype(int).astype(str)),
                       drop_first=True, dtype=float)
    est = LinearDML(**dml_nuisance_models(), cv=3, random_state=SEED)
    est.fit(Y, T, X=None, W=W)
    lo, hi = est.ate_interval(T0=0.0, T1=CONTRAST, alpha=0.05)
    print(f"\n--- SECONDARY (pooled LinearDML, 'auto' nuisances): {label} ---")
    print(f"  ATE per +{CONTRAST:.2f} HDI: {est.ate(T0=0.0, T1=CONTRAST):+.4f} "
          f"[95% CI {lo:+.4f}, {hi:+.4f}]  (n={len(d):,}; dominated by regions "
          f"with real HDI variation — DML self-localizes)")


# --------------------------------------------------------------------------- #
def selftest() -> int:
    """(a) gate logic; (b) within-region estimator recovers known slopes;
    (c) pooled LinearDML('auto') + DoWhy-wrapped estimate_dml run."""
    rng = np.random.default_rng(7)

    # (a) gate: one region flat-HDI, one thin, two healthy
    toy = pd.concat([
        pd.DataFrame({"region": "FlatLand", "origin_iso": ["FL1", "FL2"] * 100,
                      "hdi": 0.90 + rng.normal(0, 0.001, 200)}),
        pd.DataFrame({"region": "TinyLand", "origin_iso": ["TL"] * 20,
                      "hdi": rng.uniform(0.6, 0.9, 20)}),
        pd.DataFrame({"region": "GradLand", "origin_iso":
                      rng.choice(["G1", "G2", "G3", "G4"], 300),
                      "hdi": rng.uniform(0.6, 0.95, 300)}),
        pd.DataFrame({"region": "PairLand", "origin_iso":
                      rng.choice(["P1", "P2"], 150),
                      "hdi": rng.uniform(0.7, 0.9, 150)}),
    ])
    got = identification_gate(toy, verbose=False)
    ok_a = got == ["GradLand", "PairLand"]
    print(f"[selftest] gate -> {got}  -> {'OK' if ok_a else 'FAIL'}")

    # (b) within-region slope recovery (true slope 0.5 per unit HDI)
    n = 800
    orig = rng.choice(["A", "B", "C", "D", "E"], n)
    hdi = {"A": 0.65, "B": 0.72, "C": 0.80, "D": 0.88, "E": 0.94}
    d = pd.DataFrame({"region": "GradLand", "origin_iso": orig,
                      "trip_year": rng.integers(2012, 2020, n)})
    d["hdi"] = d.origin_iso.map(hdi) + rng.normal(0, 0.004, n)
    d["y"] = 0.1 + 0.5 * d.hdi + rng.normal(0, 0.05, n)
    b, ci, p, dof = region_estimate(d, "y")
    ok_b = abs(b - 0.5 * CONTRAST) < 0.01
    print(f"[selftest] within-region beta per +{CONTRAST:.2f} = {b:+.4f} "
          f"(true {0.5 * CONTRAST:+.4f})  -> {'OK' if ok_b else 'FAIL'}")
    print(f"[selftest] {loo_origins(d, 'y')}")

    # (c) pooled DML + DoWhy-wrapped path on continuous treatment
    from dowhy import CausalModel
    try:
        from analysis.refute import estimate_dml
    except ImportError:
        from refute import estimate_dml
    # DGP note: treatment needs ample residual variance (sd 0.15 here) or the
    # 'auto' nuisance fit's small overfitting of E[T|W] attenuates the slope.
    w = rng.normal(0, 1, (3000, 2))
    T = 0.7 + 0.10 * w[:, 0] + rng.normal(0, 0.15, 3000)
    Y = 0.2 + 0.1 * w[:, 0] + 0.15 * T + rng.normal(0, 0.08, 3000)
    dfd = pd.DataFrame({"y": Y, "t": T, "w0": w[:, 0], "w1": w[:, 1]})
    m = CausalModel(data=dfd, treatment="t", outcome="y",
                    common_causes=["w0", "w1"])
    e = estimate_dml(m, m.identify_effect(proceed_when_unidentifiable=True))
    ok_c = abs(float(e.value) - 0.15) < 0.05
    print(f"[selftest] DoWhy-wrapped LinearDML('auto'): {float(e.value):+.4f} "
          f"(true +0.15)  -> {'OK' if ok_c else 'FAIL'}")

    ok = ok_a and ok_b and ok_c
    print(f"\nSELF-TEST {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Stage-A scoped within-region estimation.")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--run", action="store_true",
                   help="Produce the first real estimates (headline + sensitivities).")
    p.add_argument("--gap-days", type=int, default=GAP_DAYS)
    args = p.parse_args()
    if args.selftest:
        return selftest()
    if args.run:
        df = load_frame(args.gap_days)
        in_scope = identification_gate(df)
        run_within_region(df, "y_first_trip",
                          "HEADLINE: first-trip outcome, within-region",
                          in_scope)
        run_within_region(df, "y_first_trip",
                          "SENSITIVITY: first-trip, stated-modal conflicts dropped",
                          in_scope, drop_conflicts=True)
        run_within_region(df, "y_pooled",
                          "SENSITIVITY: pooled all-trips outcome",
                          in_scope)
        run_pooled_dml(df, "y_first_trip", "first-trip outcome")
        return 0
    print("Nothing to do: pass --selftest or --run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
