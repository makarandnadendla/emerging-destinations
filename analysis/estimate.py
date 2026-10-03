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
  * SECONDARY (pooled): an ESTIMATOR-COMPLEXITY LADDER on one frame —
      rung 1: pooled OLS on HDI + region + year dummies, origin-clustered
              (one-step adjustment; no orthogonalization or cross-fitting);
      rung 2: LinearDML with plain LINEAR nuisances (Robinson partialling-out,
              cross-fit at DML_CV; zero model selection);
      rung 3: LinearDML with 'auto' nuisances (pre-registered secondary);
      rung 4: Optuna-selected nuisances (robustness) — runs automatically
              whenever tune_dml.py's tuning JSON exists, else skipped with a note.
    Rungs 2-4 are estimated THROUGH DOWHY (method name
    'backdoor.econml.dml.LinearDML' via estimate_dml, nuisances from
    dml_nuisance_models — both defined here and imported by the refute.py
    battery), so the pooled secondary is the very CausalModel/estimate object
    the battery attacks, with the backdoor adjustment set identified from the
    graph that assumptions_dag.py owns.
    Adjacent-rung differences isolate what cross-fitting and flexible
    nuisances each contribute. Continuous HDI; region dummies in W. Pooled ATE
    is dominated by regions with real HDI variation (DML self-localizes: the
    residualized treatment is ~0 where HDI doesn't vary) — labeled as such.

Design decisions carried from the EDA + conversation record:
  * Outcome Y = FIRST-TRIP photo-weighted mean cell remoteness (headline);
    pooled all-trips Y and drop-stated/modal-conflicts are sensitivity specs.
  * y_vin ROBUSTNESS ROW (pre-committed when yardstick_vintage.py returned
    MATERIAL, cell rho=0.679): the first-trip outcome re-scored with the
    2015-01-01 OSM vintage yardstick (same universe + formula as
    05_cell_remoteness.sql). Runs whenever the vintage POI parquet exists;
    reported with the mapping-completeness caveat (45.5% of cells had zero
    mapped POIs in 2015; user-level r=0.973 between rulers).
  * Effects reported per +0.10 HDI. Trip gap rule 30d (sensitivities 14/60).
  * U-shape / between-region contrasts are DESCRIPTIVE results, reported
    separately — never folded into the causal coefficients.

Usage:
    uv run python analysis/estimate.py --selftest    # synthetic machinery check
    uv run python analysis/estimate.py --run         # first real estimates
        # (the Optuna-tuned pooled-DML rung joins automatically whenever
        #  analysis/outputs/dml_nuisance_tuning.json exists — see tune_dml.py)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd

try:
    from analysis.assumptions_dag import build_model, describe_estimand, identify
    from analysis.config import CFG
except ImportError:
    from assumptions_dag import build_model, describe_estimand, identify
    from config import CFG

# Defaults live in analysis/config.yaml (single source of truth).
WAREHOUSE = CFG["warehouse"]
SEED = CFG["seed"]
GAP_DAYS = CFG["estimation"]["gap_days"]        # trip rule (sensitivities: 14 / 60)
CONTRAST = CFG["estimation"]["contrast"]        # effects reported per +0.10 HDI

# --- the stated scope condition (pre-registered thresholds) -----------------
MIN_HDI_SD = CFG["estimation"]["min_hdi_sd"]            # below: HDI effectively constant
MIN_REGION_USERS = CFG["estimation"]["min_region_users"]  # below: slope not credible

# Vintage-ruler robustness outcome (pre-committed consequence of the MATERIAL
# yardstick-vintage verdict — see analysis/yardstick_vintage.py).
VINTAGE = CFG["vintage"]


# --------------------------------------------------------------------------- #
# Shared estimation machinery. Estimators live HERE; assumptions live in
# assumptions_dag.py; attacks live in refute.py (which imports from here —
# one-way: assumptions_dag <- estimate <- refute).
# --------------------------------------------------------------------------- #
# Cross-fitting partitions for the DML (analysis/config.yaml): shared by the
# ladder rungs, the refutation battery and the nuisance tuner's CV, so the
# fold scheme cannot diverge.
DML_CV = CFG["dml"]["cv"]

# Optuna-tuned nuisance spec (written by tune_dml.py, read by the 'tuned' rung).
TUNING_JSON = Path(__file__).resolve().parent / "outputs" / "dml_nuisance_tuning.json"


def dml_nuisance_models(spec: str = "auto") -> dict:
    """Stage-A DML nuisance models (user decision, 2026-07): both nuisances
    default to econml's 'auto' model selection.

    The treatment stays CONTINUOUS (raw HDI, not binned). With a continuous
    treatment the discrete-treatment DR learners (and any propensity classifier)
    don't apply; the estimator is LinearDML, where model_t is the treatment
    model — the continuous analogue of the propensity — and model_y is the
    outcome model. NOTE: the linear final stage estimates a single slope in
    HDI; the U-shape found in the assumption-audit EDA is handled by a
    binned/spline sensitivity spec, not by the headline.

    spec picks the rung on the estimator-complexity ladder:
      'auto'   — pre-registered default (econml's CV bake-off; analysis/config.yaml)
      'linear' — plain OLS nuisances: the original Robinson partialling-out
                 estimator, still cross-fit. Zero model selection.
      'tuned'  — Optuna-selected nuisances (predictive CV only) from
                 analysis/outputs/dml_nuisance_tuning.json (tune_dml.py);
                 reported as a robustness row, never the default."""
    if spec == "auto":
        default = CFG["dml"]["nuisances"]
        return {"model_y": default, "model_t": default}
    if spec == "linear":
        from sklearn.linear_model import LinearRegression
        return {"model_y": LinearRegression(), "model_t": LinearRegression()}
    if spec == "tuned":
        import json
        if not TUNING_JSON.exists():
            raise FileNotFoundError(
                f"{TUNING_JSON} not found — run `uv run python analysis/tune_dml.py` first.")
        try:
            from analysis.tune_dml import build_model as build_sklearn
        except ImportError:
            from tune_dml import build_model as build_sklearn
        best = json.loads(TUNING_JSON.read_text())
        return {k: build_sklearn(best[k]["family"], best[k]["params"])
                for k in ("model_y", "model_t")}
    raise ValueError(f"unknown nuisance spec {spec!r}")


def estimate_dml(model, identified_estimand,
                 control_value=0, treatment_value=1,
                 confidence_intervals: bool = False,
                 fit_params: dict | None = None, **init_overrides):
    """Debiased-ML estimate via econml's LinearDML (DoWhy-wrapped so the
    refutation battery can re-run it): 'auto' nuisances + linear final stage
    with statsmodels inference. Continuous treatment.

    The defaults (contrast 0 -> 1, no CIs, no fit_params) match DoWhy's own
    and keep the battery path unchanged; run_pooled_dml passes the +CONTRAST
    treatment_value and confidence_intervals=True for its reporting, and
    diagnostics.py passes fit_params={"cache_values": True} so the fitted
    estimator exposes its cross-fitted first-stage residuals."""
    init = {**dml_nuisance_models(), "cv": DML_CV, "random_state": SEED}
    init.update(init_overrides)
    return model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        control_value=control_value, treatment_value=treatment_value,
        confidence_intervals=confidence_intervals,
        method_params={"init_params": init, "fit_params": fit_params or {}},
    )


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


def adjusted_coef(d: pd.DataFrame, outcome: str, treatment: str,
                  numeric: list[str], categorical: list[str],
                  cluster_col, ci_level: float):
    """Return (beta, se, (ci_lo, ci_hi), p, dof) for `treatment`, with cluster-
    robust SEs when cluster_col is given (CR1), else HC1. The Tier-1 headline,
    rung 1 and the design refutations all run on this one estimator."""
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
    """Analysis frame: one row per cohort user with first-trip + pooled outcomes.

    Also computes y_vin — the first-trip outcome re-scored with the vintage OSM
    yardstick (identical score formula and fixed cell universe as
    yardstick_vintage.py) — whenever the vintage POI parquet exists; NULL (and
    the robustness row is skipped) otherwise."""
    vin_pq = Path(VINTAGE["pois_parquet"])
    has_vin = vin_pq.exists()
    con = duckdb.connect(WAREHOUSE, read_only=True)
    if has_vin:
        con.execute("INSTALL h3 FROM community; LOAD h3;")
        vin_ctes = f"""
        vin AS (
            SELECT h3_latlng_to_cell(lat, lon, {VINTAGE['h3_res']}) AS h3_r6,
                   COUNT(*) AS n
            FROM read_parquet('{vin_pq.as_posix()}') GROUP BY 1
        ),
        rv AS (
            SELECT cr.h3_r6, -ln(1 + COALESCE(vin.n, 0)) AS raw_vin
            FROM cell_remoteness cr LEFT JOIN vin USING (h3_r6)
        ),
        vb AS (SELECT min(raw_vin) AS lo, max(raw_vin) AS hi FROM rv),
        rvn AS (
            SELECT h3_r6, CASE WHEN vb.hi = vb.lo THEN 0.0
                               ELSE (raw_vin - vb.lo) / (vb.hi - vb.lo) END AS r_vin
            FROM rv CROSS JOIN vb
        )"""
    else:
        vin_ctes = """
        rvn AS (SELECT h3_r6, NULL::DOUBLE AS r_vin FROM cell_remoteness)"""
    df = con.execute(f"""
        WITH {vin_ctes},
        p AS (
            SELECT ud.user_id_hash, ph.taken_ts, cr.remoteness_norm, rvn.r_vin
            FROM photos ph
            JOIN user_destination ud USING (user_id_hash)
            JOIN cell_remoteness cr ON cr.h3_r6 = ph.h3_r6
            LEFT JOIN rvn ON rvn.h3_r6 = ph.h3_r6
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
                   SUM(r_vin * (trip_id = 0)::INT)
                       / NULLIF(SUM((trip_id = 0)::INT), 0) AS y_vin,
                   SUM((trip_id = 0)::INT) AS trip1_photos
            FROM t GROUP BY 1
        )
        SELECT uf.user_id_hash, uf.origin_iso, uf.trip_year,
               uf.y_mean_remoteness AS y_pooled, uf.hdi, uf.agree_flag,
               tr.y_first_trip, tr.y_vin, tr.n_trips, tr.trip1_photos
        FROM user_features uf
        LEFT JOIN trips tr USING (user_id_hash)
    """).df()
    con.close()
    if not has_vin:
        print(f"   [vintage ruler absent — {vin_pq} not found; y_vin row will be skipped]")

    df["region"] = df.origin_iso.map(REGIONS).fillna("Other")
    # DuckDB row order is nondeterministic; pin it so seeded fold splits
    # (tune_dml CV, LinearDML cross-fitting) are exactly reproducible.
    df = df.sort_values("user_id_hash").reset_index(drop=True)
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
# Tier 2 — pooled secondary: an estimator-complexity ladder on one frame.
# Rungs (simple -> complex): OLS adjustment, LinearDML(linear), LinearDML(auto),
# optionally LinearDML(tuned). Agreement across rungs shows the pooled number
# is not an estimator artifact; disagreement localizes where complexity matters.
# --------------------------------------------------------------------------- #
def run_pooled_ols(df: pd.DataFrame, outcome: str, label: str) -> float:
    """Simplest rung: OLS of Y on HDI + region + year dummies, origin-clustered
    SEs — one-step parametric adjustment, no orthogonalization or cross-fitting.
    Same machinery as the within-region headline (sensitivity._adjusted_coef)."""
    d = df.dropna(subset=[outcome]).rename(columns={outcome: "_y"}).copy()
    d["yr"] = d.trip_year.astype(int).astype(str)
    beta, se, ci, p, dof = adjusted_coef(
        d, outcome="_y", treatment="hdi", numeric=[], categorical=["region", "yr"],
        cluster_col="origin_iso", ci_level=0.95)
    print(f"\n--- SECONDARY rung 1 (pooled OLS, region+year adj, origin-clustered): {label} ---")
    print(f"  ATE per +{CONTRAST:.2f} HDI: {beta * CONTRAST:+.4f} "
          f"[95% CI {ci[0] * CONTRAST:+.4f}, {ci[1] * CONTRAST:+.4f}]  "
          f"p={p:.3f} (dof={dof}, n={len(d):,})")
    return beta * CONTRAST


def fit_pooled_dml(df: pd.DataFrame, outcome: str, nuisances: str = "auto",
                   cache_values: bool = False):
    """Fit one pooled-ladder DML rung and return (DoWhy estimate, frame used).

    THE single construction path for rungs 2-4: run_pooled_dml reports from it
    and diagnostics.py inspects the same fitted object (cache_values=True
    makes econml keep the cross-fitted first-stage residuals on the
    estimator, exposed as .residuals_)."""
    d = df.dropna(subset=[outcome])
    model = build_model(d, outcome)       # assumptions_dag pins the adjustment set
    ident = identify(model)
    est = estimate_dml(model, ident, control_value=0.0, treatment_value=CONTRAST,
                       confidence_intervals=True,
                       fit_params={"cache_values": True} if cache_values else None,
                       **dml_nuisance_models(nuisances))
    return est, d


def run_pooled_dml(df: pd.DataFrame, outcome: str, label: str,
                   nuisances: str = "auto") -> float:
    """Rungs 2-4 run through DoWhy's interface (method name
    'backdoor.econml.dml.LinearDML', via estimate_dml above): the identical
    CausalModel -> identify_effect -> estimate chain the refutation battery
    re-fits, so the printed secondary IS the refuted object and the backdoor
    adjustment set is identified from the graph (assumptions_dag.build_model)
    rather than assumed per call site."""
    est, d = fit_pooled_dml(df, outcome, nuisances)
    ate = float(est.value)
    ci = np.asarray(est.get_confidence_intervals(), dtype=float).reshape(2, -1)
    lo, hi = float(ci[0].mean()), float(ci[1].mean())
    rung = {"linear": "rung 2 (LinearDML, linear nuisances — Robinson)",
            "auto": "rung 3 (LinearDML, 'auto' nuisances — pre-registered)",
            "tuned": "rung 4 (LinearDML, Optuna-tuned nuisances — robustness)"}[nuisances]
    print(f"\n--- SECONDARY {rung}: {label} ---")
    print(f"  ATE per +{CONTRAST:.2f} HDI: {ate:+.4f} "
          f"[95% CI {lo:+.4f}, {hi:+.4f}]  (n={len(d):,}; dominated by regions "
          f"with real HDI variation — DML self-localizes)")
    return ate


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

    # (d) simple rungs of the pooled ladder recover a known effect
    #     (true slope 0.3/unit HDI -> +0.0300 per +0.10)
    nl = 1500
    regl = rng.choice(["R1", "R2", "R3"], nl)
    hdil = (0.7 + 0.1 * (regl == "R2") + 0.15 * (regl == "R3")
            + rng.normal(0, 0.05, nl))
    dfl = pd.DataFrame({"region": regl, "origin_iso": regl,
                        "trip_year": rng.integers(2012, 2020, nl), "hdi": hdil,
                        "y": 0.2 + 0.3 * hdil + rng.normal(0, 0.05, nl)})
    b_ols = run_pooled_ols(dfl, "y", "selftest ladder DGP")
    b_lin = run_pooled_dml(dfl, "y", "selftest ladder DGP", nuisances="linear")
    ok_d = abs(b_ols - 0.03) < 0.005 and abs(b_lin - 0.03) < 0.005
    print(f"[selftest] ladder rungs 1/2: OLS {b_ols:+.4f}, linear-DML {b_lin:+.4f} "
          f"(true +0.0300)  -> {'OK' if ok_d else 'FAIL'}")

    ok = ok_a and ok_b and ok_c and ok_d
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
        if df.y_vin.notna().any():
            run_within_region(
                df, "y_vin",
                f"ROBUSTNESS: first-trip outcome, {VINTAGE['label']} vintage "
                "yardstick (completeness-limited ruler — yardstick_vintage.py)",
                in_scope)
        # One identification statement for the pooled tier (identical across
        # rungs 2-4 — the ladder varies only the nuisance models).
        d1 = df.dropna(subset=["y_first_trip"])
        print("\nIDENTIFICATION (pooled tier — DoWhy backdoor, "
              "adjustment set from assumptions_dag.py):")
        print("  " + describe_estimand(identify(build_model(d1, "y_first_trip")))
              .replace("\n", "\n  "))
        run_pooled_ols(df, "y_first_trip", "first-trip outcome")
        run_pooled_dml(df, "y_first_trip", "first-trip outcome", nuisances="linear")
        run_pooled_dml(df, "y_first_trip", "first-trip outcome")
        if TUNING_JSON.exists():
            run_pooled_dml(df, "y_first_trip", "first-trip outcome", nuisances="tuned")
        else:
            print(f"\n  [rung 4 skipped — {TUNING_JSON.name} not found; "
                  f"run `uv run python analysis/tune_dml.py` to add it]")
        return 0
    print("Nothing to do: pass --selftest or --run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
