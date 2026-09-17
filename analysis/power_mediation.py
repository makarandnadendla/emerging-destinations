"""
Power gate for the trip-order mediation decomposition — RESULT: NOT PURSUED.

QUESTION: can we decompose the HDI -> remoteness association into a direct
effect and a component mediated by repeat-trip behavior (HDI -> trip order ->
remoteness)? Trip order is post-treatment, so the decomposition (hybrid
product-of-coefficients: within-region a-path x within-user FE b-path, or
DoWhy's mediation.two_stage_regression) is only worth running if the estimate
is STABLE at our sample size. This script is the pre-analysis gate: a Monte
Carlo calibrated to the real data structure (real origins, real HDI values,
real trip-count joint — cluster bootstrap; only outcomes are synthesized from
known parameters), measuring the sampling distribution of the hybrid NIE at
1x/2x/4x/8x our cohort.

DECISION RECORD (run 2026-09, 300 reps/cell, cohort n=2,234 with HDI):
    Calibration:  a-path (HDI -> mean trip order, within-region, origin-
                  clustered) = +1.067, se 0.623, p=.09 — and spec-fragile
                  (roughly halves without year dummies). Within-user b-path
                  = +0.0084/trip across all orders (+0.033 first step only —
                  the deepening saturates). Plausible NIE per +0.10 HDI:
                  ~0.001-0.004 remoteness units (~1-4% of the East Asia vs
                  Europe gap of 0.085).
    Simulation:   at our n, power 17-20% and the NIE estimate's SIGN is a
                  coin flip (50-53% positive). At 8x our cohort, power is
                  still <=42%. MDE at 80% power is 2-7x the plausible effect.
    DECISION:     the mediation decomposition is NOT investigated further at
                  this time — the estimate is unstable at our sample size and
                  reporting it would be irresponsible. What survives: the
                  within-user dose-response (b-path) as a standalone
                  descriptive finding; the a-path reported descriptively with
                  its CI; and the FIRST-TRIP headline outcome (a controlled
                  direct effect at order = 1, selection-free because everyone
                  has a first trip) as the design-based answer to repeat-trip
                  contamination. Revisit only with a substantially larger
                  cohort (the curves suggest >~30x) or an external mediator
                  measure.

Usage:
    uv run python analysis/power_mediation.py                # full gate (300 reps)
    uv run python analysis/power_mediation.py --reps 50      # quick re-check
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
    from analysis.estimate import REGIONS
    from analysis.sensitivity import _adjusted_coef as adjusted_coef
except ImportError:
    from estimate import REGIONS
    from sensitivity import _adjusted_coef as adjusted_coef

WAREHOUSE = "data/warehouse_japan.duckdb"
SEED = 20240608
GAMMA = 0.10          # direct HDI effect in the synthetic DGP (irrelevant to NIE precision)


# --------------------------------------------------------------------------- #
# Calibration: real a-path, real b-path, real noise moments
# --------------------------------------------------------------------------- #
def calibrate() -> dict:
    con = duckdb.connect(WAREHOUSE, read_only=True)
    trips = con.execute("""
      WITH p AS (SELECT ud.user_id_hash, ph.taken_ts, cr.remoteness_norm
                 FROM photos ph JOIN user_destination ud USING(user_id_hash)
                 JOIN cell_remoteness cr ON cr.h3_r6 = ph.h3_r6
                 WHERE ph.taken_ts IS NOT NULL
                   AND EXTRACT(year FROM ph.taken_ts) BETWEEN 2012 AND 2019),
      g AS (SELECT *, CASE WHEN date_diff('day', LAG(taken_ts) OVER
                (PARTITION BY user_id_hash ORDER BY taken_ts), taken_ts) > 30
                THEN 1 ELSE 0 END AS new_trip FROM p),
      t AS (SELECT *, SUM(new_trip) OVER (PARTITION BY user_id_hash
                ORDER BY taken_ts ROWS UNBOUNDED PRECEDING) AS trip_id FROM g)
      SELECT user_id_hash, trip_id, AVG(remoteness_norm) AS y
      FROM t GROUP BY 1,2
    """).df()
    uf = con.execute("""SELECT user_id_hash, origin_iso, trip_year, hdi
                        FROM user_features WHERE hdi IS NOT NULL""").df()
    con.close()
    uf["region"] = uf.origin_iso.map(REGIONS).fillna("Other")

    n_trips = trips.groupby("user_id_hash").trip_id.max().add(1).rename("n_trips")
    u = uf.merge(n_trips, on="user_id_hash")
    u["M"] = (u.n_trips - 1) / 2.0        # mediator: user's mean 0-indexed trip order

    d = u.copy(); d["yr"] = d.trip_year.astype(int).astype(str)
    a, a_se, a_ci, a_p, _ = adjusted_coef(
        d, outcome="M", treatment="hdi", numeric=[], categorical=["region", "yr"],
        cluster_col="origin_iso", ci_level=0.95)

    tt = trips.merge(uf[["user_id_hash", "hdi"]], on="user_id_hash")
    rep = tt.groupby("user_id_hash").trip_id.transform("max") >= 1
    r = tt[rep].copy()
    r["kd"] = r.trip_id - r.groupby("user_id_hash").trip_id.transform("mean")
    r["yd"] = r.y - r.groupby("user_id_hash").y.transform("mean")
    delta = float((r.kd * r.yd).sum() / (r.kd ** 2).sum())
    sd_eps = float((r.yd - delta * r.kd).std())

    print(f"calibration: users={len(u):,} (repeaters {(u.n_trips>=2).sum():,})")
    print(f"  a-path (+1.0 HDI, region+year adj, origin-clustered): "
          f"{a:+.3f}  se={a_se:.3f}  95% CI [{a_ci[0]:+.3f},{a_ci[1]:+.3f}]  p={a_p:.3f}")
    print(f"  within-user b-path (all orders): {delta:+.4f}/trip   trip-noise sd={sd_eps:.4f}")
    return {"users": u, "a": a, "delta_fit": delta, "sd_eps": sd_eps, "sd_u": 0.14}


# --------------------------------------------------------------------------- #
# Monte Carlo: cluster bootstrap of the real structure, synthetic outcomes
# --------------------------------------------------------------------------- #
def simulate(cal: dict, reps: int, sizes: tuple, deltas: tuple) -> None:
    u = cal["users"].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    origin_of, regions = u.origin_iso.values, u.region.values
    by_origin = {o: np.flatnonzero(origin_of == o) for o in np.unique(origin_of)}
    origins_by_region = {r: sorted(set(origin_of[regions == r])) for r in np.unique(regions)}

    def one_rep(mult: int, delta_true: float):
        idx_parts = []
        for r_, origs in origins_by_region.items():
            for o in rng.choice(origs, size=len(origs) * mult, replace=True):
                pool = by_origin[o]
                idx_parts.append(rng.choice(pool, size=len(pool), replace=True))
        idx = np.concatenate(idx_parts)
        hdi, M = u.hdi.values[idx], u.M.values[idx]
        nt = u.n_trips.values[idx].astype(int)
        # a-path on the resampled joint (region dummies; year omitted — this is
        # the source of the ~2x attenuation vs the year-adjusted a, i.e. the
        # a-path's spec fragility; the power verdict holds under both)
        D = pd.get_dummies(u.region.values[idx], drop_first=True).values.astype(float)
        X = np.column_stack([np.ones(len(idx)), hdi, D])
        beta, *_ = np.linalg.lstsq(X, M, rcond=None)
        res = M - X @ beta
        XtXi = np.linalg.pinv(X.T @ X)
        cl = np.repeat(np.arange(len(idx_parts)), [len(p) for p in idx_parts])
        meat = np.zeros((X.shape[1],) * 2)
        for g_ in np.unique(cl):
            m_ = cl == g_
            sg = X[m_].T @ res[m_]; meat += np.outer(sg, sg)
        G = len(idx_parts)
        V = (G / (G - 1)) * XtXi @ meat @ XtXi
        a_hat, a_se = beta[1], np.sqrt(max(V[1, 1], 0))
        # synthetic trips, within-user FE b-path on repeaters
        rep_mask = nt >= 2
        ntr = nt[rep_mask]
        uid = np.repeat(np.arange(rep_mask.sum()), ntr)
        k = np.concatenate([np.arange(n) for n in ntr])
        u_i = GAMMA * hdi[rep_mask] + rng.normal(0, cal["sd_u"], rep_mask.sum())
        y = u_i[uid] + delta_true * k + rng.normal(0, cal["sd_eps"], len(k))
        kbar = np.bincount(uid, k) / np.bincount(uid)
        ybar = np.bincount(uid, y) / np.bincount(uid)
        kd, yd = k - kbar[uid], y - ybar[uid]
        Skk = (kd ** 2).sum()
        d_hat = (kd * yd).sum() / Skk
        d_se = np.sqrt(((yd - d_hat * kd) ** 2).sum() / (len(k) - rep_mask.sum() - 1) / Skk)
        nie = a_hat * d_hat
        nie_se = np.sqrt(a_hat**2*d_se**2 + d_hat**2*a_se**2 + a_se**2*d_se**2)
        return nie, nie_se

    print(f"\n{'d_true':>7} {'size':>5} {'n_users':>8} | {'true NIE':>9} {'mean est':>9} "
          f"{'emp SD':>8} {'power':>6} {'sign+':>6} {'MDE80':>8}")
    worst_stable = True
    for delta_true in deltas:
        for mult in sizes:
            out = np.array([one_rep(mult, delta_true) for _ in range(reps)])
            nie, se = out[:, 0], out[:, 1]
            power = float(np.mean(np.abs(nie / np.where(se > 0, se, np.inf)) > 1.96))
            signp = float(np.mean(nie > 0))
            mde = 2.8 * float(np.median(se))
            print(f"{delta_true:>7.4f} {mult:>4}x {len(u)*mult:>8,} | {cal['a']*delta_true:>9.4f} "
                  f"{nie.mean():>9.4f} {nie.std():>8.4f} {power:>6.0%} {signp:>6.0%} {mde:>8.4f}")
            if mult == 1 and (power < 0.8 or min(signp, 1 - signp) > 0.2):
                worst_stable = False
    print("\nGATE (pre-stated): pursue mediation only if, at 1x our cohort, power >= 80%"
          "\nand the NIE sign is stable. VERDICT: "
          + ("PASS" if worst_stable else "FAIL — decomposition not pursued at this time."))


def main() -> int:
    p = argparse.ArgumentParser(description="Power gate for the trip-order mediation NIE.")
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    args = p.parse_args()
    cal = calibrate()
    simulate(cal, args.reps, tuple(args.sizes), deltas=(cal["delta_fit"], 0.033))
    return 0


if __name__ == "__main__":
    sys.exit(main())
