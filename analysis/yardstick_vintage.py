"""
Yardstick-vintage check (the "rho check") — is remoteness rank-stable across OSM vintages?

WHY: cell remoteness is built from ONE modern OSM snapshot while trips span
2012-2019 (SPEC section 4; see the channel-4 "yardstick vintage" discussion).
Because remoteness is min-max normalized -ln(1+POI), it is driven by cell RANKS,
so a historical vintage can only move estimates if it re-RANKS cells. This check
rebuilds the yardstick from an archived Geofabrik snapshot (same tag filter, same
extractor, same H3 res, same cell universe) and measures how much the ruler
actually changes — bounding the payoff of a full per-year rebuild before paying
for it.

Universe is held FIXED at the warehouse's POI-bearing cells (defined by the
current snapshot): the question is whether the ruler re-orders THESE cells, not
whether the universe would differ. Vintage-zero-POI cells score most-remote raw
(-ln(1)=0) and are counted/reported.

PRE-STATED INTERPRETATION CRITERION (committed before the result exists):
  * cell-level Spearman rho >= 0.97  -> vintage effect NEGLIGIBLE: a per-year
    yardstick cannot meaningfully move any estimate; report and move on.
  * rho < 0.97 -> a vintage-matched yardstick enters the robustness table
    (multi-operationalization, CLAUDE.md note 3) as a clearly-labeled spec.
  Secondary (reported, no threshold): user-level correlation between each cohort
  user's outcome under the two yardsticks, and per-region mean shifts — the
  numbers that translate cell churn into estimate-relevant movement.

Usage (after building the vintage POI parquet with the SAME extractor):
    uv run python pipeline/extract/extract_osm_pois.py \\
        --pbf data/cache/japan-150101.osm.pbf \\
        --out data/cache/pois_raw_japan_150101.parquet \\
        --index sparse_file_array,data/cache/nodecache_150101.bin
    uv run python analysis/yardstick_vintage.py \\
        --vintage-pois data/cache/pois_raw_japan_150101.parquet --label 2015-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    from analysis.estimate import REGIONS
except ImportError:
    from estimate import REGIONS

RHO_FLOOR = 0.97   # pre-stated: at/above this, the vintage channel is negligible


def load_cfg() -> dict:
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "pipeline" / "config.yaml",
                              encoding="utf-8"))
    d = cfg["destinations"][cfg["destination"]]
    return {"warehouse": d["warehouse_path"], "h3_res": cfg["h3_res"]}


def main() -> int:
    p = argparse.ArgumentParser(description="OSM yardstick vintage rank-stability check.")
    p.add_argument("--vintage-pois", required=True, type=Path,
                   help="Parquet from extract_osm_pois.py run on the archived PBF.")
    p.add_argument("--label", default="vintage", help="Vintage label for the report.")
    p.add_argument("--out-dir", default="analysis/outputs", type=Path)
    args = p.parse_args()
    if not args.vintage_pois.exists():
        sys.exit(f"ERROR: {args.vintage_pois} not found — run extract_osm_pois.py first.")

    cfg = load_cfg()
    con = duckdb.connect(cfg["warehouse"], read_only=True)
    con.execute("INSTALL h3 FROM community; LOAD h3;")

    # Vintage counts on the FIXED current-universe cells; identical score formula
    # to 05_cell_remoteness.sql (min-max of -ln(1+n) over the same cell set).
    cells = con.execute(f"""
        WITH v AS (
            SELECT h3_latlng_to_cell(lat, lon, {cfg['h3_res']}) AS h3_r6,
                   COUNT(*) AS n
            FROM read_parquet('{args.vintage_pois.as_posix()}')
            GROUP BY 1
        ),
        j AS (
            SELECT cr.h3_r6,
                   cr.poi_count                    AS poi_now,
                   cr.remoteness_norm              AS r_now,
                   COALESCE(v.n, 0)                AS poi_vin,
                   -ln(1 + COALESCE(v.n, 0))       AS raw_vin
            FROM cell_remoteness cr
            LEFT JOIN v USING (h3_r6)
        ),
        b AS (SELECT min(raw_vin) AS lo, max(raw_vin) AS hi FROM j)
        SELECT j.*,
               CASE WHEN b.hi = b.lo THEN 0.0
                    ELSE (j.raw_vin - b.lo) / (b.hi - b.lo) END AS r_vin,
               h3_cell_to_lat(j.h3_r6) AS lat, h3_cell_to_lng(j.h3_r6) AS lng
        FROM j CROSS JOIN b
    """).df()

    # Cohort outcomes under both rulers (photo-weighted, as in 09_user_features).
    users = con.execute(f"""
        WITH v AS (
            SELECT h3_latlng_to_cell(lat, lon, {cfg['h3_res']}) AS h3_r6, COUNT(*) AS n
            FROM read_parquet('{args.vintage_pois.as_posix()}') GROUP BY 1
        ),
        rv AS (
            SELECT cr.h3_r6, -ln(1 + COALESCE(v.n, 0)) AS raw_vin
            FROM cell_remoteness cr LEFT JOIN v USING (h3_r6)
        ),
        b AS (SELECT min(raw_vin) AS lo, max(raw_vin) AS hi FROM rv),
        rvn AS (
            SELECT h3_r6, CASE WHEN b.hi = b.lo THEN 0.0
                               ELSE (raw_vin - b.lo) / (b.hi - b.lo) END AS r_vin
            FROM rv CROSS JOIN b
        )
        SELECT uf.user_id_hash, uf.origin_iso, uf.y_mean_remoteness AS y_now,
               SUM(ucv.photo_count * rvn.r_vin) / SUM(ucv.photo_count) AS y_vin
        FROM user_features uf
        JOIN user_cell_visits ucv USING (user_id_hash)
        JOIN rvn USING (h3_r6)
        GROUP BY 1, 2, 3
    """).df()
    con.close()

    # ---- cell-level stats ---------------------------------------------------
    rho, _ = stats.spearmanr(cells.r_now, cells.r_vin)
    pear, _ = stats.pearsonr(cells.r_now, cells.r_vin)
    d = (cells.r_now - cells.r_vin).abs()
    dec_now = pd.qcut(cells.r_now.rank(method="first"), 10, labels=False)
    dec_vin = pd.qcut(cells.r_vin.rank(method="first"), 10, labels=False)
    dec_stable = float((dec_now == dec_vin).mean())
    n_zero_vin = int((cells.poi_vin == 0).sum())

    print(f"=== yardstick vintage check: current snapshot vs {args.label} ===")
    print(f"cells (fixed universe): {len(cells):,}   vintage POIs total: {int(cells.poi_vin.sum()):,} "
          f"vs current {int(cells.poi_now.sum()):,}")
    print(f"cells with ZERO vintage POIs: {n_zero_vin:,} ({n_zero_vin/len(cells):.1%})")
    print(f"\ncell-level:  Spearman rho = {rho:.4f}   Pearson r = {pear:.4f}")
    print(f"|delta remoteness_norm|: mean={d.mean():.4f}  p95={d.quantile(0.95):.4f}  max={d.max():.4f}")
    print(f"same-decile share: {dec_stable:.1%}")

    movers = cells.assign(dd=cells.r_now - cells.r_vin)
    movers = movers.reindex(movers.dd.abs().sort_values(ascending=False).index).head(10)
    print("\ntop movers (r_now - r_vin; + = scored MORE remote today than in vintage):")
    for _, m in movers.iterrows():
        print(f"  ({m.lat:.3f},{m.lng:.3f})  poi {int(m.poi_vin):>5} -> {int(m.poi_now):>5}   "
              f"r {m.r_vin:.3f} -> {m.r_now:.3f}   d={m.dd:+.3f}")

    # ---- user-level (estimate-relevant) stats -------------------------------
    u_pear, _ = stats.pearsonr(users.y_now, users.y_vin)
    u_rho, _ = stats.spearmanr(users.y_now, users.y_vin)
    users["region"] = users.origin_iso.map(REGIONS).fillna("Other")
    reg = (users.groupby("region")[["y_now", "y_vin"]].mean()
                 .assign(n=users.groupby("region").size())
                 .sort_values("n", ascending=False).head(5).round(4))
    print(f"\nuser-level (cohort n={len(users):,}):  Pearson r = {u_pear:.4f}   Spearman = {u_rho:.4f}")
    print("per-region mean outcome under each ruler:")
    print(reg.to_string())

    verdict = "NEGLIGIBLE" if rho >= RHO_FLOOR else "MATERIAL"
    print(f"\nVERDICT (pre-stated floor rho >= {RHO_FLOOR}): vintage effect {verdict}")
    if verdict == "MATERIAL":
        print("-> add a vintage-matched yardstick to the robustness/operationalization table.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"yardstick_vintage_{args.label.replace('-', '')}.json"
    out.write_text(json.dumps({
        "label": args.label, "rho_floor": RHO_FLOOR, "verdict": verdict,
        "n_cells": len(cells), "n_zero_poi_vintage": n_zero_vin,
        "poi_total_vintage": int(cells.poi_vin.sum()), "poi_total_now": int(cells.poi_now.sum()),
        "cell_spearman": rho, "cell_pearson": pear,
        "abs_delta_norm": {"mean": d.mean(), "p95": d.quantile(0.95), "max": d.max()},
        "same_decile_share": dec_stable,
        "user_pearson": u_pear, "user_spearman": u_rho,
        "region_means": reg.to_dict(orient="index"),
    }, indent=2, default=float), encoding="utf-8")
    print(f"-> wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
