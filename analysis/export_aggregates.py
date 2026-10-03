"""
Export docs/data/aggregates.js — the REAL data behind the docs/ slide deck.

PRIVACY CONTRACT (same as 10_aggregates_for_frontend.sql): no row-level user
data leaves this script. Everything exported is grouped — cell x group x
season photo counts, origin-country aggregates with the min_origin_users
suppression floor, region rollups, and the published model results. No
user_id (hashed or otherwise), no per-user outcomes.

The model numbers are READ FROM THE RECORDED RUNS, never re-derived here:
gate / within-region headline / pooled ladder come from
analysis/outputs/estimate_results.json (written by estimate.py --run), and
the refutation-battery rows from the refutation reports in analysis/outputs/
(written by refute.py --run). The deck therefore shows exactly what the
recorded runs produced; this script only adds the privacy-safe DATA
aggregates (cells, origins, region means, scatter anchors).

Output is a .js file (window.REAL = {...}) instead of .json so the deck can
load it with a plain <script> tag — no fetch, no CORS, works over file://.

Usage: uv run python analysis/export_aggregates.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import yaml

try:
    from analysis.estimate import REGIONS, WAREHOUSE, load_frame
except ImportError:
    from estimate import REGIONS, WAREHOUSE, load_frame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "aggregates.js"
OUTPUTS = Path(__file__).resolve().parent / "outputs"

# Regional = short-haul neighbors; Long-haul = everyone else. Descriptive
# split for the twin maps only — the causal story is per-region.
REGIONAL = {"East Asia", "Southeast Asia"}

SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer",
                   9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

NAME_OVERRIDES = {
    "USA": "United States", "GBR": "UK", "KOR": "South Korea",
    "RUS": "Russia", "TWN": "Taiwan", "HKG": "Hong Kong", "MAC": "Macau",
    "CZE": "Czechia", "VNM": "Vietnam", "IRN": "Iran", "SYR": "Syria",
    "LAO": "Laos", "BRN": "Brunei", "MDA": "Moldova", "BOL": "Bolivia",
    "VEN": "Venezuela", "TZA": "Tanzania",
}


def iso_name(iso: str) -> str:
    if iso in NAME_OVERRIDES:
        return NAME_OVERRIDES[iso]
    try:
        import pycountry
        c = pycountry.countries.get(alpha_3=iso)
        return c.common_name if hasattr(c, "common_name") else c.name
    except Exception:
        return iso


def group_of(region: str) -> str:
    return "Regional" if region in REGIONAL else "Long-haul"


def main() -> int:
    # small-cell suppression floor — same source of truth as Stage T
    pcfg = yaml.safe_load((ROOT / "pipeline" / "config.yaml").read_text(encoding="utf-8"))
    floor = int(pcfg.get("min_origin_users", 5))

    # The recorded estimation run — the ONLY source of the model numbers.
    results_path = OUTPUTS / "estimate_results.json"
    if not results_path.exists():
        sys.exit(f"ERROR: {results_path} not found — run "
                 f"`uv run python analysis/estimate.py --run` first.")
    RES = json.loads(results_path.read_text(encoding="utf-8"))
    C = float(RES["contrast"])

    df = load_frame()
    df["region"] = df.origin_iso.map(REGIONS).fillna("Other")
    df["group"] = df.region.map(group_of)

    con = duckdb.connect(WAREHOUSE, read_only=True)

    # ---- cells: photo counts per cell x origin x month (grouped in SQL), ----
    # ---- rolled up to cell x group x season in pandas. Counts only.      ----
    cell_rows = con.execute("""
        SELECT ph.h3_r6,
               uf.origin_iso,
               EXTRACT(month FROM ph.taken_ts) AS month,
               COUNT(*) AS n_photos
        FROM photos ph
        JOIN user_destination ud USING (user_id_hash)
        JOIN user_features uf USING (user_id_hash)
        WHERE ph.taken_ts IS NOT NULL
          AND EXTRACT(year FROM ph.taken_ts) BETWEEN 2012 AND 2019
        GROUP BY 1, 2, 3
    """).df()
    coords = con.execute("""
        SELECT h3_r6, centroid_lat, centroid_lon, remoteness_norm
        FROM agg_cells
    """).df()

    cell_rows["group"] = cell_rows.origin_iso.map(REGIONS).fillna("Other").map(group_of)
    cell_rows["season"] = cell_rows.month.astype(int).map(SEASON_OF_MONTH)
    pivot = (cell_rows.groupby(["h3_r6", "group", "season"]).n_photos.sum()
             .unstack(["group", "season"], fill_value=0))
    pivot = pivot.reindex(
        columns=[(g, s) for g in ("Long-haul", "Regional") for s in SEASONS],
        fill_value=0)
    pivot.columns = [f"{g}|{s}" for g, s in pivot.columns]
    cells = pivot.reset_index().merge(coords, on="h3_r6", how="left")
    # export as compact arrays: [lat, lon, remoteness, LH W/Sp/Su/A, RG W/Sp/Su/A]
    cell_arr = [
        [round(float(r["centroid_lat"]), 4), round(float(r["centroid_lon"]), 4),
         round(float(r["remoteness_norm"]), 4),
         *[int(r[f"Long-haul|{s}"]) for s in SEASONS],
         *[int(r[f"Regional|{s}"]) for s in SEASONS]]
        for r in cells.to_dict("records")
    ]

    n_photos_window = int(con.execute("""
        SELECT COUNT(*) FROM photos ph
        JOIN user_destination ud USING (user_id_hash)
        WHERE ph.taken_ts IS NOT NULL
          AND EXTRACT(year FROM ph.taken_ts) BETWEEN 2012 AND 2019
    """).fetchone()[0])
    n_cohort = int(con.execute("SELECT COUNT(*) FROM user_features").fetchone()[0])
    con.close()

    # ---- origins (floor applied) ------------------------------------------ #
    org = (df.dropna(subset=["y_first_trip"])
             .groupby("origin_iso")
             .agg(n=("hdi", "size"), hdi=("hdi", "mean"),
                  y_first=("y_first_trip", "mean"), y_pooled=("y_pooled", "mean"),
                  region=("region", "first"), group=("group", "first"))
             .reset_index())
    org = org[org.n >= floor]
    origins = [
        {"iso": r.origin_iso, "name": iso_name(r.origin_iso), "region": r.region,
         "group": r.group, "n": int(r.n), "hdi": round(float(r.hdi), 4),
         "y_first": round(float(r.y_first), 4), "y_pooled": round(float(r.y_pooled), 4)}
        for r in org.itertuples()
    ]

    # ---- gate: rows from the recorded run + DATA rollups (means, group) ---- #
    dfe = df.dropna(subset=["y_first_trip"])
    y_first_mean = dfe.groupby("region").y_first_trip.mean()
    gate = [
        {**row, "group": group_of(row["region"]),
         # same suppression floor as origins: a region mean over < floor
         # users is effectively those users' own outcomes
         "y_first": (round(float(y_first_mean.get(row["region"], float("nan"))), 4)
                     if row["users"] >= floor and row["region"] in y_first_mean
                     else None)}
        for row in RES["gate"]
    ]

    # ---- headline: the recorded estimates + DATA anchors for the fit lines - #
    headline = []
    for row in RES["within_region"]["headline"]:
        sub = dfe[dfe.region == row["region"]]
        headline.append({
            **row,
            "x_mean": round(float(sub.hdi.mean()), 4),
            "y_mean": round(float(sub.y_first_trip.mean()), 4),
            "x_min": round(float(sub.hdi.min()), 4),
            "x_max": round(float(sub.hdi.max()), 4),
            "significant": bool(row["p"] < 0.05),
        })

    # ---- pooled ladder: as recorded --------------------------------------- #
    ladder = RES["ladder"]

    # ---- refutation batteries: from the canonical reports ------------------ #
    def report(stem):
        p = OUTPUTS / f"{stem}.json"
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {"passed": raw["n_passed"], "pending": raw["n_pending"],
                "total": raw["n_total"],
                "rows": [{"name": r["name"], "result": ("PEND" if r["passed"] is None
                                                        else "PASS" if r["passed"] else "FAIL"),
                          "stat": r["statistic"]} for r in raw["results"]]}

    battery = {"dowhy": report("refutation_report"),
               "design": report("design_refutation_report")}

    payload = {
        "generated": date.today().isoformat(),
        "estimate_run": {"generated": RES["generated"],
                         "gap_days": RES["gap_days"], "seed": RES["seed"]},
        "contrast": C,
        "min_origin_users": floor,
        "summary": {
            "n_cohort": n_cohort,
            "n_analysis": int(len(dfe)),
            "n_origins": int(org.shape[0]),
            "n_photos": n_photos_window,
            "n_cells_visited": len(cell_arr),
            "window": "2012–2019",
        },
        "origins": origins,
        "gate": gate,
        "headline": headline,
        "ladder": ladder,
        "battery": battery,
        "cell_columns": ["lat", "lon", "remoteness",
                         *[f"LH_{s}" for s in SEASONS], *[f"RG_{s}" for s in SEASONS]],
        "cells": cell_arr,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "// GENERATED by analysis/export_aggregates.py — do not edit by hand.\n"
        "// Aggregated data only: cell x group x season photo counts, origin\n"
        f"// rollups (n >= {floor} floor), region stats, live model results.\n"
        "window.REAL = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"-> wrote {OUT} ({kb:,.0f} KB; {len(cell_arr):,} visited cells, "
          f"{len(origins)} origins, {len(headline)} in-scope regions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
