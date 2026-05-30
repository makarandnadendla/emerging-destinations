"""
Extract country-year development indicators from World Bank, UNDP HDR, and WHO GHO.

Output: long-format parquet keyed (country_iso ISO3, year, indicator, value).

Indicators per SPEC sections 4.2 step 08 and 5.4:
  HDI            -- UNDP HDR composite-indices CSV (panel back to 1990)
  GDP_PC_PPP     -- World Bank NY.GDP.PCAP.PP.CD (current international $)
  LPI            -- World Bank LP.LPI.OVRL.XQ (Logistics Performance Index)
  WGI_GOV_EFFECT -- World Bank GE.EST (Worldwide Governance Indicators)
  UHC            -- WHO GHO UHC_INDEX_REPORTED (Universal Health Coverage)

LPI and UHC publish intermittently; carry-forward / interpolation is a
transform-stage concern, not done here.

Usage:
    uv run python pipeline/extract/extract_indicators.py \\
        --out data/cache/indicators_raw.parquet \\
        --start-year 2012 --end-year 2019
"""
from __future__ import annotations

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import polars as pl
import requests
import wbgapi as wb

# Matches columns like `hdi_2012` (case-insensitive) but NOT `hdi_f_2012`,
# `hdi_m_2012`, `hdi_rank_2012`, etc. We want the overall HDI only.
HDI_YEAR_COL_RE = re.compile(r"^hdi_(\d{4})$", re.IGNORECASE)

UNDP_URL = (
    "https://hdr.undp.org/sites/default/files/2025_HDR/"
    "HDR25_Composite_indices_complete_time_series.csv"
)
WHO_UHC_URL = "https://ghoapi.azureedge.net/api/UHC_INDEX_REPORTED"

# (label, wbgapi series code, source db id: 2=WDI, 3=WGI, 66=LPI)
WB_SERIES = [
    ("GDP_PC_PPP",     "NY.GDP.PCAP.PP.CD",  2),
    ("LPI",            "LP.LPI.OVRL.XQ",    66),
    ("WGI_GOV_EFFECT", "GOV_WGI_GE.EST",     3),
]


def get_country_isos() -> set[str]:
    """Country-only ISO3 codes from World Bank metadata (excludes aggregates)."""
    return {e["id"] for e in wb.economy.list() if not e.get("aggregate", False)}


def fetch_wb(start: int, end: int, countries: set[str]) -> list[dict]:
    # Fetch all economies and filter aggregates post-hoc. Passing a 200-item
    # economy list explodes the URL past WB's request-length limit on db=3.
    rows: list[dict] = []
    for label, code, db in WB_SERIES:
        print(f"  WB[db={db}]: {label} ({code})")
        wb.db = db
        for r in wb.data.fetch(code, time=range(start, end + 1)):
            v = r.get("value")
            if v is None:
                continue
            iso = r["economy"]
            if iso not in countries:
                continue
            t = r["time"]
            year = int(t[2:]) if isinstance(t, str) and t.startswith("YR") else int(t)
            rows.append({
                "country_iso": iso,
                "year": year,
                "indicator": label,
                "value": float(v),
            })
    return rows


def fetch_undp(start: int, end: int, countries: set[str]) -> list[dict]:
    print(f"  UNDP: HDI ({UNDP_URL})")
    resp = requests.get(UNDP_URL, timeout=60)
    resp.raise_for_status()
    df = pl.read_csv(StringIO(resp.text))

    iso_col = "iso3" if "iso3" in df.columns else "ISO3"
    hdi_cols = [c for c in df.columns if HDI_YEAR_COL_RE.match(c)]
    if not hdi_cols:
        sys.exit(f"ERROR: no HDI year columns in UNDP CSV. Got first 20: {df.columns[:20]}")

    long = (
        df.select([iso_col] + hdi_cols)
        .unpivot(index=iso_col, on=hdi_cols, variable_name="year_col", value_name="value")
        .with_columns([
            pl.col("year_col").str.split("_").list.last().cast(pl.Int64).alias("year"),
            pl.col(iso_col).alias("country_iso"),
            pl.lit("HDI").alias("indicator"),
        ])
        .filter(pl.col("country_iso").is_in(list(countries)))
        .filter((pl.col("year") >= start) & (pl.col("year") <= end))
        .filter(pl.col("value").is_not_null())
        .with_columns(pl.col("value").cast(pl.Float64))
        .select(["country_iso", "year", "indicator", "value"])
    )
    return long.to_dicts()


def fetch_who(start: int, end: int, countries: set[str]) -> list[dict]:
    print(f"  WHO: UHC_INDEX_REPORTED ({WHO_UHC_URL})")
    resp = requests.get(WHO_UHC_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json().get("value", [])
    rows: list[dict] = []
    for d in data:
        if d.get("SpatialDimType") != "COUNTRY":
            continue
        iso = d.get("SpatialDim")
        if iso not in countries:
            continue
        nv = d.get("NumericValue")
        year = d.get("TimeDim")
        if nv is None or year is None or year < start or year > end:
            continue
        rows.append({
            "country_iso": iso,
            "year": int(year),
            "indicator": "UHC",
            "value": float(nv),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Extract country-year indicators.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start-year", type=int, default=2012)
    p.add_argument("--end-year", type=int, default=2019)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"-> Loading WB country list (for aggregate filter)")
    countries = get_country_isos()
    print(f"   {len(countries)} country ISOs")

    print(f"\n-> Fetching indicators for {args.start_year}-{args.end_year}")
    all_rows: list[dict] = []
    all_rows.extend(fetch_wb(args.start_year, args.end_year, countries))
    all_rows.extend(fetch_undp(args.start_year, args.end_year, countries))
    all_rows.extend(fetch_who(args.start_year, args.end_year, countries))

    if not all_rows:
        sys.exit("ERROR: no rows fetched")

    df = pl.DataFrame(all_rows, schema={
        "country_iso": pl.Utf8,
        "year": pl.Int64,
        "indicator": pl.Utf8,
        "value": pl.Float64,
    })

    print(f"\n-> Writing {args.out}")
    df.write_parquet(args.out)

    print("\nrows per indicator (country-years with non-null value):")
    for r in df.group_by("indicator").len().sort("indicator").to_dicts():
        print(f"  {r['indicator']:<18} {r['len']:>7,}")

    print(f"\nunique countries: {df['country_iso'].n_unique()}")
    print(f"year range:       {df['year'].min()} - {df['year'].max()}")
    print(f"total rows:       {len(df):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
