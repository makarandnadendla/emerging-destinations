"""Spot-check the indicators parquet (cp1252-safe)."""
import sys
import polars as pl

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

df = pl.read_parquet("data/cache/indicators_raw.parquet")

print(f"total: {len(df):,} rows | countries: {df['country_iso'].n_unique()} | years: {df['year'].min()}-{df['year'].max()}\n")

print("value range per indicator (min / med / max / n):")
for ind in sorted(df["indicator"].unique().to_list()):
    sub = df.filter(pl.col("indicator") == ind)["value"]
    print(f"  {ind:<18} min={sub.min():>10.3f}  med={sub.median():>10.3f}  max={sub.max():>10.3f}  n={len(sub):>5,}")

for iso, label in [("GEO", "Georgia"), ("DEU", "Germany"), ("USA", "USA")]:
    print(f"\n{iso} ({label}) 2018:")
    for r in df.filter((pl.col("country_iso") == iso) & (pl.col("year") == 2018)).sort("indicator").to_dicts():
        print(f"  {r['indicator']:<16} {r['value']:>12.3f}")
