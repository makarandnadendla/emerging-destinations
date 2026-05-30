"""One-off: read the POI parquet and print a summary."""
import polars as pl

df = pl.read_parquet("data/cache/pois_raw.parquet")

print(f"total rows: {len(df):,}")

GEO_LAT = (41.0, 43.7)
GEO_LON = (40.0, 46.8)

inside = df.filter(
    (pl.col("lat") >= GEO_LAT[0]) & (pl.col("lat") <= GEO_LAT[1]) &
    (pl.col("lon") >= GEO_LON[0]) & (pl.col("lon") <= GEO_LON[1])
)
outside = df.filter(
    ~((pl.col("lat") >= GEO_LAT[0]) & (pl.col("lat") <= GEO_LAT[1]) &
      (pl.col("lon") >= GEO_LON[0]) & (pl.col("lon") <= GEO_LON[1]))
)
print(f"inside  Georgia bbox: {len(inside):,}")
print(f"outside Georgia bbox: {len(outside):,}")

print("\nout-of-bbox by osm_type:")
for r in outside.group_by("osm_type").len().sort("len", descending=True).to_dicts():
    print(f"  {r['osm_type']:<10} {r['len']:>5,}")

print("\nout-of-bbox by primary_key:")
for r in outside.group_by("primary_key").len().sort("len", descending=True).to_dicts():
    print(f"  {r['primary_key']:<20} {r['len']:>5,}")

print("\nfarthest 10 out-of-bbox rows (by lon distance from Georgia center 43.4):")
sample = (
    outside
    .with_columns((pl.col("lon") - 43.4).abs().alias("d"))
    .sort("d", descending=True)
    .head(10)
    .select(["osm_id", "osm_type", "primary_key", "primary_value", "lat", "lon", "name"])
    .to_dicts()
)
for r in sample:
    name = (r["name"] or "")[:30]
    print(f"  {r['osm_type']:<5} {r['primary_key']:<10}/{r['primary_value']:<14} "
          f"lat={r['lat']:.3f} lon={r['lon']:.3f}  {name}")
