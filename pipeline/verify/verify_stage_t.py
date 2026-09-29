"""Spot-check Stage T outputs in the Japan warehouse."""
import sys
import duckdb

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

con = duckdb.connect("data/warehouse_japan.duckdb", read_only=True)
con.execute("INSTALL h3 FROM community; LOAD h3;")

print("== photos: time fields ==")
r = con.execute("""
    SELECT MIN(taken_hour), MAX(taken_hour), MIN(taken_month), MAX(taken_month),
           COUNT(*) FILTER (WHERE taken_ts IS NULL) AS null_ts
    FROM photos
""").fetchone()
print(f"   hour {r[0]}..{r[1]}  month {r[2]}..{r[3]}  null_ts={r[4]}")

print("\n== photos: distinct H3 r6 cells + centroid bounds (should sit in Japan) ==")
r = con.execute("""
    WITH c AS (SELECT DISTINCT h3_r6 FROM photos)
    SELECT COUNT(*) AS n_cells,
           MIN(h3_cell_to_lat(h3_r6)), MAX(h3_cell_to_lat(h3_r6)),
           MIN(h3_cell_to_lng(h3_r6)), MAX(h3_cell_to_lng(h3_r6))
    FROM c
""").fetchone()
print(f"   distinct cells={r[0]:,}  lat {r[1]:.2f}..{r[2]:.2f}  lon {r[3]:.2f}..{r[4]:.2f}")

print("\n== photos: top 5 cells by photo count (with centroid) ==")
for row in con.execute("""
    SELECT h3_h3_to_string(h3_r6) AS cell,
           h3_cell_to_lat(h3_r6) AS lat, h3_cell_to_lng(h3_r6) AS lon,
           COUNT(*) AS n
    FROM photos GROUP BY h3_r6 ORDER BY n DESC LIMIT 5
""").fetchall():
    print(f"   {row[0]}  ({row[1]:.3f},{row[2]:.3f})  {row[3]:,} photos")

print("\n== indicators_panel: JPN across years ==")
for row in con.execute("""
    SELECT year, hdi, gdp_pc_ppp, lpi, uhc, wgi_gov_effect
    FROM indicators_panel WHERE country_iso3='JPN' ORDER BY year
""").fetchall():
    print(f"   {row[0]}  hdi={row[1]}  gdp={row[2]}  lpi={row[3]}  uhc={row[4]}  wgi={row[5]}")

print("\n== indicators_panel: coverage ==")
r = con.execute("""
    SELECT COUNT(*) AS rows, COUNT(DISTINCT country_iso3) AS countries,
           MIN(year), MAX(year),
           COUNT(*) FILTER (WHERE hdi IS NOT NULL) AS hdi_nonnull
    FROM indicators_panel
""").fetchone()
print(f"   rows={r[0]:,}  countries={r[1]}  years {r[2]}..{r[3]}  hdi_nonnull={r[4]:,}")
con.close()
