"""
SPEC §11/§13 go/no-go: does inverse-POI-density remoteness rank Japan sensibly?
Beaten-path Golden Route should be LOW remoteness (norm near 0, many POIs);
off-path interiors/remote regions should be HIGH remoteness (norm near 1).
"""
import sys
import duckdb

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

con = duckdb.connect("data/warehouse_japan.duckdb", read_only=True)
con.execute("INSTALL h3 FROM community; LOAD h3;")

# (name, lat, lon, expected band)
anchors = [
    ("Tokyo (Shinjuku)",      35.690, 139.700, "LOW"),
    ("Kyoto (centre)",        35.011, 135.768, "LOW"),
    ("Osaka (Namba)",         34.665, 135.501, "LOW"),
    ("Nara (centre)",         34.685, 135.805, "LOW"),
    ("Hakone",                35.232, 139.106, "LOW"),
    ("Hiroshima (centre)",    34.395, 132.459, "LOW"),
    ("Iya Valley (Shikoku)",  33.880, 133.800, "HIGH"),
    ("Daisetsuzan (Hokkaido)",43.550, 142.900, "HIGH"),
    ("Towada/Tohoku interior",40.470, 140.910, "HIGH"),
    ("Kumano/Kii interior",   33.950, 135.780, "HIGH"),
    ("Shiretoko (NE Hokkaido)",44.100, 145.100, "HIGH"),
    ("Okutama (W of Tokyo)",  35.790, 139.000, "HIGH"),
]

print("== anchor probe (expected LOW = beaten path, HIGH = off path) ==")
print(f"   {'anchor':<26}{'exp':<6}{'poi_cnt':>8}{'remote_norm':>13}  zero?")
rows = []
for name, lat, lon, exp in anchors:
    r = con.execute("""
        SELECT cr.poi_count, cr.remoteness_norm, cr.is_zero_poi_cell
        FROM cell_remoteness cr
        WHERE cr.h3_r6 = h3_latlng_to_cell(?, ?, 6)
    """, [lat, lon]).fetchone()
    if r is None:
        print(f"   {name:<26}{exp:<6}{'--':>8}{'(not in universe: 0 POIs/photos)':>13}")
        rows.append((name, exp, None, None))
    else:
        poi, norm, zero = r
        print(f"   {name:<26}{exp:<6}{poi:>8,}{norm:>13.3f}  {zero}")
        rows.append((name, exp, poi, norm))

print("\n== overall remoteness distribution ==")
d = con.execute("""
    SELECT COUNT(*) n,
           COUNT(*) FILTER (WHERE is_zero_poi_cell) zero_poi,
           quantile_cont(remoteness_norm, 0.25) q25,
           quantile_cont(remoteness_norm, 0.50) q50,
           quantile_cont(remoteness_norm, 0.75) q75,
           quantile_cont(poi_count, 0.50) poi_med,
           MAX(poi_count) poi_max
    FROM cell_remoteness
""").fetchone()
print(f"   cells={d[0]:,}  zero-POI cells={d[1]:,} ({100*d[1]/d[0]:.1f}%)")
print(f"   remoteness_norm  q25={d[2]:.3f}  median={d[3]:.3f}  q75={d[4]:.3f}")
print(f"   poi_count        median={d[5]:.0f}  max={d[6]:,}")

print("\n== 5 densest cells (should be major cities; remoteness_norm ~0) ==")
for row in con.execute("""
    SELECT h3_cell_to_lat(h3_r6) lat, h3_cell_to_lng(h3_r6) lon, poi_count, remoteness_norm
    FROM cell_remoteness ORDER BY poi_count DESC LIMIT 5
""").fetchall():
    print(f"   ({row[0]:.3f},{row[1]:.3f})  poi={row[2]:,}  norm={row[3]:.3f}")

# Separation summary
lows = [n for (_, e, p, n) in rows if e == "LOW" and n is not None]
highs = [n for (_, e, p, n) in rows if e == "HIGH" and n is not None]
if lows and highs:
    print(f"\n== separation ==")
    print(f"   mean remoteness_norm  LOW anchors={sum(lows)/len(lows):.3f}   HIGH anchors={sum(highs)/len(highs):.3f}")
    print(f"   max(LOW)={max(lows):.3f}   min(HIGH)={min(highs):.3f}   "
          f"{'CLEAN SEPARATION' if max(lows) < min(highs) else 'OVERLAP — inspect'}")
con.close()
