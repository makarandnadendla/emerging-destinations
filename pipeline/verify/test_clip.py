"""
Validate the Japan clip: POI-bearing cells from the Geofabrik Japan PBF must
contain NO cells in Korea (Seoul, Busan) or the Russian Far East (Vladivostok),
which the rectangular Flickr bbox sweeps in. Reads the POI parquet only — does
NOT touch the locked raw japan.duckdb.
"""
import sys
import duckdb

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

con = duckdb.connect()
con.execute("INSTALL h3 FROM community; LOAD h3;")
con.execute("""
    CREATE TABLE cells AS
    WITH poi_cells AS (
        SELECT DISTINCT h3_latlng_to_cell(lat, lon, 6) AS h3_r6
        FROM read_parquet('data/cache/pois_raw_japan.parquet')
        WHERE lat IS NOT NULL AND lon IS NOT NULL
    )
    SELECT h3_r6,
           h3_cell_to_lat(h3_r6) AS lat,
           h3_cell_to_lng(h3_r6) AS lon
    FROM poi_cells;
""")

n = con.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
ext = con.execute("SELECT MIN(lat), MAX(lat), MIN(lon), MAX(lon) FROM cells").fetchone()
print(f"POI-bearing cells: {n:,}")
print(f"extent: lat {ext[0]:.2f}..{ext[1]:.2f}  lon {ext[2]:.2f}..{ext[3]:.2f}")

boxes = {
    "Seoul (KR)":       (37.3, 37.8, 126.6, 127.3),
    "Busan (KR)":       (35.0, 35.3, 128.9, 129.2),
    "Vladivostok (RU)": (42.9, 43.3, 131.7, 132.2),
}
print("\nbleed check (expect 0 in each):")
clean = True
for name, (la0, la1, lo0, lo1) in boxes.items():
    c = con.execute("""
        SELECT COUNT(*) FROM cells
        WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
    """, [la0, la1, lo0, lo1]).fetchone()[0]
    flag = "OK" if c == 0 else "!! BLEED"
    if c != 0:
        clean = False
    print(f"   {name:<18} cells={c}  {flag}")

print(f"\nverdict: {'CLEAN — clip excludes non-Japan territory' if clean else 'BLEED DETECTED — needs boundary clip'}")

# Northwesternmost cells: should be Japanese (Tsushima ~34.4N/129.3E, San'in coast), not Korea.
print("\nnorthwest-most POI cells (lat>35.5 & lon<133, should be Japan San'in/Oki, not KR):")
for row in con.execute("""
    SELECT lat, lon FROM cells
    WHERE lat > 35.5 AND lon < 133.0
    ORDER BY lon ASC LIMIT 5
""").fetchall():
    print(f"   ({row[0]:.3f}, {row[1]:.3f})")
con.close()
