"""Distinct location strings + top users-by-location (cp1252 safe)."""
import sys
import duckdb

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

con = duckdb.connect("data/flickr.duckdb", read_only=True)
distinct_locs = con.execute("""
    SELECT COUNT(DISTINCT location_raw)
    FROM users
    WHERE location_raw IS NOT NULL AND TRIM(location_raw) != ''
""").fetchone()[0]
print(f"distinct location strings: {distinct_locs}")
print(f"eta for geocoding (1.1 s/req): {distinct_locs * 1.1 / 60:.1f} min")

print("\ntop 15 location strings by user count:")
rows = con.execute("""
    SELECT location_raw, COUNT(*) AS n
    FROM users
    WHERE location_raw IS NOT NULL AND TRIM(location_raw) != ''
    GROUP BY 1 ORDER BY n DESC LIMIT 15
""").fetchall()
for s, n in rows:
    print(f"  {n:>4}  {s[:80]}")
con.close()
