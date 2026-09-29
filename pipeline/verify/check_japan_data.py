"""Pre-transform data-state check for Japan."""
import sys
from pathlib import Path
import duckdb

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pois = Path("data/cache/pois_raw_japan.parquet")
print("== OSM POIs ==")
if pois.exists():
    con = duckdb.connect()
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pois.as_posix()}')").fetchone()[0]
    by = con.execute(f"""
        SELECT primary_key, COUNT(*) n FROM read_parquet('{pois.as_posix()}')
        GROUP BY 1 ORDER BY n DESC
    """).fetchall()
    print(f"   {pois}  exists, {n:,} rows")
    for k, c in by:
        print(f"     {k:<18} {c:>8,}")
    con.close()
else:
    print(f"   MISSING: {pois}")

print("\n== japan.duckdb (raw extraction) ==")
con = duckdb.connect("data/japan.duckdb", read_only=True)
tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
print(f"   tables: {tables}")
if "photos" in tables:
    n = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    u = con.execute("SELECT COUNT(DISTINCT user_id) FROM photos").fetchone()[0]
    print(f"   photos: {n:,}  distinct users: {u:,}")
if "frontier" in tables:
    fr = con.execute("SELECT status, COUNT(*) FROM frontier GROUP BY 1 ORDER BY 1").fetchall()
    print(f"   frontier: {dict(fr)}")
if "users" in tables:
    st = con.execute("SELECT status, COUNT(*) FROM users GROUP BY 1 ORDER BY 1").fetchall()
    loc = con.execute("SELECT COUNT(*) FROM users WHERE location_raw IS NOT NULL AND TRIM(location_raw)!=''").fetchone()[0]
    print(f"   users: {dict(st)}  with location_raw: {loc:,}")
if "user_geocodes" in tables:
    gst = con.execute("SELECT status, COUNT(*) FROM user_geocodes GROUP BY 1 ORDER BY 1").fetchall()
    iso = con.execute("SELECT COUNT(*) FROM user_geocodes WHERE country_iso3 IS NOT NULL").fetchone()[0]
    print(f"   user_geocodes: {dict(gst)}  with iso3: {iso:,}")
con.close()
