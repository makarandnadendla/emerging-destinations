import sys, duckdb
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
con = duckdb.connect("data/warehouse_japan.duckdb", read_only=True)
con.execute("INSTALL h3 FROM community; LOAD h3;")
# Korea/Russia bleed boxes (same as test_clip.py) — do any VISITED cells fall in them?
boxes = {
  "Seoul (KR)":(37.3,37.8,126.6,127.3),
  "Busan (KR)":(35.0,35.3,128.9,129.2),
  "Gangneung (KR)":(37.5,37.9,128.7,129.1),
  "Vladivostok (RU)":(42.9,43.3,131.7,132.2),
}
print("== bleed cells that received tourist visits ==")
total_bleed_visits = 0
for name,(la0,la1,lo0,lo1) in boxes.items():
    r = con.execute("""
      SELECT COUNT(*) ncells, COALESCE(SUM(n_visits),0) nv, COALESCE(SUM(n_users),0) nu
      FROM agg_cells
      WHERE centroid_lat BETWEEN ? AND ? AND centroid_lon BETWEEN ? AND ?
    """,[la0,la1,lo0,lo1]).fetchone()
    total_bleed_visits += int(r[1])
    flag = "OK (inert)" if r[1]==0 else "!! VISITED"
    print(f"   {name:<16} cells={r[0]} visits={int(r[1])} users={int(r[2])}  {flag}")
print(f"\n   total visits landing in bleed boxes: {total_bleed_visits}")
print(f"   (of {con.execute('SELECT SUM(n_visits) FROM agg_cells').fetchone()[0]:,.0f} total visits)")
con.close()
