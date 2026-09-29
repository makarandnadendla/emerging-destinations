import sys, duckdb
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
con = duckdb.connect("data/warehouse_japan.duckdb", read_only=True)

print("== agg_summary (headline) ==")
cols = [c[0] for c in con.execute("DESCRIBE agg_summary").fetchall()]
vals = con.execute("SELECT * FROM agg_summary").fetchone()
for k, v in zip(cols, vals):
    print(f"   {k:<26} {v}")

print("\n== users: stated-origin resolution ==")
tot = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
with_iso = con.execute("SELECT COUNT(*) FROM users WHERE stated_country_iso IS NOT NULL").fetchone()[0]
print(f"   {tot:,} users, {with_iso:,} with stated_country_iso ({100*with_iso/tot:.1f}%)")

print("\n== user_features: cohort outcome distribution ==")
q = con.execute("""
  SELECT COUNT(*) n,
         ROUND(AVG(y_mean_remoteness),3),
         ROUND(MIN(y_mean_remoteness),3),
         ROUND(MAX(y_mean_remoteness),3),
         ROUND(AVG(hdi),3),
         COUNT(hdi)
  FROM user_features
""").fetchone()
print(f"   n={q[0]}  y[mean={q[1]} min={q[2]} max={q[3]}]  hdi[mean={q[4]} non-null={q[5]}]")

print("\n== agg_origin: top 15 origins by cohort size ==")
print(f"   {'iso':<5}{'n':>6}{'mean_remote':>13}{'mean_hdi':>10}")
for r in con.execute("""
  SELECT origin_iso, n_users, ROUND(mean_remoteness,3), ROUND(mean_hdi,3)
  FROM agg_origin WHERE origin_iso IS NOT NULL
  ORDER BY n_users DESC LIMIT 15
""").fetchall():
    print(f"   {str(r[0]):<5}{r[1]:>6}{str(r[2]):>13}{str(r[3]):>10}")

null_origin = con.execute("SELECT n_users FROM agg_origin WHERE origin_iso IS NULL").fetchall()
print(f"\n   null-origin cohort rows: {null_origin}")

print("\n== agg_cells: visit coverage ==")
r = con.execute("""
  SELECT COUNT(*),
         SUM(CASE WHEN n_visits>0 THEN 1 ELSE 0 END),
         SUM(n_visits), MAX(n_visits)
  FROM agg_cells
""").fetchone()
print(f"   {r[0]:,} cells, {r[1]:,} visited, {int(r[2]):,} total visits, max {r[3]:,}/cell")
con.close()
