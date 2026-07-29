# Emerging Destinations — Project Notes

Portfolio project: does origin-country development (HDI) predict how far off the beaten path travelers go *within Japan* — sticking to the canonical "Golden Route" beaten path vs. venturing to off-the-beaten-path regions? (acclimation vs. status-good vs. heterogeneous hypotheses)

Current state: **real pipeline built and run for Japan.** Stage E (extraction) is complete — millions of geotagged Flickr photos, the user/geocode tables, OSM POIs, indicator panels, and a global per-user photo pull all live in `data/japan.duckdb` (gitignored). Stage T (`pipeline/run.py --transform`) builds the analysis warehouse (`data/warehouse_japan.duckdb`; current sizes live in its `agg_summary` table, column semantics in `DATA_DICTIONARY.md`). Stage A causal machinery (`analysis/`) exists: DAG, DoWhy refutation battery, and design refutations. The `prototype/` slide deck still renders **mocked** data — wiring it to the real aggregates is pending.

The prototype is a **horizontal slide deck**: 6 full-viewport slides you swipe through on mobile or arrow-key through on desktop. The map views render as smooth MapLibre **heatmaps** (clipped to hand-simplified country outlines), not discrete hex polygons.

---

## Running the prototype

The prototype is a static site in `prototype/`. It needs a tiny local web server because MapLibre and CDN scripts misbehave over `file://`.

### Quickest (Windows, from this project root)

```powershell
python -m http.server 8910 --directory prototype
```

then open <http://localhost:8910> in your browser.

### If `python` is not on PATH

Use the Python Launcher (ships with the Windows installer):

```powershell
py -m http.server 8910 --directory prototype
```

Or, if you only have Python 3 explicitly:

```powershell
python3 -m http.server 8910 --directory prototype
```

### Alternative: cd into the prototype dir first

```powershell
cd prototype
python -m http.server 8910
```

### Stopping the server

`Ctrl+C` in the terminal window running it.

### Common mistakes

- `python -m server` — there's no `server` module; the correct module name is `http.server`.
- Running from the wrong directory without `--directory prototype` will 404 on `index.html`.
- Using port 8000 when something else (Django, a previous server) already holds it. Pick a fresh port like 8910 or 8765.

### Slide-deck controls

- **Phone:** swipe left/right.
- **Desktop:** `←` / `→`, `PageUp` / `PageDown`, `Space` (Shift+Space goes back), `Home` / `End`. Or click the dot indicators / prev-next arrows at the bottom.
- **Filter on slide 2** (the maps) updates both heatmaps live.

---

## Project layout

```
Emerging Destinations Project/
├── CLAUDE.md                 ← you are here
├── SPEC.md                   full pre-registered study spec
├── DATA_DICTIONARY.md        Stage-T warehouse column reference
├── pipeline/
│   ├── config.yaml           destination + window + thresholds (single source of truth)
│   ├── run.py                Stage-T orchestrator (runs sql/01..10 into the warehouse)
│   ├── extract/              Stage-E extractors (Flickr quadtree, users, geocodes,
│   │                         global photo pull, OSM POIs, indicators, tourism B1)
│   └── sql/                  numbered warehouse transforms (01_users … 10_aggregates)
├── analysis/
│   ├── generate_dag.py       causal DAG -> dag.html (static SVG)
│   ├── refute.py             pre-registered DoWhy refutation battery (7 tests)
│   └── sensitivity.py        DAG node-specific refutations (negative control,
│                             E-value, home-resolution agreement)
├── prototype/                slide-deck viz (STILL MOCKED DATA)
│   ├── index.html / style.css / data.js / app.js / README.md
├── data/                     gitignored: raw + warehouse DuckDBs, parquet caches
└── .claude/
    └── launch.json           dev-server config consumed by the preview tool
```

Python side is uv-managed (`pyproject.toml` / `uv.lock`; run things with `uv run python …`).
The prototype itself has no build step — all its libraries via CDN: Observable Plot 0.6, MapLibre GL 4.7, Turf.js 7, D3 7.

---

## Data sourcing — what's settled

| Source | Decision |
|---|---|
| Flickr API (`photos.search` + `people.getInfo`) | Use it. Quadtree extraction needed (250-result geo cap, 4000-result query cap). |
| OSM POIs | Use Geofabrik **Japan** PBF (<https://download.geofabrik.de/asia/japan-latest.osm.pbf>, ~2.3 GB) + osmium filter — Overpass is too flaky for batch. |
| World Bank `wbgapi` | Use for GDP/cap PPP, Logistics Performance Index, governance indicators. |
| UNDP HDR API | Use for HDI / IHDI. |
| WHO GHO API | Use for UHC service coverage index, sanitation. |
| Nominatim | Geocode Flickr free-text user `location` field (1 req/sec). |
| **Numbeo** | **Dropped.** Their ToS forbids scraping/redistribution. The construct it was proxying ("cleanliness/transport/convenience") is covered by WB LPI + WHO UHC + WGI Government Effectiveness, which are all openly licensed. |

---

## Hypotheses the real analysis will test

- **H1 acclimation:** high-HDI origin → less remote travel within Japan (sticks to the Golden Route, comfort-seeking)
- **H2 status-good:** high-HDI origin → more remote travel within Japan (off-path as status/adventure)
- **H3 heterogeneous:** sign depends on origin-region (this prototype mocks H3 — long-haul Western origins trend one way, regional East-Asian origins another)
- **H4 null:** origin development indicators don't meaningfully predict within-Japan remoteness exposure

---

## Notes for future work

1. **Replace mocks with real data**, stage by stage. Each pipeline stage produces a versioned parquet/SQLite artifact that the next consumes — don't re-run Flickr extraction just to tweak a model.
2. **Validate user-home resolution** by cross-checking the two methods (stated profile location vs. photo-pattern modal country). Report agreement rate. Drop conflicts or analyze them separately.
3. **Compute multiple competing remoteness scores** (Flickr density, tourist-only Flickr density, OSM POI density, distance-to-core) so the headline finding survives across operationalizations.
4. **Sensitivity analysis** is what makes this a credible portfolio piece — E-values for unmeasured confounding, negative-control outcomes, multiple matching specs.
5. **Write the README narrative**, not just dump charts. The story is "we tested three competing theories of tourist self-sorting on a natural experiment with strong origin-pool variation, and found heterogeneity, not a universal effect."
