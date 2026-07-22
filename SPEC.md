# Emerging Destinations — Build Spec

> Build spec for the real (non-mocked) version of the Emerging Destinations project. Companion to `CLAUDE.md`. The prototype in `prototype/` is the mocked predecessor.

---

## 1. Context

The prototype in `prototype/` is a 6-slide horizontal deck rendering **mocked** data. It exists to validate the visual story before any real extraction. The real question is empirical: **does origin-country development (HDI) predict how far off the beaten path travelers go *within Japan* — sticking to the canonical "Golden Route" beaten path (Tokyo, Hakone, Kyoto, Nara, Osaka; plus Hiroshima/Miyajima and the Mt. Fuji area) vs. venturing to off-the-beaten-path regions (Tōhoku interior, the San'in coast, Shikoku interior, Hokkaido interior/east, the Kii Peninsula / Kumano Kodō, and remote islands)?**

This spec replaces the mock with a real pipeline:

- **Primary data:** Flickr geotagged photos + user profile location, 2012–2019, with iNaturalist as a fallback if Flickr access stalls.
- **Outcome:** remoteness defined externally via **inverse OSM POI density**, not Flickr density (decouples the outcome from the photographer-population we're sampling).
- **Unit of analysis:** **(user × hex cell) visits** on an **H3 resolution-6** grid (~36 km² hexes).
- **Identification:** **doubly-robust regression** (outcome model + propensity score) on continuous HDI, with E-value sensitivity and a hour-of-day negative control.
- **Deliverables:** (a) the existing 6-slide findings deck with real numbers, (b) a longer methods deck, (c) a 12–20 page Quarto methodology PDF for hiring managers + technical reviewers.

**Why this shape:** the headline framing in CLAUDE.md (H1 acclimation / H2 status-good / H3 heterogeneous / H4 null) and the prototype's visual language stay intact, but the analysis backbone is moved from a synthetic generator to a reproducible DuckDB-based pipeline. The methodology PDF carries the causal-inference rigor; the deck carries the story.

**Hard constraint:** 2 weeks aggressive. Scope is **within-Japan only** — a single-destination, within-country analysis of beaten-path vs. off-path travel.

---

## 2. Scope decisions (locked from interview)

| Decision | Choice |
|---|---|
| Destination | **Japan only — within-country (beaten-path vs. off-path)** |
| Time window | **2012–2019** (pre-COVID, stable Flickr adoption) |
| Time budget | 2 weeks aggressive |
| Hosting | **GitHub Pages** from `/docs` |
| Pipeline | **DuckDB + SQL transforms** (dbt-style, hand-rolled — no dbt itself) |
| PDF tool | **Quarto** → PDF + HTML |
| Data in repo | **Aggregated only, no raw user data** |
| Primary data | Flickr API (`photos.search`, `people.getInfo`) |
| Fallback | iNaturalist API if Flickr key delayed >5 days |
| Grid | **H3 r6** (~36 km² hexes) |
| Outcome | **Inverse OSM POI density per cell**, normalized 0–1 |
| User-home rule | **Stated Flickr profile location wins**; modal-country flags audit |
| Tourist filter | Drop users whose STATED profile country = destination (consistent with "stated location wins"; the destination-excluded modal cannot flag residents by construction) |
| Inclusion threshold | ≥5 photos AND ≥2 distinct hex cells in destination |
| Origin predictor | **HDI primary**; GDP/cap PPP, LPI, composite as robustness |
| HDI vintage | **Year-matched** (panel join on trip year) |
| Treatment spec | **Continuous HDI**, doubly-robust regression (no binarization) |
| Model 1 (headline) | User-level OLS on photo-weighted mean cell-remoteness |
| Model 2 (supplemental) | Cell-level multilevel logistic on (user × cell) visits |
| Sensitivity | E-values for unmeasured confounding |
| Negative control | Photo timestamp hour-of-day (should NOT depend on HDI conditional on season) |
| Ship criterion | Any defensible finding, including null, if pipeline is sound and rigorous |
| Deck structure | Keep existing 6-slide structure for findings; build separate methods deck |

---

## 3. Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Extraction     │ →  │  DuckDB warehouse│ →  │  Analysis       │
│  (Python jobs)  │    │  (single .duckdb)│    │  (Python + SQL) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Cloud storage         Aggregated JSON         Quarto → PDF/HTML
   (raw, gitignored)     (committed to repo)     Slide decks (static)
```

Three stage boundaries, each with a **versioned artifact**:

1. **Extract** raw observations from Flickr / OSM / WB / UNDP / WHO. Outputs: parquet files in cloud storage (B2 or R2 — cheap, S3-compatible).
2. **Transform** into a DuckDB warehouse with SQL views. Outputs: `data/warehouse.duckdb` (gitignored if it contains user-level rows; committed if aggregated).
3. **Analyze + render**. Outputs: aggregated JSON (`docs/data/*.json`) consumed by slide decks; figures + PDF rendered by Quarto.

Re-running stage 3 must never require re-running stage 1. Re-running stage 2 must be fast (<2 min) so analysis iteration is cheap.

---

## 4. Pipeline stages & artifacts

### 4.1 Stage E (Extract) — `pipeline/extract/`

Each script is idempotent: runs against the cloud bucket, skips already-extracted partitions, writes parquet.

| Script | Source | Output (parquet) |
|---|---|---|
| `extract_flickr_photos.py` | Flickr `photos.search` (quadtree over Japan bbox `[122.93, 24.04, 145.82, 45.55]`, 2012–2019) | `photos_raw/dt=YYYYMMDD/*.parquet` — one row per photo |
| `extract_flickr_users.py` | Flickr `people.getInfo` for unique user_ids from above | `users_raw/dt=YYYYMMDD/*.parquet` |
| `extract_osm_pois.py` | Geofabrik Japan PBF + `osmium tags-filter` | `pois_raw.parquet` — one row per POI |
| `geocode_user_locations.py` | Nominatim, 1 req/sec, only for unparseable users | `user_geocodes.parquet` |
| `extract_indicators.py` | World Bank `wbgapi`, UNDP HDR API, WHO GHO API | `indicators.parquet` — country × year × indicator |
| (fallback) `extract_inaturalist.py` | iNat observations API, Japan bbox | `inat_raw.parquet` |

**Flickr quadtree extraction** (`extract_flickr_photos.py`): recursive bbox subdivision when a query returns ≥4000 results. Pages 500 results at a time (Flickr cap is 250/page for geo queries; the 4000-per-query cap is what triggers subdivision). Persist a `frontier` table in DuckDB so the extraction can resume cleanly. Estimated: an **overnight, possibly multi-day** drain for Japan 2012–2019 — early frontier probes show ~40k–47k geotagged photos per year at the top bbox level, so expect hundreds of thousands of photos and tens of thousands of users, and consider per-tile sampling if a full drain is impractical. Rate-limited at ~1 req/sec to stay polite (no documented rate limit but courtesy).

**User-home resolution** runs **two methods**, agreement rate reported in methodology PDF:
1. **Stated location** (primary): parse `location` from `people.getInfo` via Nominatim, retain country-level result. Drop if no country resolved.
2. **Modal-photo-country** (audit): for each user, find the modal country across **all** their geotagged photos (not just Japan). Used only to flag disagreements and as a sensitivity check — does not replace stated location.

**OSM POI extraction** uses `osmium tags-filter` on the Japan PBF. POI tag set (subject to one tweak after first inspection):
- `tourism=*` (hotels, viewpoints, attractions, guest_houses, hostels)
- `amenity=restaurant|cafe|bar|fast_food|hospital|pharmacy|atm|bank|fuel`
- `shop=supermarket|convenience|bakery|mall`
- `public_transport=*`

Each POI becomes a (lat, lon, primary_tag) row. Hex-aggregation happens in Stage T.

### 4.2 Stage T (Transform) — `pipeline/sql/`

Numbered SQL files in `pipeline/sql/` (`01_users.sql`, `02_photos.sql`, etc.) executed against `data/warehouse.duckdb`. Each file creates/replaces a view or materialized table.

| Step | Output table | Notes |
|---|---|---|
| 01 | `users` | One row per user. Columns: `user_id_hash`, `stated_country_iso`, `modal_country_iso`, `agree_flag`, `total_photos`, `n_countries_visited`. user_id hashed (sha256, no salt needed since IDs are public, but we don't surface them) |
| 02 | `photos` | One row per photo. `photo_id`, `user_id_hash`, `lat`, `lon`, `taken_ts`, `dt_local` (with timezone), `h3_r6` |
| 03 | `cells` | One row per H3 r6 cell that intersects Japan. `h3_r6`, `centroid_lat`, `centroid_lon`, `area_km2_in_country` |
| 04 | `poi_per_cell` | `h3_r6`, `poi_count`, `poi_count_by_category{}` |
| 05 | `cell_remoteness` | `h3_r6`, `remoteness_raw = -log(1 + poi_count)`, `remoteness_norm` ∈ [0,1] (min–max scaled on the destination), `is_zero_poi_cell` flag |
| 06 | `user_destination` | Filter to tourists: `users.stated_country_iso ≠ 'JPN'` (stated location wins; modal is the audit signal, and the destination-excluded modal can never equal the destination for non-residents), ≥5 photos & ≥2 cells in Japan. Origin country + year-of-first-photo carried forward |
| 07 | `user_cell_visits` | One row per (user × h3_r6). Photo count, first/last visit ts. Inner join to `cell_remoteness` |
| 08 | `indicators_panel` | Country × year × indicator (HDI, GDP_PC_PPP, LPI, UHC, WGI_GOVE). Year-matched to user's first-photo year in destination |
| 09 | `user_features` | Join `user_destination` to `indicators_panel` on (origin_iso, trip_year). One row per analysis user |
| 10 | `aggregates_for_frontend` | Aggregated rollups for slide-deck JSON: cell-level mean remoteness, origin-region composition, etc. No row-level user data |

`docs/data/aggregates.json` is exported from step 10. **Nothing else lands in the public repo from the warehouse** — raw user-level tables stay in cloud storage.

### 4.3 Stage A (Analyze + render) — `analysis/`

`analysis/01_descriptives.qmd` — sample sizes, coverage maps, agreement rate, missingness.

`analysis/02_models.qmd` — the two-part regression:

- **Model 1 (headline, user-level OLS, doubly robust):**
  - Outcome: `Y_i = Σ_c (photos_ic / Σ_c' photos_ic') × remoteness_c` — photo-weighted mean cell-remoteness for user *i*.
  - Treatment: continuous `HDI_origin_country, trip_year`.
  - Confounders: origin region (7 levels), GDP/cap PPP, distance origin↔destination capital, total_photos (log).
  - **DR estimator:** outcome model (OLS with confounders) + propensity score for the treatment (linear regression of HDI on confounders). Combined via Robins–Rotnitzky–Zhao DR estimator (use `econml` or hand-coded; `econml` preferred).
  - Standard errors: bootstrap (500 reps), cluster on origin country.

- **Model 2 (supplemental, cell-level multilevel logistic):**
  - One row per (user × cell-that-could-be-visited-in-Japan). Outcome: visited (1/0).
  - Predictors: cell remoteness × HDI interaction, cell fixed effects, user random intercept.
  - Use `pymer4` (lme4 binding) or `statsmodels.mixedlm` (limited but adequate).
  - Slow: subsample cells if needed; report runtime.

- **Robustness table:** swap HDI → GDP/cap PPP → LPI → 1st PC composite. Same DR spec each time.

- **Negative control:** mean photo timestamp hour-of-day per user, **conditional on month-of-year** (to remove seasonal daylight confounding). Should not depend on HDI. Report the same DR coefficient on this placebo outcome. If it's significant, we have unmeasured confounding to flag.

- **E-values:** Compute Vanderweele/Ding E-value for the headline coefficient. Report what unmeasured confounder strength would be needed to overturn the result.

`analysis/03_figures.qmd` — every figure that appears in the deck and PDF is rendered here, saved to `docs/figures/`. Single source of truth.

`analysis/04_methodology_paper.qmd` — the 12–20 page PDF.

---

## 5. Data sources — specific configurations

### 5.1 Flickr
- API key: register at https://www.flickr.com/services/apps/create/apply/ — **non-commercial use is auto-approved typically within hours**. Key delay >5 days triggers iNat fallback.
- Endpoint: `flickr.photos.search` with `bbox` (Japan: `[122.93, 24.04, 145.82, 45.55]`), `min_taken_date`, `max_taken_date`, `extras=geo,date_taken,owner_name,tags`, `per_page=500`.
- User profile: `flickr.people.getInfo` with `user_id`. Field of interest: `person.location._content`.
- ToS: API use OK; redistribution of user-level metadata is restricted. **We commit only aggregates.**

### 5.2 iNaturalist (fallback)
- No key required. API rate limit ~1 req/sec.
- Endpoint: `GET /v1/observations?place_id=<japan>&per_page=200&order_by=observed_on`.
- User home: each user has a `place_id` they self-set — cleaner than Flickr's free-text.
- Bias: skews toward nature observers. **Arguably a better signal for remoteness** but a different population from the Flickr cohort. Methods PDF discusses.

### 5.3 OSM via Geofabrik
- Download `japan-latest.osm.pbf` (~2.3 GB) from https://download.geofabrik.de/asia/japan.html
- `osmium tags-filter` (must have `osmium-tool` installed; spec includes install step).
- Filter: `nwr/tourism nwr/amenity=restaurant,cafe,...` etc.
- Re-download monthly if pipeline is re-run (POIs are not stable over the 2012–2019 window, but using a single recent snapshot is defensible — methods PDF will note this is a static remoteness measure applied to a dynamic photo dataset).

### 5.4 World Bank / UNDP / WHO indicators
- `wbgapi.data.DataFrame(['NY.GDP.PCAP.PP.CD', 'LP.LPI.OVRL.XQ'], time=range(2012,2020))` — both indicators by country×year.
- UNDP HDR API: `https://hdr.undp.org/sites/default/files/data/2023/HDR23-24_Composite_indices_complete_time_series.csv` (CSV download — simpler than API). HDI panel back to 1990.
- WHO GHO: `ghoclient` Python package or direct API at `https://ghoapi.azureedge.net/api/UHC_INDEX_REPORTED`.
- All indicators land in `indicators_panel` keyed (country_iso, year).

### 5.5 Nominatim
- Public instance at https://nominatim.openstreetmap.org with strict 1 req/sec, user-agent must identify the project.
- Used only for parsing Flickr profile location strings (e.g. "Munich, Germany" → DE).
- Cache aggressively in DuckDB (`user_geocodes`) — same string never queries twice.
- Quality threshold: keep only results with `address.country_code` populated.

---

## 6. Frontend

### 6.1 Findings deck (existing 6-slide structure)
- **No structural change** to `prototype/index.html` or the swipe-deck UX.
- Replace data file `prototype/data.js` (the mock generator) with `prototype/data.js` that fetches `docs/data/aggregates.json` and exposes the same shape (`trips`, `hexFeatures`, etc.) so `app.js` is unchanged.
- Charts re-render against real numbers. Copy in `index.html` rewritten to reflect actual finding (which could be H1, H2, H3, or null).
- Same library stack (Observable Plot, MapLibre, Turf, D3 — all CDN). **Turf grid logic in `app.js` retained** for the case where the analysis grid (H3 r6) is also exported as polygon features in the JSON. We render H3 hexes by passing precomputed GeoJSON polygons to MapLibre; client doesn't need an H3 library.

### 6.2 Methods deck (new)
- Separate page: `docs/methods.html`. Same swipe-deck framework (factor out the deck controller from `app.js` into `deck.js`).
- ~8–10 slides:
  1. Problem statement + DAG diagram (SVG)
  2. Data sources + sample funnel (extracted → tourist-filtered → analysis-eligible)
  3. User-home resolution + agreement rate
  4. Remoteness construction (OSM POI → hex → normalized)
  5. Coverage map + sample size by origin region
  6. Headline regression (DR) result with CI
  7. Robustness table (HDI/GDP/LPI/composite)
  8. Negative control + E-value sensitivity
  9. Limitations
  10. Links to PDF + code

### 6.3 Shared deck controller
Extract slide-deck logic (keyboard, swipe, autoplay, progress) from `app.js` into `prototype/deck.js`. Both `index.html` (findings) and `methods.html` (methods) import it.

---

## 7. Methodology PDF (Quarto)

`analysis/04_methodology_paper.qmd` → `docs/methodology.pdf` (and `methodology.html` rendered alongside).

**Target length:** 12–20 pages. Hiring-manager + technical reviewer register: executive summary up front, full method, results, robustness, limitations. Math in appendix.

**Sections:**
1. Executive summary (1 page) — finding + 3 charts
2. Motivation + hypotheses (1 p)
3. Data (2–3 p) — Flickr extraction, sample funnel, user-home resolution, OSM POI definition, indicator vintages
4. Identification (1–2 p) — DAG, assumptions, why DR
5. Methods (2 p) — DR estimator, multilevel logistic, robustness specs
6. Results (3–4 p) — headline coefficient, robustness, heterogeneity by origin region
7. Sensitivity + negative control (2 p) — E-values, hour-of-day placebo
8. Limitations (1 p) — Flickr selection, static OSM, stated location selection, single-destination scope
9. Appendix — math, code links, reproducibility instructions

Math rendered via Quarto's native LaTeX. Code from analysis notebooks pulled in via Quarto `embed` directives, not re-typed.

---

## 8. Repo structure

```
Emerging Destinations Project/
├── CLAUDE.md
├── SPEC.md                          ← this document
├── README.md                        ← project overview (rewritten from prototype/README.md)
├── .gitignore                       ← excludes data/, .venv/, *.duckdb (when raw)
├── pyproject.toml                   ← uv/pip project, pinned deps
├── pipeline/
│   ├── extract/
│   │   ├── extract_flickr_photos.py
│   │   ├── extract_flickr_users.py
│   │   ├── extract_osm_pois.py
│   │   ├── geocode_user_locations.py
│   │   ├── extract_indicators.py
│   │   └── extract_inaturalist.py   ← fallback only
│   ├── sql/
│   │   ├── 01_users.sql
│   │   ├── 02_photos.sql
│   │   ├── ... (10 files)
│   │   └── 10_aggregates_for_frontend.sql
│   ├── run.py                       ← orchestrator: runs extract → SQL in order, idempotent
│   └── config.yaml                  ← bbox, time window, thresholds, paths
├── analysis/
│   ├── 01_descriptives.qmd
│   ├── 02_models.qmd
│   ├── 03_figures.qmd
│   ├── 04_methodology_paper.qmd
│   └── _quarto.yml
├── docs/                            ← GitHub Pages root
│   ├── index.html                   ← findings deck (was prototype/index.html)
│   ├── methods.html                 ← methods deck (new)
│   ├── methodology.pdf
│   ├── methodology.html
│   ├── style.css
│   ├── app.js                       ← was prototype/app.js
│   ├── deck.js                      ← extracted controller
│   ├── data/
│   │   └── aggregates.json
│   └── figures/                     ← all PNG/SVG from analysis/
├── prototype/                       ← kept for reference, eventually deleted
└── data/                            ← gitignored; local cache only
    ├── warehouse.duckdb
    └── cache/                       ← raw extraction artifacts, mirrored from cloud
```

**Cloud storage:** Backblaze B2 bucket (cheapest S3-compatible). Single bucket `emerging-destinations` with `raw/photos/`, `raw/users/`, `raw/pois/`, `raw/indicators/` prefixes. Credentials in `.env` (gitignored). `pipeline/run.py` reads `B2_*` env vars; falls back to local `data/cache/` if not set.

---

## 9. Deployment

GitHub Pages serves `/docs` on push to `main`. No build step in CI (everything is static after Quarto pre-renders locally).

**Update flow:**
1. Re-run pipeline locally → updates `docs/data/aggregates.json` + `docs/figures/`.
2. Re-render Quarto → updates `docs/methodology.pdf`, `docs/methodology.html`.
3. Commit `docs/` changes → push → GitHub Pages auto-deploys.

No Vercel, no Netlify, no custom domain (deferred).

---

## 10. Timeline (2 weeks aggressive)

| Day | Track A: data | Track B: analysis | Track C: writeup |
|---|---|---|---|
| 1 | Register Flickr key (submit immediately). Set up B2. Geofabrik download. | Repo scaffolding, pyproject, .gitignore. | Quarto + LaTeX install. |
| 2 | OSM POI extraction (Japan PBF ~2.3 GB; osmium filter still fast). World Bank + UNDP + WHO indicator pulls. | SQL stubs 01–05. | Outline methodology PDF. |
| 3 | Flickr quadtree extraction kicks off (background; expect an overnight, possibly multi-day drain). | Cell remoteness model finalized; verify against intuition. | — |
| 4 | Flickr extraction continues. Start Nominatim geocoding queue. | Sample funnel sanity-check (low-fi). | DAG draft. |
| 5 | Flickr extraction continues. | First user-level OLS on whatever data is in. | — |
| 6 | Flickr extraction completes (target). User-home resolution. | Full DR regression. Bootstrap SEs. | — |
| 7 | All data settled. Final aggregates JSON. | Robustness table. Negative control. E-values. | — |
| 8 | — | Multilevel logistic (Model 2). | Methods PDF sections 1–4 draft. |
| 9 | — | Figures finalized in 03_figures.qmd. | Methods PDF sections 5–7 draft. |
| 10 | — | — | Findings deck copy rewrite + chart replacement. |
| 11 | — | — | Methods deck build (deck.js extraction + new slides). |
| 12 | — | — | PDF polish + render. |
| 13 | — | Sensitivity / proofreading buffer. | Repo cleanup, README, deploy to GitHub Pages. |
| 14 | Buffer day. | Buffer. | Buffer. |

**Future work (out of scope):** additional within-Japan robustness (finer H3 resolutions, origin-region sub-splits) or other destinations as comparators. Documented as future work in the PDF; not part of this build.

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Flickr key delayed >5 days | Med | High | Switch to iNaturalist at day 5 checkpoint. Methodology PDF discusses cohort difference. |
| Flickr quadtree extraction stalls / hits hidden quota | Med | High | Resume-able frontier in DuckDB. Worst case: smaller time window (2015–2019). |
| User-home parse rate <40% | Med | Med | Lower bound: 30% is still publishable with caveat. If <20%, switch primary signal to modal-country. |
| Japan sample very **large** (hundreds of thousands of photos, tens of thousands of users) | High | Med | Flickr drain becomes an overnight/multi-day job. Mitigate with resume-able frontier, per-tile sampling (cap photos/tile or sample users), and a smaller time window (2015–2019) if extraction runtime is impractical. Document any sampling in methods. |
| Null result | High | Low | Pre-committed to ship. Null result with E-values + neg-control passing is a respectable portfolio piece. |
| Sign of coefficient flips between Model 1 and Model 2 | Med | Med | Document the divergence carefully — actually an *interesting* finding about user-vs-cell framing differences. |
| Nominatim rate limits exhausted | Low | Low | Cache + back off. Worst case: spin up local Nominatim Docker. |
| OSM POI density poorly proxies remoteness | Low | High | Pre-validate against known anchor points (Golden Route — Tokyo/Kyoto/Osaka — should be lowest remoteness; Tōhoku interior, the San'in coast, and remote islands highest). Sanity check first. |
| Quarto PDF render fails on Windows | Low | Low | Render in WSL or Docker. |

---

## 12. Critical files to be created or replaced

(Pattern: scaffolding new files for the pipeline + analysis + deck split. Representative paths, not exhaustive.)

- **New:** `pipeline/run.py`, `pipeline/extract/*.py`, `pipeline/sql/*.sql`, `pipeline/config.yaml`
- **New:** `analysis/01_descriptives.qmd`, `analysis/02_models.qmd`, `analysis/03_figures.qmd`, `analysis/04_methodology_paper.qmd`, `analysis/_quarto.yml`
- **New:** `pyproject.toml` (uv-managed), `.gitignore`, `.env.example`, `README.md` (project-level, replacing `prototype/README.md` as the canonical entry point)
- **Refactor:** `prototype/app.js` → split into `docs/app.js` (findings) + `docs/deck.js` (shared controller). `prototype/data.js` (mock generator) → replaced by a thin loader that fetches `docs/data/aggregates.json`.
- **New:** `docs/methods.html` and assets, `docs/figures/*`, `docs/data/aggregates.json`
- **Keep:** `prototype/style.css` → moved to `docs/style.css`, color tokens unchanged.

---

## 13. Verification

The pipeline is verified end-to-end if:

1. `python pipeline/run.py --extract` produces parquet artifacts in cloud storage (or `data/cache/` for local dev) with row counts within expected order of magnitude (photos: hundreds of thousands; users: tens of thousands; POIs: hundreds of thousands for Japan).
2. `python pipeline/run.py --transform` produces `data/warehouse.duckdb` with all 10 SQL views/tables populated. Spot-check: `SELECT COUNT(*) FROM user_features` returns ≥300.
3. `quarto render analysis/02_models.qmd` produces a notebook with the DR coefficient + 95% CI on HDI, both robustness table and negative-control coefficient. No errors.
4. `quarto render analysis/04_methodology_paper.qmd` produces `docs/methodology.pdf` ≥10 pages.
5. Open `docs/index.html` via `python -m http.server 8910 --directory docs`. The 6-slide deck renders, the map heatmaps show non-uniform color (real data), and slide-2 filter dropdown updates the maps. No console errors.
6. Open `docs/methods.html`. The methods deck renders with at least 8 slides, including the DAG, sample funnel, and E-value chart.
7. **Sanity check** on the analysis: Golden Route locations (Tokyo, Kyoto, Osaka) appear as low-remoteness cells; off-the-beaten-path regions (Tōhoku interior, the San'in coast, remote islands) appear as high-remoteness cells. If this fails, the OSM POI definition needs tuning before any inferential claim is made.
8. **Reviewer dry-run:** a colleague can clone the repo, run `uv sync && python pipeline/run.py`, and reproduce the headline coefficient within numerical bootstrap noise.

---

## 14. Open questions (to revisit during execution, not blocking)

- **Flickr quadtree starting bbox subdivision threshold:** when to subdivide vs. paginate? Probably subdivide when single-bbox results > 3500 (margin under 4000 cap).
- **Hour-of-day negative control timezone:** convert photo timestamps to local Japan time, or origin-country time? Probably destination-local (asking what hour they were in Japan, not when they would have been awake at home).
- **DR vs DML (double machine learning):** if `econml` is included anyway, DML with random-forest nuisance might be more modern. Decision deferred to day 6–7 based on sample size.
- **Composite indicator construction:** PCA on standardized HDI, GDP_PC_PPP, LPI, UHC, WGI_GOVE. Whether to include WGI given correlation with HDI > 0.9 is open.
- **H3 resolution:** Japan uses **H3 r6** (~10k cells over the country). Whether a finer resolution (r7) better separates within-city beaten-path gradients is open; revisit only if r6 cells prove too coarse for the urban Golden-Route hubs.
