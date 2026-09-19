# Data Dictionary — Stage-T Warehouse (`warehouse_japan.duckdb`)

The transform layer builds one DuckDB warehouse from the raw extraction DB. Each
numbered SQL step (`pipeline/sql/NN_*.sql`) materializes one table; this dictionary
documents every table and column in build order.

## Conventions

- **`h3_r6`** — an [H3](https://h3geo.org) resolution-6 cell id (64-bit, ~36 km²). The
  spatial unit for the whole analysis.
- **`user_id_hash`** — SHA-256 of the raw Flickr `user_id`. The anonymized key used by
  every downstream table. The raw `user_id` lives **only** in `users` and is never exported.
- **Export firewall** — only the `agg_*` tables (step 10) leave the warehouse (to
  `docs/data/aggregates.json`). No row-level user data is published.
- **`NULL`** generally means "not reported / not resolvable" (e.g. a missing indicator,
  or a user the optional global photo pull hasn't covered).
- Cohort = the tourist analysis set defined in step 06 (`user_destination`): a resolved
  non-destination stated origin, ≥5 in-destination photos in ≥2 distinct cells.

---

## 01 · `users`
**Grain:** one row per Flickr user appearing in the destination dataset. **Built by:** `01_users.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `user_id_hash` | VARCHAR | SHA-256 of the raw Flickr user id; anonymized join key used everywhere downstream. |
| `user_id` | VARCHAR | Raw Flickr user id. **Internal only — never exported to the public repo.** |
| `stated_country_iso` | VARCHAR | ISO 3166-1 alpha-3 of the user's stated profile location (Nominatim-geocoded). Primary origin signal. |
| `modal_country_iso` | VARCHAR | ISO-3 modal home from the user's *worldwide* geotagged photos (global pull), with **destination photos excluded from the vote unless the user's stated home IS the destination** (a resident's destination photos are concordant home evidence). Ties break deterministically (highest count, then alphabetical ISO-3). `NULL` if not pulled, or if a non-resident's only geotagged photos are destination photos (no independent home signal). |
| `agree_flag` | BOOLEAN | `TRUE` when `stated_country_iso = modal_country_iso`; `NULL` when either side is missing. |
| `total_photos` | BIGINT | Count of the user's geotagged photos inside the destination bbox. |
| `n_cells` | BIGINT | Distinct H3 r6 cells those photos fall in. |
| `n_countries_visited` | INTEGER | Distinct countries among the user's worldwide geotagged photos (global pull); `NULL` if not pulled. |

## 02 · `photos`
**Grain:** one row per geotagged photo inside the destination bbox. **Built by:** `02_photos.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `photo_id` | BIGINT | Flickr photo id (public). |
| `user_id_hash` | VARCHAR | Owner (hashed). |
| `lat` | DOUBLE | WGS84 latitude. |
| `lon` | DOUBLE | WGS84 longitude. |
| `taken_ts` | TIMESTAMP | Capture time, camera-**local** (= destination-local for in-destination photos). |
| `taken_hour` | BIGINT | Hour of day 0–23 from `taken_ts`. Input to the hour-of-day negative control. |
| `taken_month` | BIGINT | Month 1–12 from `taken_ts`. Seasonal control. |
| `h3_r6` | UBIGINT | H3 r6 cell containing `(lat, lon)`. |

## 03 · `cells`
**Grain:** one row per analyzable H3 r6 cell (= POI-bearing cells; also the implicit Japan clip). **Built by:** `03_cells.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `h3_r6` | UBIGINT | H3 r6 cell id. The "visitable" universe — only cells containing ≥1 OSM POI. |
| `centroid_lat` | DOUBLE | Cell center latitude. |
| `centroid_lon` | DOUBLE | Cell center longitude. |
| `area_km2` | DOUBLE | Full hexagon area (km²); ~33–37 depending on latitude. |

## 04 · `poi_per_cell`
**Grain:** one row per cell, with POI counts. **Built by:** `04_poi_per_cell.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `h3_r6` | UBIGINT | Cell id. |
| `poi_count` | BIGINT | Total filtered OSM POIs in the cell. |
| `poi_tourism` | BIGINT | POIs tagged `tourism=*`. |
| `poi_amenity` | BIGINT | POIs tagged `amenity` (restaurant/cafe/bar/fast_food/hospital/pharmacy/atm/bank/fuel). |
| `poi_shop` | BIGINT | POIs tagged `shop` (supermarket/convenience/bakery/mall). |
| `poi_public_transport` | BIGINT | POIs tagged `public_transport=*`. |

## 05 · `cell_remoteness`
**Grain:** one row per cell, with the remoteness score. **Built by:** `05_cell_remoteness.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `h3_r6` | UBIGINT | Cell id. |
| `poi_count` | BIGINT | Carried from `poi_per_cell`. |
| `remoteness_raw` | DOUBLE | `−ln(1 + poi_count)`. Less negative = more remote. |
| `remoteness_norm` | DOUBLE | Min-max scaled to `[0,1]` across **all** cells; 1 = most remote, 0 = densest. *Population-dependent: the anchors are the global min/max, so changing the cell set rebases every value.* |
| `is_zero_poi_cell` | BOOLEAN | `TRUE` if `poi_count = 0`. Always `FALSE` in the current build (the universe is POI-bearing). |

## 06 · `user_destination`
**Grain:** one row per cohort user (the tourist analysis set). **Built by:** `06_user_destination.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `user_id_hash` | VARCHAR | Cohort member (hashed). |
| `origin_iso` | VARCHAR | Assigned origin = `stated_country_iso` (≠ destination by cohort rule). |
| `trip_year` | BIGINT | Year of the user's first in-destination photo (within 2012–2019). |
| `n_japan_photos` | BIGINT | In-destination geotagged photos (after the Japan clip). |
| `n_japan_cells` | BIGINT | Distinct destination cells visited. |

## 07 · `user_cell_visits`
**Grain:** one row per (cohort user × destination cell visited). **Built by:** `07_user_cell_visits.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `user_id_hash` | VARCHAR | Cohort member (hashed). |
| `h3_r6` | UBIGINT | Visited destination cell. |
| `photo_count` | BIGINT | Photos this user took in this cell. |
| `first_visit_ts` | TIMESTAMP | Earliest photo timestamp in the cell. |
| `last_visit_ts` | TIMESTAMP | Latest photo timestamp in the cell. |
| `remoteness_norm` | DOUBLE | Carried cell remoteness (so the user outcome is a simple weighted mean). |
| `poi_count` | BIGINT | Carried POI count. |

## 08 · `indicators_panel`
**Grain:** one row per (country × year). **Built by:** `08_indicators_panel.sql` (pivots the long indicator parquet wide). All indicator columns are `NULL` where the source didn't report that country-year.

| Column | Type | Description |
| --- | --- | --- |
| `country_iso3` | VARCHAR | ISO 3166-1 alpha-3 country code (join key to a user's origin). |
| `year` | BIGINT | Calendar year (2012–2019 window). |
| `hdi` | DOUBLE | **Human Development Index** (UNDP). Composite of life expectancy, education, income. 0–1, higher = more developed. **Headline treatment.** |
| `gdp_pc_ppp` | DOUBLE | **GDP per capita, PPP** (World Bank). Economic output per person, cost-of-living adjusted. Int'l dollars. Alt treatment / economic control. |
| `lpi` | DOUBLE | **Logistics Performance Index** (World Bank). Trade-logistics quality (customs, infrastructure, timeliness). 1–5. **Biennial** — `NULL` in odd years. |
| `uhc` | DOUBLE | **Universal Health Coverage** service coverage index (WHO). Essential-service coverage. 0–100. |
| `wgi_gov_effect` | DOUBLE | **WGI Government Effectiveness** (World Bank). Quality of public services/civil service/policy. ≈ −2.5 to +2.5. |

## 09 · `user_features`
**Grain:** one row per cohort user — outcome + treatment + controls, year-matched to the trip. **Built by:** `09_user_features.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `user_id_hash` | VARCHAR | Cohort member (hashed). |
| `origin_iso` | VARCHAR | Assigned origin country (ISO-3). |
| `trip_year` | BIGINT | Year of first in-destination photo. |
| `n_photos` | HUGEINT | Total photos summed across the user's visited cells. |
| `n_cells` | BIGINT | Distinct destination cells visited. |
| `y_mean_remoteness` | DOUBLE | **OUTCOME.** Photo-weighted mean cell remoteness = `Σ(photo_count·remoteness_norm) / Σ photo_count`. |
| `y_neg_hour` | DOUBLE | **Negative-control outcome.** Circular mean of in-destination photo hour-of-day, wrapped to `[0,24)`. |
| `trip_month_modal` | BIGINT | Modal photo month (1–12). Seasonal control for the negative-control regression. |
| `hdi` | DOUBLE | **Treatment.** Origin HDI, year-matched to `trip_year` with carry-forward. `NULL` for countries UNDP doesn't cover (e.g. Taiwan). |
| `gdp_pc_ppp` | DOUBLE | Alt treatment / control, same year-match + carry-forward. |
| `lpi` | DOUBLE | Alt treatment, same year-match + carry-forward (fills biennial gaps). |
| `uhc` | DOUBLE | Alt treatment, same year-match + carry-forward. |
| `wgi_gov_effect` | DOUBLE | Alt treatment, same year-match + carry-forward. |

## 10a · `agg_cells`
**Grain:** one row per cell — rollup for the frontend heatmap. No user rows. **Built by:** `10_aggregates_for_frontend.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `h3_r6` | UBIGINT | Cell id. |
| `centroid_lat` | DOUBLE | Cell center latitude. |
| `centroid_lon` | DOUBLE | Cell center longitude. |
| `remoteness_norm` | DOUBLE | Cell remoteness `[0,1]`. |
| `poi_count` | BIGINT | POIs in the cell. |
| `n_visits` | HUGEINT | Total cohort photos taken in the cell. |
| `n_users` | BIGINT | Distinct cohort users who photographed the cell. |

## 10b · `agg_origin`
**Grain:** one row per origin country — rollup. No user rows. **Built by:** `10_aggregates_for_frontend.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `origin_iso` | VARCHAR | Origin country (ISO-3). |
| `n_users` | BIGINT | Cohort size from this origin. |
| `mean_remoteness` | DOUBLE | Mean of `y_mean_remoteness` over the origin's users. |
| `mean_hdi` | DOUBLE | Mean origin HDI (`NULL` where uncovered, e.g. Taiwan). |
| `mean_gdp_pc_ppp` | DOUBLE | Mean origin GDP/cap PPP. |

## 10c · `agg_summary`
**Grain:** one row — headline figures. **Built by:** `10_aggregates_for_frontend.sql`.

| Column | Type | Description |
| --- | --- | --- |
| `n_analysis_users` | BIGINT | Cohort size (rows in `user_features`). |
| `n_origin_countries` | BIGINT | Distinct origin countries in the cohort. |
| `n_cells` | BIGINT | Analyzable cells (rows in `cells`). |
| `mean_user_remoteness` | DOUBLE | Mean outcome across the cohort. |
| `raw_corr_hdi_remoteness` | DOUBLE | Pooled Pearson correlation of HDI and the outcome (descriptive only — not the causal estimate). |

---

## Appendix · Key source tables (Stage E, raw DB — not exported)

These live in `data/japan.duckdb` and feed the transform; documented for provenance.

| Table | Grain | Role |
| --- | --- | --- |
| `raw.photos` | one geotagged photo | Source for `photos` / `users`. |
| `raw.users` | one Flickr user | Profile `location_raw` → stated origin. |
| `raw.user_geocodes` | one distinct location string | Nominatim → ISO-3 (`stated_country_iso`). |
| `raw.user_global_status` | one user | Worldwide modal country + counts (`modal_country_iso`, `n_countries_visited`). |
| `raw.user_country_counts` | one (user × country) | Per-country worldwide photo tally; source for any modal definition. |
| `frontier` | one quadtree tile | Flickr extraction checkpoint/resume state. |
