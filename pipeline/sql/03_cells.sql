-- 03_cells — the universe of analyzable H3 r6 cells = cells containing >=1 POI.
--
-- This both (a) defines "reachable / visitable" territory (a cell with no OSM
-- infrastructure isn't a plausible visit target) and (b) CLIPS TO JAPAN for free:
-- the Geofabrik Japan PBF contains only Japan POIs, so POI-bearing cells exclude
-- the Korea (Seoul) and Russian Far East (Vladivostok) territory that the
-- rectangular Flickr bbox unavoidably sweeps in. Every downstream photo count
-- INNER JOINs to this table, so non-Japan photos are dropped automatically.
--
-- One leak remains: the Geofabrik Japan extract carries a small buffer past the
-- maritime border that includes ~2 Busan (KR) POI cells across the Tsushima
-- Strait (111 cohort visits, 0.04%). We drop them with a targeted box exclusion
-- (lat 34.9-35.4, lon 128.8-129.3) that brackets Busan WITHOUT touching Japanese
-- Tsushima island (34.1-34.7N), which sits south of the box.
--
-- Trade-off: genuine zero-POI Japan wilderness cells that have photos but no OSM
-- POIs are excluded (they were 0.2% of the prior union and not plausible visit
-- targets). If we later want them back, add a Japan admin-boundary polygon +
-- ST_Contains and union photo cells whose centroid falls inside it.
--
-- area_km2 is the full H3 hex area; coastal clipping to land area is a refinement.
CREATE OR REPLACE TABLE cells AS
WITH poi_cells AS (
    SELECT DISTINCT h3_latlng_to_cell(lat, lon, getvariable('h3_res')) AS h3_r6
    FROM read_parquet(getvariable('pois_path'))
    WHERE lat IS NOT NULL AND lon IS NOT NULL
)
SELECT
    h3_r6,
    h3_cell_to_lat(h3_r6)       AS centroid_lat,
    h3_cell_to_lng(h3_r6)       AS centroid_lon,
    h3_cell_area(h3_r6, 'km^2') AS area_km2
FROM poi_cells
-- Busan (KR) bleed exclusion — JAPAN ONLY. The box is gated on dest_iso3 so
-- other destinations (georgia today, or a future South Korea build where this
-- box IS Busan) are untouched. Margins are half a cell (~0.05 deg) wider than
-- the bleed cells so a re-extraction or --h3-res change can't shift a cell
-- CENTER just past the edge and sneak the leak back in; Japanese Tsushima
-- (<= 34.72N) stays safely south of the box.
WHERE NOT (
        getvariable('dest_iso3') = 'JPN'
    AND h3_cell_to_lat(h3_r6) BETWEEN 34.85 AND 35.45
    AND h3_cell_to_lng(h3_r6) BETWEEN 128.75 AND 129.35
);
