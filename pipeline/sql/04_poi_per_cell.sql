-- 04_poi_per_cell — POI counts per H3 r6 cell: total + by category.
--
-- LEFT JOIN from cells so every in-universe cell gets a row; photo-only cells
-- with no POIs come through with poi_count = 0 (these become the most-remote
-- cells downstream).
CREATE OR REPLACE TABLE poi_per_cell AS
WITH binned AS (
    SELECT
        h3_latlng_to_cell(lat, lon, getvariable('h3_res')) AS h3_r6,
        primary_key
    FROM read_parquet(getvariable('pois_path'))
    WHERE lat IS NOT NULL AND lon IS NOT NULL
)
SELECT
    c.h3_r6,
    COUNT(b.h3_r6)                                             AS poi_count,
    COUNT(*) FILTER (WHERE b.primary_key = 'tourism')          AS poi_tourism,
    COUNT(*) FILTER (WHERE b.primary_key = 'amenity')          AS poi_amenity,
    COUNT(*) FILTER (WHERE b.primary_key = 'shop')             AS poi_shop,
    COUNT(*) FILTER (WHERE b.primary_key = 'public_transport') AS poi_public_transport
FROM cells c
LEFT JOIN binned b ON b.h3_r6 = c.h3_r6
GROUP BY c.h3_r6;
