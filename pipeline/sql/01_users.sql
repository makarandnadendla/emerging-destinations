-- 01_users — one modeled row per Flickr user in the destination dataset.
--
-- Reads raw.photos (geotagged photos inside the destination bbox) and
-- raw.user_geocodes (Nominatim-resolved profile location -> ISO3).
--
-- LIMITATION: modal_country_iso / agree_flag / n_countries_visited need each
-- user's WORLDWIDE photo distribution, which the bbox-limited extraction does
-- NOT capture. They are left NULL pending an optional global per-user photo
-- pull. Residency is therefore decided by STATED profile location, which the
-- SPEC designates as the primary signal ("stated location wins").
CREATE OR REPLACE TABLE users AS
WITH photo_agg AS (
    SELECT
        p.user_id,
        any_value(p.owner_name) AS owner_name,
        COUNT(*)                AS total_photos,
        COUNT(DISTINCT h3_latlng_to_cell(p.lat, p.lon, getvariable('h3_res'))) AS n_cells
    FROM raw.photos p
    WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL
    GROUP BY p.user_id
),
geo AS (
    SELECT
        u.user_id,
        u.location_raw,
        g.country_iso3 AS stated_country_iso
    FROM raw.users u
    LEFT JOIN raw.user_geocodes g ON g.location_raw = u.location_raw
)
SELECT
    sha256(pa.user_id)        AS user_id_hash,
    pa.user_id               AS user_id,            -- internal only; never exported to repo
    geo.stated_country_iso   AS stated_country_iso,
    CAST(NULL AS VARCHAR)    AS modal_country_iso,  -- TODO: requires global per-user photo pull
    CAST(NULL AS BOOLEAN)    AS agree_flag,         -- TODO: stated vs modal agreement
    pa.total_photos          AS total_photos,
    pa.n_cells               AS n_cells,
    CAST(NULL AS INTEGER)    AS n_countries_visited -- TODO: requires global per-user photo pull
FROM photo_agg pa
LEFT JOIN geo ON geo.user_id = pa.user_id;
