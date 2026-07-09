-- 01_users — one modeled row per Flickr user in the destination dataset.
--
-- Reads raw.photos (geotagged photos inside the destination bbox) and
-- raw.user_geocodes (Nominatim-resolved profile location -> ISO3).
--
-- modal_country_iso / agree_flag / n_countries_visited come from each user's
-- WORLDWIDE photo distribution, which the bbox-limited base extraction cannot
-- see. The optional global pull (extract_global_photos.py) fills the per-country
-- tally into raw.user_country_counts, surfaced here via the `user_modal_counts`
-- view (bootstrap in run.py falls back to an empty view when the pull hasn't
-- run, so these go NULL). Residency is still decided by STATED profile location
-- (SPEC: "stated location wins"); modal is the behavior-based cross-check for
-- the home-resolution agreement refutation, not a replacement.
--
-- Destination-skew rule: DESTINATION photos are excluded from the home vote —
-- tourists who shot Japan more than home would otherwise get modal = JPN
-- mechanically (53% of raw conflicts in the first pull) — UNLESS the user's
-- stated home IS the destination, where those photos are concordant evidence,
-- not skew. A non-destination user with only-destination photos gets NULL
-- (no usable home signal). n_countries_visited still counts ALL countries.
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
),
modal AS (
    SELECT
        user_id,
        arg_max(country_iso3, photo_count)               AS modal_all,   -- incl. destination
        arg_max(country_iso3, photo_count)
            FILTER (WHERE country_iso3 <> getvariable('dest_iso3'))
                                                         AS modal_excl,  -- destination removed
        COUNT(DISTINCT country_iso3)                     AS n_countries
    FROM user_modal_counts
    GROUP BY user_id
)
SELECT
    sha256(pa.user_id)        AS user_id_hash,
    pa.user_id               AS user_id,            -- internal only; never exported to repo
    geo.stated_country_iso   AS stated_country_iso,
    CASE WHEN geo.stated_country_iso = getvariable('dest_iso3')
         THEN m.modal_all                            -- resident: destination photos are concordant
         ELSE m.modal_excl                           -- tourist: exclude destination-skew
    END                      AS modal_country_iso,
    CASE WHEN geo.stated_country_iso IS NOT NULL
          AND (CASE WHEN geo.stated_country_iso = getvariable('dest_iso3')
                    THEN m.modal_all ELSE m.modal_excl END) IS NOT NULL
         THEN geo.stated_country_iso =
              (CASE WHEN geo.stated_country_iso = getvariable('dest_iso3')
                    THEN m.modal_all ELSE m.modal_excl END)
    END                      AS agree_flag,         -- stated vs modal (NULL when either is missing)
    pa.total_photos          AS total_photos,
    pa.n_cells               AS n_cells,
    m.n_countries            AS n_countries_visited -- distinct worldwide photo countries (incl. dest)
FROM photo_agg pa
LEFT JOIN geo ON geo.user_id = pa.user_id
LEFT JOIN modal m ON m.user_id = pa.user_id;
