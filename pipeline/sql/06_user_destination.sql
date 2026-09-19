-- 06_user_destination — the tourist analysis cohort.
--
-- Tourist filter (SPEC §2): keep users whose STATED profile country is non-null
-- and NOT the destination (residents dropped), with at least min_photos photos
-- AND min_cells distinct cells *inside the destination*. "Inside the destination"
-- = photo cell is in `cells` (POI-bearing Japan cells), which clips out the
-- Korea/Russia bbox bleed. trip_year = year of the user's first in-Japan photo.
CREATE OR REPLACE TABLE user_destination AS
WITH japan_photos AS (
    SELECT
        p.user_id_hash,
        p.h3_r6,
        p.taken_ts,
        p.taken_month,
        EXTRACT(year FROM p.taken_ts) AS yr
    FROM photos p
    INNER JOIN cells c ON c.h3_r6 = p.h3_r6   -- clip to Japan (POI-bearing) cells
    WHERE p.taken_ts IS NOT NULL
),
in_window AS (
    SELECT * FROM japan_photos
    WHERE yr BETWEEN getvariable('year_min') AND getvariable('year_max')
),
agg AS (
    SELECT
        user_id_hash,
        COUNT(*)                 AS n_japan_photos,
        COUNT(DISTINCT h3_r6)    AS n_japan_cells,
        MIN(yr)                  AS trip_year
    FROM in_window
    GROUP BY user_id_hash
)
SELECT
    u.user_id_hash,
    u.stated_country_iso AS origin_iso,
    a.trip_year,
    a.n_japan_photos,
    a.n_japan_cells
FROM agg a
JOIN users u ON u.user_id_hash = a.user_id_hash
WHERE u.stated_country_iso IS NOT NULL
  AND u.stated_country_iso <> getvariable('dest_iso3')          -- tourists only
  AND a.n_japan_photos >= getvariable('min_photos')
  AND a.n_japan_cells  >= getvariable('min_cells');
