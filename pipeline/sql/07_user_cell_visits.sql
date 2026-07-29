-- 07_user_cell_visits — one row per (analysis user x Japan cell) visited.
--
-- Restricted to the cohort in user_destination and to Japan cells (INNER JOIN to
-- cell_remoteness, which only contains POI-bearing Japan cells). Carries the
-- cell's remoteness so the user-level outcome in 09 is a simple weighted mean.
--
-- Windowed to the analysis years so the outcome is measured on the same trip
-- set as cohort membership (06), the trip_year treatment match, and the
-- negative-control outcome (09) — out-of-window photos in the raw pull
-- (taken-date boundary leakage) must not leak into y_mean_remoteness.
CREATE OR REPLACE TABLE user_cell_visits AS
SELECT
    p.user_id_hash,
    p.h3_r6,
    COUNT(*)            AS photo_count,
    MIN(p.taken_ts)     AS first_visit_ts,
    MAX(p.taken_ts)     AS last_visit_ts,
    cr.remoteness_norm,
    cr.poi_count
FROM photos p
JOIN user_destination ud ON ud.user_id_hash = p.user_id_hash   -- analysis cohort only
JOIN cell_remoteness  cr ON cr.h3_r6        = p.h3_r6          -- Japan cells only (clip)
WHERE p.taken_ts IS NOT NULL
  AND EXTRACT(year FROM p.taken_ts)
      BETWEEN getvariable('year_min') AND getvariable('year_max')
GROUP BY p.user_id_hash, p.h3_r6, cr.remoteness_norm, cr.poi_count;
