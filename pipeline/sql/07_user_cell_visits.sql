-- 07_user_cell_visits — one row per (analysis user x Japan cell) visited.
--
-- Restricted to the cohort in user_destination and to Japan cells (INNER JOIN to
-- cell_remoteness, which only contains POI-bearing Japan cells). Carries the
-- cell's remoteness so the user-level outcome in 09 is a simple weighted mean.
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
GROUP BY p.user_id_hash, p.h3_r6, cr.remoteness_norm, cr.poi_count;
