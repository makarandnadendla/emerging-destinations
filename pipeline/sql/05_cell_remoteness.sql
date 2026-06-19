-- 05_cell_remoteness — remoteness as inverse POI density.
--
--   remoteness_raw  = -ln(1 + poi_count)
--       0 POIs       ->  0      (least negative = most remote)
--       many POIs    -> large negative (densest = least remote)
--   remoteness_norm = min-max scaled to [0,1] across the destination,
--       1 = most remote (fewest POIs), 0 = least remote (densest cell)
--   is_zero_poi_cell = TRUE when the cell has no POIs at all
CREATE OR REPLACE TABLE cell_remoteness AS
WITH r AS (
    SELECT
        pc.h3_r6,
        pc.poi_count,
        -ln(1 + pc.poi_count) AS remoteness_raw
    FROM poi_per_cell pc
),
bounds AS (
    SELECT min(remoteness_raw) AS lo, max(remoteness_raw) AS hi FROM r
)
SELECT
    r.h3_r6,
    r.poi_count,
    r.remoteness_raw,
    CASE WHEN b.hi = b.lo THEN 0.0
         ELSE (r.remoteness_raw - b.lo) / (b.hi - b.lo)
    END AS remoteness_norm,
    (r.poi_count = 0) AS is_zero_poi_cell
FROM r CROSS JOIN bounds b;
