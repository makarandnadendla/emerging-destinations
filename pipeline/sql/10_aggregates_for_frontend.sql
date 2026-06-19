-- 10_aggregates_for_frontend — aggregated rollups for the slide-deck JSON.
-- NO row-level user data leaves this stage; everything here is grouped.
-- Exported to docs/data/aggregates.json downstream.

-- Cell-level rollup for the remoteness heatmap + visit overlay.
CREATE OR REPLACE TABLE agg_cells AS
SELECT
    cr.h3_r6,
    h3_cell_to_lat(cr.h3_r6) AS centroid_lat,
    h3_cell_to_lng(cr.h3_r6) AS centroid_lon,
    cr.remoteness_norm,
    cr.poi_count,
    COALESCE(v.n_visits, 0) AS n_visits,
    COALESCE(v.n_users, 0)  AS n_users
FROM cell_remoteness cr
LEFT JOIN (
    SELECT h3_r6,
           SUM(photo_count)            AS n_visits,
           COUNT(DISTINCT user_id_hash) AS n_users
    FROM user_cell_visits
    GROUP BY h3_r6
) v ON v.h3_r6 = cr.h3_r6;

-- Origin-country rollup: cohort size + mean outcome + mean HDI. No user rows.
CREATE OR REPLACE TABLE agg_origin AS
SELECT
    origin_iso,
    COUNT(*)               AS n_users,
    AVG(y_mean_remoteness) AS mean_remoteness,
    AVG(hdi)               AS mean_hdi,
    AVG(gdp_pc_ppp)        AS mean_gdp_pc_ppp
FROM user_features
GROUP BY origin_iso;

-- One-row headline summary.
CREATE OR REPLACE TABLE agg_summary AS
SELECT
    (SELECT COUNT(*) FROM user_features)                       AS n_analysis_users,
    (SELECT COUNT(DISTINCT origin_iso) FROM user_features)     AS n_origin_countries,
    (SELECT COUNT(*) FROM cells)                               AS n_cells,
    (SELECT AVG(y_mean_remoteness) FROM user_features)         AS mean_user_remoteness,
    (SELECT corr(hdi, y_mean_remoteness) FROM user_features)   AS raw_corr_hdi_remoteness;
