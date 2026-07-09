-- 09_user_features — one row per analysis user: headline outcome + treatment +
-- robustness indicators, year-matched to trip_year.
--
-- Outcome  Y_i = photo-weighted mean cell remoteness (SPEC Model 1):
--          sum(photo_count * remoteness_norm) / sum(photo_count).
-- Treatment = hdi (continuous). Robustness vars = gdp_pc_ppp, lpi, uhc, wgi.
--
-- Indicators are carried forward per-column to fill the biennial LPI gaps (LPI is
-- published every other year): `filled` takes the most recent non-null value at or
-- before each year. Then matched on (origin_iso, trip_year).
--
-- Negative-control outcome (SPEC §147, DAG node HOUR): y_neg_hour is the per-user
-- circular mean of in-Japan photo hour-of-day, with trip_month_modal as the
-- seasonal control. taken_hour is destination-local (see 02_photos). The mean is
-- circular (atan2 of mean sin/cos) so 23:00 and 01:00 don't average to noon; the
-- result is wrapped back into [0, 24). Stage A regresses y_neg_hour on HDI +
-- confounders + month and expects ~0 — a non-null coefficient flags residual
-- confounding the back-door set didn't close.
CREATE OR REPLACE TABLE user_features AS
WITH filled AS (
    SELECT
        country_iso3,
        year,
        last_value(hdi            IGNORE NULLS) OVER w AS hdi,
        last_value(gdp_pc_ppp     IGNORE NULLS) OVER w AS gdp_pc_ppp,
        last_value(lpi            IGNORE NULLS) OVER w AS lpi,
        last_value(uhc            IGNORE NULLS) OVER w AS uhc,
        last_value(wgi_gov_effect IGNORE NULLS) OVER w AS wgi_gov_effect
    FROM indicators_panel
    WINDOW w AS (
        PARTITION BY country_iso3 ORDER BY year
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),
outcome AS (
    SELECT
        user_id_hash,
        SUM(photo_count)                                  AS n_photos,
        COUNT(*)                                          AS n_cells,
        SUM(photo_count * remoteness_norm) / SUM(photo_count) AS y_mean_remoteness
    FROM user_cell_visits
    GROUP BY user_id_hash
),
-- Negative-control placebo outcome: circular mean in-Japan photo hour + modal month.
-- Same Japan clip + window as the cohort (06_user_destination) so it is defined on
-- exactly the analysis trip, not the user's whole Flickr history.
neg AS (
    SELECT
        p.user_id_hash,
        MOD(
            atan2(AVG(sin(2 * pi() * p.taken_hour / 24.0)),
                  AVG(cos(2 * pi() * p.taken_hour / 24.0))) / (2 * pi()) * 24.0 + 24.0,
            24.0
        )                                                 AS y_neg_hour,
        mode(p.taken_month)                               AS trip_month_modal,
        COUNT(*)                                          AS n_neg_photos
    FROM photos p
    INNER JOIN cells c ON c.h3_r6 = p.h3_r6              -- clip to Japan POI cells
    WHERE p.taken_ts IS NOT NULL
      AND EXTRACT(year FROM p.taken_ts) BETWEEN 2012 AND 2019
    GROUP BY p.user_id_hash
)
SELECT
    ud.user_id_hash,
    ud.origin_iso,
    ud.trip_year,
    o.n_photos,
    o.n_cells,
    o.y_mean_remoteness,
    n.y_neg_hour,                 -- negative-control outcome (SPEC §147)
    n.trip_month_modal,           -- seasonal control for the negative-control regression
    -- Home-resolution cross-check (SPEC §97): origin_iso above is the STATED home;
    -- agree_flag = (stated == worldwide-modal home), where the modal excludes
    -- destination photos unless stated home IS the destination (see 01_users.sql).
    -- Stage A uses agree_flag to drop / separately analyze conflicts rather than
    -- trusting the pooled treatment label. NULL agree_flag = no usable modal
    -- (no worldwide geo, or only-destination photos for a non-resident).
    u.modal_country_iso,
    u.agree_flag,
    u.n_countries_visited,
    f.hdi,
    f.gdp_pc_ppp,
    f.lpi,
    f.uhc,
    f.wgi_gov_effect
FROM user_destination ud
JOIN outcome o ON o.user_id_hash = ud.user_id_hash
LEFT JOIN neg n ON n.user_id_hash = ud.user_id_hash
LEFT JOIN users u ON u.user_id_hash = ud.user_id_hash
LEFT JOIN filled f
       ON f.country_iso3 = ud.origin_iso
      AND f.year         = ud.trip_year;
