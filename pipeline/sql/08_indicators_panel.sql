-- 08_indicators_panel — pivot the long indicators_raw parquet into a wide
-- country x year panel keyed (country_iso3, year). One column per indicator.
-- Year-matched to the user's first-photo year downstream in 09_user_features.
CREATE OR REPLACE TABLE indicators_panel AS
SELECT
    country_iso AS country_iso3,
    year,
    MAX(value) FILTER (WHERE indicator = 'HDI')            AS hdi,
    MAX(value) FILTER (WHERE indicator = 'GDP_PC_PPP')     AS gdp_pc_ppp,
    MAX(value) FILTER (WHERE indicator = 'LPI')            AS lpi,
    MAX(value) FILTER (WHERE indicator = 'UHC')            AS uhc,
    MAX(value) FILTER (WHERE indicator = 'WGI_GOV_EFFECT') AS wgi_gov_effect
FROM read_parquet(getvariable('indicators_path'))
GROUP BY country_iso, year;
