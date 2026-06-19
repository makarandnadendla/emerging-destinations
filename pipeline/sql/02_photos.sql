-- 02_photos — one modeled row per geotagged photo, with H3 r6 cell + local time.
--
-- Flickr `date_taken` is camera-LOCAL time, i.e. destination-local for photos
-- taken in the destination. That is exactly what the hour-of-day negative
-- control wants (the hour the traveler was active in Japan), so taken_ts is
-- kept as-is and hour / month are exposed for the placebo + seasonal control.
CREATE OR REPLACE TABLE photos AS
SELECT
    p.photo_id,
    sha256(p.user_id)                                      AS user_id_hash,
    p.lat,
    p.lon,
    p.taken_ts,
    EXTRACT(hour  FROM p.taken_ts)                         AS taken_hour,
    EXTRACT(month FROM p.taken_ts)                         AS taken_month,
    h3_latlng_to_cell(p.lat, p.lon, getvariable('h3_res')) AS h3_r6
FROM raw.photos p
WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL;
