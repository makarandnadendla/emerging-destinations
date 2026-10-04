-- 02_photos — one modeled row per geotagged photo, with H3 r6 cell + local time.
--
-- Flickr `date_taken` is camera-LOCAL time, i.e. destination-local for photos
-- taken in the destination. That is exactly what the hour-of-day negative
-- control wants (the hour the traveler was active in Japan), so taken_ts is
-- kept as-is and hour / month are exposed for the placebo + seasonal control.
--
-- Flickr `datetakengranularity` (0=exact, 4=month known, 6=year known,
-- 8=circa) marks photos whose sub-date time is FAKE (stored as 00:00:00).
-- All ~9k granularity>0 photos in the Japan pull have hour=0; letting them
-- through would inject a spurious midnight spike into the negative-control
-- outcome. taken_hour is therefore NULL unless the timestamp is exact, and
-- taken_month is NULL when even the month is unknown (granularity > 4).
CREATE OR REPLACE TABLE photos AS
SELECT
    p.photo_id,
    sha256(p.user_id)                                      AS user_id_hash,
    p.lat,
    p.lon,
    p.taken_ts,
    p.granularity,
    CASE WHEN p.granularity = 0
         THEN EXTRACT(hour  FROM p.taken_ts) END           AS taken_hour,
    CASE WHEN p.granularity <= 4
         THEN EXTRACT(month FROM p.taken_ts) END           AS taken_month,
    h3_latlng_to_cell(p.lat, p.lon, getvariable('h3_res')) AS h3_r6
FROM raw.photos p
WHERE p.lat IS NOT NULL AND p.lon IS NOT NULL;
