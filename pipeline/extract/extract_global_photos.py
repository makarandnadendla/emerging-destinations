"""
Global per-user photo pull -> modal home country (the stated-vs-modal cross-check).

SPEC section 4.2 / 4.3 "Modal-photo-country (audit)": for each cohort user, look
at their WORLDWIDE geotagged photos (not just the ones inside the destination
bbox) and take the modal country. That second, behavior-based home estimate is
cross-checked against the stated profile country in the Stage-A home-resolution
agreement refutation (analysis/sensitivity.py, DAG node HOMERES). The base
extraction is bbox-limited to the destination, so it cannot see a user's home
photos — this script fills that gap with a per-user `flickr.photos.search`
(user_id + has_geo, no bbox).

Country assignment is OFFLINE: each photo's lat/lon is reverse-geocoded with
reverse_geocoder (a k-d tree over GeoNames cities, no network, ISO-2), then mapped
to ISO-3 through the same pycountry path the stated-location geocoder uses, so the
two home estimates are directly comparable. We never reverse-geocode via Nominatim
here (millions of points at 1 req/s is infeasible) and we never store raw
worldwide photos — only the compact per-(user, country) tally.

State lives in the destination's raw DuckDB (data/<dest>.duckdb):
  user_global_status  (user_id PK, n_geo_photos, n_countries, modal_iso3, ...)
  user_country_counts (user_id, country_iso3, photo_count)

Target users = the cohort superset: a resolved non-destination stated country AND
at least `min_photos` photos inside the destination (so we don't spend API calls
on users who can never enter the cohort). Re-running consumes only NOT-yet-pulled
users; soft failures (deleted users) record status='not_found' so they aren't
retried forever. `--reset-errors` re-queues status='error'.

**Run AFTER extract_flickr_photos.py + extract_flickr_users.py + geocode_user_locations.py.**
DuckDB allows one writer at a time, so don't run it alongside another extractor.

Usage:
    uv run python pipeline/extract/extract_global_photos.py
    uv run python pipeline/extract/extract_global_photos.py --max-iters 20
    uv run python pipeline/extract/extract_global_photos.py --reset-errors
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

FLICKR_API = "https://api.flickr.com/services/rest/"
RATE_LIMIT_SLEEP = 1.0
RETRY_BACKOFF = [1, 2, 4, 8, 16]
HTTP_TIMEOUT = 30
PROGRESS_EVERY = 20
MAX_PAGES = 8            # Flickr serves ~4000 results max; 4000 photos is ample for a modal


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    active = cfg["destination"]
    dest = cfg["destinations"][active]
    flickr = cfg.get("flickr", {})
    return {
        "db_path": Path(dest["db_path"]),
        "dest_iso3": dest["iso3"],
        "per_page": flickr.get("per_page", 500),
        "rate_limit_sleep": flickr.get("rate_limit_sleep", RATE_LIMIT_SLEEP),
        "min_photos": cfg["min_photos"],
    }


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def init_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS user_global_status (
            user_id       VARCHAR PRIMARY KEY,
            n_geo_photos  INTEGER,   -- worldwide geotagged photos seen (capped at MAX_PAGES*per_page)
            n_countries   INTEGER,   -- distinct countries among them
            modal_iso3    VARCHAR,   -- most frequent country (ISO-3)
            modal_count   INTEGER,
            modal_share   DOUBLE,    -- modal_count / n_geo_photos_geocoded
            capped        BOOLEAN,   -- hit Flickr's retrieval wall (>MAX_PAGES pages)
            status        VARCHAR,   -- ok | no_geo | not_found | error
            last_error    VARCHAR,
            fetched_at    TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS user_country_counts (
            user_id       VARCHAR,
            country_iso3  VARCHAR,
            photo_count   INTEGER,
            PRIMARY KEY (user_id, country_iso3)
        );
    """)


def iso2_to_iso3_map() -> dict:
    import pycountry
    return {c.alpha_2: c.alpha_3 for c in pycountry.countries}


def get_pending(con: duckdb.DuckDBPyConnection, dest_iso3: str, min_photos: int) -> list[str]:
    """Cohort superset not yet pulled: resolved non-destination stated country AND
    >= min_photos photos inside the destination."""
    rows = con.execute("""
        WITH jp AS (SELECT user_id, COUNT(*) AS n FROM photos GROUP BY user_id)
        SELECT DISTINCT u.user_id
        FROM users u
        JOIN user_geocodes g ON g.location_raw = u.location_raw
        JOIN jp ON jp.user_id = u.user_id
        LEFT JOIN user_global_status s ON s.user_id = u.user_id
        WHERE g.country_iso3 IS NOT NULL
          AND g.country_iso3 <> ?
          AND jp.n >= ?
          AND s.user_id IS NULL
        ORDER BY u.user_id
    """, [dest_iso3, min_photos]).fetchall()
    return [r[0] for r in rows]


def flickr_user_geo(api_key: str, user_id: str, page: int, per_page: int
                    ) -> tuple[str, dict | str]:
    """One page of a user's worldwide geotagged photos. status: ok|not_found|error."""
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "user_id": user_id,
        "has_geo": 1,
        "extras": "geo",
        "per_page": per_page,
        "page": page,
        "format": "json",
        "nojsoncallback": 1,
        "content_type": 1,
        "media": "photos",
    }
    last_err: str | None = None
    for backoff in RETRY_BACKOFF:
        try:
            r = requests.get(FLICKR_API, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if data.get("stat") == "ok":
                return "ok", data
            if data.get("stat") == "fail":
                code = data.get("code")
                if code in (1, 2, 5):       # user not found / unknown / deleted
                    return "not_found", data.get("message", "")
                if code == 100:
                    raise RuntimeError(f"invalid API key: {data.get('message')}")
                last_err = f"stat=fail code={code} msg={data.get('message')}"
            else:
                last_err = f"unexpected stat: {data.get('stat')}"
        except requests.HTTPError as e:
            sc = e.response.status_code if e.response is not None else None
            if sc and 400 <= sc < 500 and sc != 429:
                return "error", f"HTTP {sc}"
            last_err = str(e)
        except (requests.RequestException, RuntimeError, ValueError) as e:
            last_err = str(e)
        time.sleep(backoff)
    return "error", (last_err or "unknown error")[:500]


def coords_from_payload(data: dict) -> list[tuple[float, float]]:
    out = []
    for p in data.get("photos", {}).get("photo", []):
        try:
            lat = float(p.get("latitude") or 0)
            lon = float(p.get("longitude") or 0)
        except (TypeError, ValueError):
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        out.append((lat, lon))
    return out


def fetch_user_coords(api_key: str, user_id: str, per_page: int, sleep: float
                      ) -> tuple[str, list[tuple[float, float]], bool]:
    """Returns (status, coords, capped). status: ok|no_geo|not_found|error."""
    st, data = flickr_user_geo(api_key, user_id, 1, per_page)
    if st != "ok":
        return st, [], False
    photos = data.get("photos", {})
    total = int(photos.get("total", 0) or 0)
    pages = int(photos.get("pages", 0) or 0)
    if total == 0:
        return "no_geo", [], False
    coords = coords_from_payload(data)
    capped = pages > MAX_PAGES
    last_page = min(pages, MAX_PAGES)
    for page in range(2, last_page + 1):
        time.sleep(sleep + random.uniform(0, 0.2))
        st2, data2 = flickr_user_geo(api_key, user_id, page, per_page)
        if st2 != "ok":
            break  # partial pages are fine for a modal; keep what we have
        coords.extend(coords_from_payload(data2))
    return "ok", coords, capped


def tally_countries(geo, iso_map: dict, coords: list[tuple[float, float]]) -> dict:
    """Reverse-geocode coords offline and return {iso3: count}."""
    if not coords:
        return {}
    results = geo.query(coords)  # list of dicts with 'cc' (ISO-2)
    counts: dict[str, int] = {}
    for r in results:
        iso3 = iso_map.get((r.get("cc") or "").strip().upper())
        if iso3:
            counts[iso3] = counts.get(iso3, 0) + 1
    return counts


def upsert_user(con, user_id: str, status: str, counts: dict, capped: bool,
                n_coords: int, err: str | None = None) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.execute("DELETE FROM user_country_counts WHERE user_id = ?", [user_id])
    if status == "ok" and counts:
        rows = [(user_id, iso3, n) for iso3, n in counts.items()]
        con.executemany(
            "INSERT INTO user_country_counts (user_id, country_iso3, photo_count) VALUES (?, ?, ?)",
            rows,
        )
        modal_iso3 = max(counts, key=counts.get)
        modal_count = counts[modal_iso3]
        geocoded = sum(counts.values())
        con.execute("""
            INSERT INTO user_global_status
                (user_id, n_geo_photos, n_countries, modal_iso3, modal_count,
                 modal_share, capped, status, last_error, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'ok', NULL, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                n_geo_photos=excluded.n_geo_photos, n_countries=excluded.n_countries,
                modal_iso3=excluded.modal_iso3, modal_count=excluded.modal_count,
                modal_share=excluded.modal_share, capped=excluded.capped,
                status='ok', last_error=NULL, fetched_at=excluded.fetched_at
        """, [user_id, n_coords, len(counts), modal_iso3, modal_count,
              modal_count / geocoded if geocoded else None, capped, now])
    else:
        # no_geo (no geotagged photos), not_found, error, or ok-but-unmappable
        eff = "no_geo" if status == "ok" else status
        con.execute("""
            INSERT INTO user_global_status
                (user_id, n_geo_photos, n_countries, modal_iso3, modal_count,
                 modal_share, capped, status, last_error, fetched_at)
            VALUES (?, ?, 0, NULL, NULL, NULL, ?, ?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                n_geo_photos=excluded.n_geo_photos, n_countries=0,
                modal_iso3=NULL, modal_count=NULL, modal_share=NULL,
                capped=excluded.capped, status=excluded.status,
                last_error=excluded.last_error, fetched_at=excluded.fetched_at
        """, [user_id, n_coords, capped, eff, (str(err)[:500] if err else None), now])


def status_snapshot(con: duckdb.DuckDBPyConnection, label: str) -> None:
    rows = con.execute(
        "SELECT status, COUNT(*) FROM user_global_status GROUP BY status ORDER BY status"
    ).fetchall()
    n_agree = con.execute("""
        SELECT COUNT(*) FROM users u
        JOIN user_geocodes g ON g.location_raw = u.location_raw
        JOIN user_global_status s ON s.user_id = u.user_id
        WHERE s.status='ok' AND g.country_iso3 = s.modal_iso3
    """).fetchone()[0]
    n_ok = con.execute("SELECT COUNT(*) FROM user_global_status WHERE status='ok'").fetchone()[0]
    rate = f"{n_agree}/{n_ok} ({n_agree/n_ok:.1%})" if n_ok else "n/a"
    print(f"[{label}] by_status={dict(rows)}  stated==modal agreement={rate}")


def main() -> int:
    p = argparse.ArgumentParser(description="Global per-user photo pull -> modal home country.")
    p.add_argument("--db", type=Path, default=None,
                   help="Override DB path (default: active destination's db_path).")
    p.add_argument("--max-iters", type=int, default=None, help="Cap users this run (debug).")
    p.add_argument("--reset-errors", action="store_true",
                   help="Re-queue status='error' users.")
    args = p.parse_args()

    cfg = load_config()
    db_path = args.db or cfg["db_path"]
    if not db_path.exists():
        sys.exit(f"ERROR: {db_path} not found. Run the base extract scripts first.")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    api_key = os.environ.get("FLICKR_API_KEY")
    if not api_key:
        sys.exit("ERROR: FLICKR_API_KEY not set (expected in .env)")

    try:
        import reverse_geocoder as rg
    except ImportError:
        sys.exit("ERROR: reverse_geocoder not installed. Run: uv add reverse_geocoder")

    try:
        con = duckdb.connect(str(db_path))
    except duckdb.IOException as e:
        sys.exit(f"ERROR opening {db_path}: {e}\nHint: another extractor may still hold the DB "
                 "(DuckDB allows one writer).")
    init_tables(con)

    if args.reset_errors:
        n = con.execute("DELETE FROM user_global_status WHERE status='error' RETURNING user_id").fetchall()
        print(f"-> cleared {len(n)} errored users (will re-pull)")

    pending = get_pending(con, cfg["dest_iso3"], cfg["min_photos"])
    if not pending:
        print("-> no pending users (modal pull already covers the cohort superset)")
        status_snapshot(con, "done")
        return 0

    print(f"-> {len(pending):,} users pending (cohort superset; "
          f"resolved non-{cfg['dest_iso3']} stated + >= {cfg['min_photos']} dest photos)")
    print("-> building offline reverse-geocoder (one-time k-d tree) ...")
    geo = rg.RGeocoder(mode=1, verbose=False)
    iso_map = iso2_to_iso3_map()
    status_snapshot(con, "start")

    sleep = cfg["rate_limit_sleep"]
    processed = n_ok = n_nogeo = n_nf = n_err = 0
    try:
        for uid in pending:
            if args.max_iters is not None and processed >= args.max_iters:
                print(f"-> reached --max-iters={args.max_iters}, stopping")
                break
            status, coords, capped = fetch_user_coords(api_key, uid, cfg["per_page"], sleep)
            if status == "ok":
                counts = tally_countries(geo, iso_map, coords)
                upsert_user(con, uid, "ok" if counts else "no_geo", counts, capped, len(coords))
                if counts:
                    n_ok += 1
                else:
                    n_nogeo += 1
            elif status == "no_geo":
                upsert_user(con, uid, "no_geo", {}, capped, 0)
                n_nogeo += 1
            elif status == "not_found":
                upsert_user(con, uid, "not_found", {}, False, 0, err=str(coords))
                n_nf += 1
            else:
                upsert_user(con, uid, "error", {}, False, 0, err=str(coords))
                n_err += 1
            processed += 1
            if processed % PROGRESS_EVERY == 0:
                print(f"  +{processed:>5,}/{len(pending):,}  ok={n_ok}  no_geo={n_nogeo}  "
                      f"not_found={n_nf}  error={n_err}")
            time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n-> interrupted; partial state persisted, re-run to resume")

    status_snapshot(con, "end")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
