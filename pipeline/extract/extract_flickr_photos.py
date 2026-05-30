"""
Quadtree extraction of geotagged Flickr photos over Georgia, 2012-2019.

Per SPEC sections 4.1 / 5.1: flickr.photos.search with bbox + date filter.
Subdivide spatially when a single query would return >= SUBDIVIDE_THRESHOLD
(under Flickr's 4000-per-query cap with buffer). Paginate at PER_PAGE.
Rate-limit ~1 req/sec. Persist a frontier table in DuckDB so the run resumes
cleanly after Ctrl+C, network blips, or rate-limit hits.

State lives in data/flickr.duckdb:
  frontier (id, parent_id, bbox, time range, status, ...)
  photos   (photo_id, user_id, owner_name, lat, lon, taken_ts, ...)

Resume semantics: re-run the script. 'pending' frontier entries get processed.
Pass --reset-frontier to bump 'in_progress' or 'error' rows back to 'pending'.

Usage:
    # Verify it runs (15 iters, ~30s)
    uv run python pipeline/extract/extract_flickr_photos.py --max-iters 15

    # Production: drain frontier (background, multi-day)
    uv run python pipeline/extract/extract_flickr_photos.py
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

# Georgia: WGS84 bbox (min_lon, min_lat, max_lon, max_lat)
GEORGIA_BBOX = (40.00, 41.05, 46.75, 43.60)

FLICKR_API = "https://api.flickr.com/services/rest/"
PER_PAGE = 500
SUBDIVIDE_THRESHOLD = 3500   # under Flickr's 4000-per-query cap
MIN_BBOX_SIDE_DEG = 0.005    # ~500 m; smaller -> split by time instead
RATE_LIMIT_SLEEP = 1.0       # seconds between API calls
RETRY_BACKOFF = [1, 2, 4, 8, 16]
HTTP_TIMEOUT = 30


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE SEQUENCE IF NOT EXISTS frontier_id_seq START 1;
        CREATE TABLE IF NOT EXISTS frontier (
            id              BIGINT PRIMARY KEY,
            parent_id       BIGINT,
            min_lon         DOUBLE, min_lat DOUBLE,
            max_lon         DOUBLE, max_lat DOUBLE,
            min_ts          BIGINT, max_ts  BIGINT,
            status          VARCHAR,           -- pending|in_progress|done|subdivided|error|empty
            total_returned  INTEGER,
            pages_done      INTEGER,
            last_error      VARCHAR,
            updated_at      TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS photos (
            photo_id     BIGINT PRIMARY KEY,
            user_id      VARCHAR,
            owner_name   VARCHAR,
            lat          DOUBLE,
            lon          DOUBLE,
            taken_ts     TIMESTAMP,
            granularity  INTEGER,
            title        VARCHAR,
            tags         VARCHAR,
            frontier_id  BIGINT,
            fetched_at   TIMESTAMP
        );
    """)


def seed_frontier(
    con: duckdb.DuckDBPyConnection,
    bbox: tuple[float, float, float, float],
    years: range,
) -> None:
    """One frontier entry per year covering the full destination bbox."""
    n = con.execute("SELECT COUNT(*) FROM frontier").fetchone()[0]
    if n > 0:
        return
    for y in years:
        min_ts = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp())
        max_ts = int(datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp())
        con.execute(
            """
            INSERT INTO frontier
                (id, parent_id, min_lon, min_lat, max_lon, max_lat,
                 min_ts, max_ts, status, updated_at)
            VALUES
                (nextval('frontier_id_seq'), NULL, ?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP)
            """,
            [bbox[0], bbox[1], bbox[2], bbox[3], min_ts, max_ts],
        )


def flickr_search(api_key: str, bbox, min_ts: int, max_ts: int, page: int) -> dict:
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "min_taken_date": min_ts,
        "max_taken_date": max_ts,
        "extras": "geo,date_taken,owner_name,tags",
        "per_page": PER_PAGE,
        "page": page,
        "format": "json",
        "nojsoncallback": 1,
        "content_type": 1,   # photos only (not screenshots/illustrations)
        "media": "photos",
        "has_geo": 1,
    }
    last_err: Exception | None = None
    for backoff in RETRY_BACKOFF:
        try:
            r = requests.get(FLICKR_API, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if data.get("stat") != "ok":
                raise RuntimeError(f"flickr stat={data.get('stat')} msg={data.get('message')}")
            return data
        except (requests.RequestException, RuntimeError, ValueError) as e:
            last_err = e
            print(f"    ! retry in {backoff}s: {e}")
            time.sleep(backoff)
    raise RuntimeError(f"flickr_search failed after retries: {last_err}")


def subdivide(con: duckdb.DuckDBPyConnection, f) -> int:
    """Spatial quadtree split, or time bisect if bbox is already tiny.
    Returns number of children inserted."""
    w = f["max_lon"] - f["min_lon"]
    h = f["max_lat"] - f["min_lat"]
    if w > MIN_BBOX_SIDE_DEG or h > MIN_BBOX_SIDE_DEG:
        mlon = (f["min_lon"] + f["max_lon"]) / 2
        mlat = (f["min_lat"] + f["max_lat"]) / 2
        children = [
            (f["min_lon"], f["min_lat"], mlon,         mlat,         f["min_ts"], f["max_ts"]),
            (mlon,         f["min_lat"], f["max_lon"], mlat,         f["min_ts"], f["max_ts"]),
            (f["min_lon"], mlat,         mlon,         f["max_lat"], f["min_ts"], f["max_ts"]),
            (mlon,         mlat,         f["max_lon"], f["max_lat"], f["min_ts"], f["max_ts"]),
        ]
    else:
        mts = (f["min_ts"] + f["max_ts"]) // 2
        children = [
            (f["min_lon"], f["min_lat"], f["max_lon"], f["max_lat"], f["min_ts"], mts),
            (f["min_lon"], f["min_lat"], f["max_lon"], f["max_lat"], mts + 1,     f["max_ts"]),
        ]
    for c in children:
        con.execute(
            """
            INSERT INTO frontier
                (id, parent_id, min_lon, min_lat, max_lon, max_lat,
                 min_ts, max_ts, status, updated_at)
            VALUES
                (nextval('frontier_id_seq'), ?, ?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP)
            """,
            [f["id"], *c],
        )
    con.execute(
        "UPDATE frontier SET status='subdivided', updated_at=CURRENT_TIMESTAMP WHERE id=?",
        [f["id"]],
    )
    return len(children)


def parse_taken(s: str | None) -> datetime | None:
    if not s or s.startswith("0000"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def insert_photos(con: duckdb.DuckDBPyConnection, photos_payload: dict, frontier_id: int) -> int:
    rows: list[tuple] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for p in photos_payload.get("photo", []):
        try:
            pid = int(p["id"])
        except (KeyError, ValueError):
            continue
        try:
            lat = float(p.get("latitude") or 0)
            lon = float(p.get("longitude") or 0)
        except ValueError:
            continue
        if lat == 0.0 and lon == 0.0:
            continue
        owner = p.get("owner")
        if not owner:
            continue
        try:
            gran = int(p.get("datetakengranularity") or 0)
        except ValueError:
            gran = 0
        rows.append((
            pid,
            owner,
            p.get("ownername"),
            lat,
            lon,
            parse_taken(p.get("datetaken")),
            gran,
            p.get("title"),
            p.get("tags"),
            frontier_id,
            now,
        ))
    if not rows:
        return 0
    con.executemany(
        """
        INSERT INTO photos
            (photo_id, user_id, owner_name, lat, lon, taken_ts,
             granularity, title, tags, frontier_id, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (photo_id) DO NOTHING
        """,
        rows,
    )
    return len(rows)


def fetch_one_frontier(con: duckdb.DuckDBPyConnection) -> dict | None:
    row = con.execute("""
        SELECT id, min_lon, min_lat, max_lon, max_lat, min_ts, max_ts
        FROM frontier WHERE status='pending'
        ORDER BY id LIMIT 1
    """).fetchone()
    if row is None:
        return None
    cols = ["id", "min_lon", "min_lat", "max_lon", "max_lat", "min_ts", "max_ts"]
    return dict(zip(cols, row))


def process(con: duckdb.DuckDBPyConnection, api_key: str, max_iters: int | None) -> None:
    iters = 0
    while True:
        if max_iters is not None and iters >= max_iters:
            print(f"-> reached --max-iters={max_iters}, stopping")
            return
        f = fetch_one_frontier(con)
        if f is None:
            print("-> frontier drained")
            return

        con.execute(
            "UPDATE frontier SET status='in_progress', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            [f["id"]],
        )

        days = (f["max_ts"] - f["min_ts"]) / 86400.0
        label = (
            f"F#{f['id']:>4} "
            f"bbox=({f['min_lon']:.3f},{f['min_lat']:.3f})-({f['max_lon']:.3f},{f['max_lat']:.3f}) "
            f"days={days:.0f}"
        )

        try:
            data = flickr_search(
                api_key,
                (f["min_lon"], f["min_lat"], f["max_lon"], f["max_lat"]),
                f["min_ts"], f["max_ts"], page=1,
            )
        except Exception as e:
            print(f"  {label}  ERROR: {e}")
            con.execute(
                "UPDATE frontier SET status='error', last_error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                [str(e)[:500], f["id"]],
            )
            iters += 1
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        total = int(data["photos"].get("total", 0))
        pages = int(data["photos"].get("pages", 0))

        if total == 0:
            con.execute(
                "UPDATE frontier SET status='empty', total_returned=0, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                [f["id"]],
            )
            print(f"  {label}  total=0 (empty)")
            iters += 1
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        if total >= SUBDIVIDE_THRESHOLD:
            n_children = subdivide(con, f)
            print(f"  {label}  total={total} -> subdivide ({n_children} children)")
            iters += 1
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        # Paginate this leaf tile
        inserted = insert_photos(con, data["photos"], f["id"])
        completed = True
        for page in range(2, pages + 1):
            time.sleep(RATE_LIMIT_SLEEP + random.uniform(0, 0.2))
            try:
                page_data = flickr_search(
                    api_key,
                    (f["min_lon"], f["min_lat"], f["max_lon"], f["max_lat"]),
                    f["min_ts"], f["max_ts"], page=page,
                )
            except Exception as e:
                print(f"  {label}  page {page} ERROR: {e}")
                con.execute(
                    "UPDATE frontier SET status='error', last_error=?, pages_done=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    [str(e)[:500], page - 1, f["id"]],
                )
                completed = False
                break
            inserted += insert_photos(con, page_data["photos"], f["id"])

        if completed:
            con.execute(
                """
                UPDATE frontier SET status='done', total_returned=?, pages_done=?,
                                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                [total, pages, f["id"]],
            )
            print(f"  {label}  total={total} pages={pages} inserted={inserted}")

        iters += 1
        time.sleep(RATE_LIMIT_SLEEP)


def status_snapshot(con: duckdb.DuckDBPyConnection, label: str) -> None:
    rows = con.execute("SELECT status, COUNT(*) FROM frontier GROUP BY status ORDER BY status").fetchall()
    n_photos = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    n_users  = con.execute("SELECT COUNT(DISTINCT user_id) FROM photos").fetchone()[0]
    print(f"[{label}] frontier={dict(rows)}  photos={n_photos:,}  unique_users={n_users:,}")


def main() -> int:
    p = argparse.ArgumentParser(description="Quadtree Flickr extractor for Georgia 2012-2019.")
    p.add_argument("--db", type=Path, default=Path("data/flickr.duckdb"))
    p.add_argument("--max-iters", type=int, default=None,
                   help="Cap number of frontier tiles processed in this run (debug).")
    p.add_argument("--reset-frontier", action="store_true",
                   help="Reset 'in_progress' or 'error' rows to 'pending' before running.")
    args = p.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    api_key = os.environ.get("FLICKR_API_KEY")
    if not api_key:
        sys.exit("ERROR: FLICKR_API_KEY not set (expected in .env)")

    con = duckdb.connect(str(args.db))
    init_db(con)

    if args.reset_frontier:
        n = con.execute(
            "UPDATE frontier SET status='pending', last_error=NULL "
            "WHERE status IN ('in_progress','error') RETURNING id"
        ).fetchall()
        print(f"-> reset {len(n)} frontier rows to 'pending'")

    seed_frontier(con, GEORGIA_BBOX, range(2012, 2020))

    status_snapshot(con, "start")
    try:
        process(con, api_key, max_iters=args.max_iters)
    except KeyboardInterrupt:
        print("\n-> interrupted; partial state is persisted, re-run to resume")
    status_snapshot(con, "end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
