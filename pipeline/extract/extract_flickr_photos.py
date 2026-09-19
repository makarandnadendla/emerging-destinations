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
import yaml

from _common import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

FLICKR_API = "https://api.flickr.com/services/rest/"
# Flickr-mechanics defaults; overridden from config.yaml `flickr:` block in main().
PER_PAGE = 500
SUBDIVIDE_THRESHOLD = 3500   # under Flickr's 4000-per-query cap
MIN_BBOX_SIDE_DEG = 0.005    # ~500 m; at/below this, split by time instead
RATE_LIMIT_SLEEP = 1.0       # seconds between API calls
RETRY_BACKOFF = [1, 2, 4, 8, 16]
HTTP_TIMEOUT = 30
# Termination guards for the quadtree. A hyper-dense single point (one venue or a
# bulk uploader with thousands of near-identical lat/lon+timestamp photos) can sit
# above SUBDIVIDE_THRESHOLD no matter how finely we split, because Flickr stops
# honoring sub-tile filters and pins `total` near its ~4000 retrieval wall. Without
# a guard the bbox shrinks below MIN_BBOX_SIDE_DEG and time-bisection collapses to a
# zero/negative span ("days=-0"), looping forever. So: stop subdividing once the
# bbox is minimal AND the time span is too small to split, and instead CAP the tile
# (grab Flickr's retrievable max and move on).
MIN_TIME_SPAN_SEC = 7 * 86400   # don't time-bisect spans <= 7 days
MAX_PAGES = 8                   # Flickr serves ~4000 results max (8 * 500); never page past


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return the active destination's params merged with shared/flickr params."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    active = cfg["destination"]
    dest = cfg["destinations"][active]
    flickr = cfg.get("flickr", {})
    y0, y1 = cfg["years"]
    return {
        "active": active,
        "name": dest["name"],
        "bbox": tuple(dest["bbox"]),
        "db_path": dest["db_path"],
        "years": range(y0, y1 + 1),
        "per_page": flickr.get("per_page", PER_PAGE),
        "subdivide_threshold": flickr.get("subdivide_threshold", SUBDIVIDE_THRESHOLD),
        "min_bbox_side_deg": flickr.get("min_bbox_side_deg", MIN_BBOX_SIDE_DEG),
        "rate_limit_sleep": flickr.get("rate_limit_sleep", RATE_LIMIT_SLEEP),
    }


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
    for attempt, backoff in enumerate(RETRY_BACKOFF):
        try:
            r = requests.get(FLICKR_API, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if data.get("stat") != "ok":
                raise RuntimeError(f"flickr stat={data.get('stat')} msg={data.get('message')}")
            return data
        except (requests.RequestException, RuntimeError, ValueError) as e:
            last_err = e
            if attempt < len(RETRY_BACKOFF) - 1:   # no point sleeping after the last try
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


def can_subdivide(f: dict) -> bool:
    """True if this tile can be meaningfully split further: either the bbox is
    still larger than the minimum side (spatial split), or the time span is large
    enough to bisect without collapsing (time split). When both are exhausted the
    tile is terminal and must be capped, not subdivided."""
    w = f["max_lon"] - f["min_lon"]
    h = f["max_lat"] - f["min_lat"]
    if w > MIN_BBOX_SIDE_DEG or h > MIN_BBOX_SIDE_DEG:
        return True
    return (f["max_ts"] - f["min_ts"]) > MIN_TIME_SPAN_SEC


def drain_pages(con, api_key, f, first_data, pages, label, max_pages):
    """Insert page 1 (already fetched) then pages 2..min(pages, max_pages).
    Returns (inserted, completed, pages_fetched)."""
    inserted = insert_photos(con, first_data["photos"], f["id"])
    last_page = min(pages, max_pages)
    for page in range(2, last_page + 1):
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
            return inserted, False, page - 1
        inserted += insert_photos(con, page_data["photos"], f["id"])
    return inserted, True, last_page


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

        if total >= SUBDIVIDE_THRESHOLD and can_subdivide(f):
            n_children = subdivide(con, f)
            print(f"  {label}  total={total} -> subdivide ({n_children} children)")
            iters += 1
            time.sleep(RATE_LIMIT_SLEEP)
            continue

        # Either a normal leaf (total < threshold) or a hyper-dense tile that can't
        # be split further (bbox minimal AND time span too small). Both paginate up
        # to Flickr's retrievable wall; the dense case is marked 'capped' so it is
        # never revisited and the data loss is logged rather than silent.
        # pages > MAX_PAGES also caps: with per_page below ~440 (config or Flickr
        # serving short pages) a tile can be under the subdivide threshold yet
        # have more pages than we drain — that loss must be flagged, not 'done'.
        capped = total >= SUBDIVIDE_THRESHOLD or pages > MAX_PAGES
        inserted, completed, fetched = drain_pages(con, api_key, f, data, pages, label, MAX_PAGES)
        if completed:
            status = "capped" if capped else "done"
            con.execute(
                """
                UPDATE frontier SET status=?, total_returned=?, pages_done=?,
                                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                [status, total, fetched, f["id"]],
            )
            if capped:
                print(f"  {label}  total={total} CAPPED at Flickr wall — kept ~{inserted} (pages={fetched})")
            else:
                print(f"  {label}  total={total} pages={fetched} inserted={inserted}")

        iters += 1
        time.sleep(RATE_LIMIT_SLEEP)


def status_snapshot(con: duckdb.DuckDBPyConnection, label: str) -> None:
    rows = con.execute("SELECT status, COUNT(*) FROM frontier GROUP BY status ORDER BY status").fetchall()
    n_photos = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    n_users  = con.execute("SELECT COUNT(DISTINCT user_id) FROM photos").fetchone()[0]
    print(f"[{label}] frontier={dict(rows)}  photos={n_photos:,}  unique_users={n_users:,}")


def main() -> int:
    global PER_PAGE, SUBDIVIDE_THRESHOLD, MIN_BBOX_SIDE_DEG, RATE_LIMIT_SLEEP

    p = argparse.ArgumentParser(description="Quadtree Flickr extractor (destination from config.yaml).")
    p.add_argument("--db", type=Path, default=None,
                   help="Override DB path (default: active destination's db_path from config.yaml).")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Cap number of frontier tiles processed in this run (debug).")
    p.add_argument("--reset-frontier", action="store_true",
                   help="Reset 'in_progress' or 'error' rows to 'pending' before running.")
    args = p.parse_args()

    cfg = load_config()
    PER_PAGE = cfg["per_page"]
    SUBDIVIDE_THRESHOLD = cfg["subdivide_threshold"]
    MIN_BBOX_SIDE_DEG = cfg["min_bbox_side_deg"]
    RATE_LIMIT_SLEEP = cfg["rate_limit_sleep"]

    db_path = args.db if args.db is not None else Path(cfg["db_path"])
    bbox = cfg["bbox"]
    years = cfg["years"]
    print(f"-> destination={cfg['name']}  bbox={bbox}  years={years.start}-{years.stop - 1}  db={db_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    api_key = os.environ.get("FLICKR_API_KEY")
    if not api_key:
        sys.exit("ERROR: FLICKR_API_KEY not set (expected in .env)")

    con = duckdb.connect(str(db_path))
    init_db(con)

    if args.reset_frontier:
        n = con.execute(
            "UPDATE frontier SET status='pending', last_error=NULL "
            "WHERE status IN ('in_progress','error') RETURNING id"
        ).fetchall()
        print(f"-> reset {len(n)} frontier rows to 'pending'")

    seed_frontier(con, bbox, years)

    status_snapshot(con, "start")
    try:
        process(con, api_key, max_iters=args.max_iters)
    except KeyboardInterrupt:
        print("\n-> interrupted; partial state is persisted, re-run to resume")
    status_snapshot(con, "end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
