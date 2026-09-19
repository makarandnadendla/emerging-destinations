"""
Fetch flickr.people.getInfo for each distinct user_id in the photos table.

Per SPEC sections 4.1 / 5.1: profile metadata for tourist-filter, user-home
resolution (the free-text `location` field downstream goes to Nominatim).

State lives in the same data/flickr.duckdb. The `users` table is keyed on
user_id; re-running consumes only NOT-yet-fetched IDs. Soft failures
(deleted users, banned accounts) are recorded as status='not_found' so they
aren't retried forever. Hard errors record status='error'; clear with
--reset-errors to retry them.

**Run AFTER extract_flickr_photos.py finishes.** DuckDB does not support
concurrent writers to one file, and this script wants the same DB the
photo extractor is still writing to.

Usage:
    uv run python pipeline/extract/extract_flickr_users.py
    uv run python pipeline/extract/extract_flickr_users.py --max-iters 100
    uv run python pipeline/extract/extract_flickr_users.py --reset-errors
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests

from _common import FatalApiError, active_db_path, load_dotenv

FLICKR_API = "https://api.flickr.com/services/rest/"
RATE_LIMIT_SLEEP = 1.0
RETRY_BACKOFF = [1, 2, 4, 8, 16]
HTTP_TIMEOUT = 30
PROGRESS_EVERY = 25


def init_users_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id         VARCHAR PRIMARY KEY,
            username        VARCHAR,
            realname        VARCHAR,
            location_raw    VARCHAR,
            timezone_label  VARCHAR,
            timezone_offset VARCHAR,
            photos_total    INTEGER,
            first_photo_ts  TIMESTAMP,
            status          VARCHAR,   -- ok | not_found | error
            last_error      VARCHAR,
            fetched_at      TIMESTAMP
        );
    """)


def get_pending(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute("""
        SELECT DISTINCT p.user_id
        FROM photos p
        LEFT JOIN users u ON u.user_id = p.user_id
        WHERE u.user_id IS NULL AND p.user_id IS NOT NULL
        ORDER BY p.user_id
    """).fetchall()
    return [r[0] for r in rows]


def safe(d, *keys):
    """Walk nested dict, unwrap {'_content': ...} at the leaf."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    if isinstance(cur, dict) and "_content" in cur:
        cur = cur["_content"]
    if isinstance(cur, str):
        cur = cur.strip()
        return cur or None
    return cur


def parse_first_photo_ts(p: dict) -> datetime | None:
    s = safe(p, "photos", "firstdatetaken")
    if not s or s.startswith("0000"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def flickr_get_info(api_key: str, user_id: str) -> tuple[str, dict | str | None]:
    """Returns (status, payload) where status is 'ok' | 'not_found' | 'error'."""
    params = {
        "method": "flickr.people.getInfo",
        "api_key": api_key,
        "user_id": user_id,
        "format": "json",
        "nojsoncallback": 1,
    }
    last_err: Exception | str | None = None
    for attempt, backoff in enumerate(RETRY_BACKOFF):
        try:
            r = requests.get(FLICKR_API, params=params, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if data.get("stat") == "ok":
                return "ok", data.get("person", {})
            if data.get("stat") == "fail":
                code = data.get("code")
                # 1 = user not found, 5 = user deleted, 100 = invalid api key (hard fail)
                if code in (1, 5):
                    return "not_found", data.get("message", "")
                if code == 100:
                    # Must escape the retry loop (FatalApiError is not caught below).
                    raise FatalApiError(f"invalid API key: {data.get('message')}")
                # other failure codes: fall through to the shared backoff sleep
                last_err = f"stat=fail code={code} msg={data.get('message')}"
            else:
                last_err = f"unexpected stat: {data.get('stat')}"
        except requests.HTTPError as e:
            # 4xx are usually permanent for that user
            sc = e.response.status_code if e.response is not None else None
            if sc and 400 <= sc < 500 and sc != 429:
                return "error", f"HTTP {sc}"
            last_err = str(e)
        except (requests.RequestException, RuntimeError, ValueError) as e:
            last_err = str(e)
        if attempt < len(RETRY_BACKOFF) - 1:   # no point sleeping after the last try
            time.sleep(backoff)
    return "error", (last_err or "unknown error")[:500] if isinstance(last_err, str) else str(last_err)[:500]


def upsert_user(con, user_id: str, status: str, payload) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if status == "ok":
        p = payload or {}
        try:
            photos_total = int(safe(p, "photos", "count") or 0) or None
        except (TypeError, ValueError):
            photos_total = None
        con.execute("""
            INSERT INTO users
                (user_id, username, realname, location_raw, timezone_label, timezone_offset,
                 photos_total, first_photo_ts, status, last_error, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ok', NULL, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                username=excluded.username, realname=excluded.realname,
                location_raw=excluded.location_raw,
                timezone_label=excluded.timezone_label, timezone_offset=excluded.timezone_offset,
                photos_total=excluded.photos_total, first_photo_ts=excluded.first_photo_ts,
                status='ok', last_error=NULL, fetched_at=excluded.fetched_at
        """, [
            user_id,
            safe(p, "username"),
            safe(p, "realname"),
            safe(p, "location"),
            safe(p, "timezone", "label"),
            safe(p, "timezone", "offset"),
            photos_total,
            parse_first_photo_ts(p),
            now,
        ])
    elif status == "not_found":
        con.execute("""
            INSERT INTO users (user_id, status, last_error, fetched_at)
            VALUES (?, 'not_found', ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                status='not_found', last_error=excluded.last_error, fetched_at=excluded.fetched_at
        """, [user_id, str(payload)[:500] if payload else None, now])
    else:  # error
        con.execute("""
            INSERT INTO users (user_id, status, last_error, fetched_at)
            VALUES (?, 'error', ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                status='error', last_error=excluded.last_error, fetched_at=excluded.fetched_at
        """, [user_id, str(payload)[:500], now])


def status_snapshot(con: duckdb.DuckDBPyConnection, label: str) -> None:
    n_photos = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    n_distinct_users = con.execute("SELECT COUNT(DISTINCT user_id) FROM photos").fetchone()[0]
    rows = con.execute("SELECT status, COUNT(*) FROM users GROUP BY status ORDER BY status").fetchall()
    n_with_loc = con.execute("SELECT COUNT(*) FROM users WHERE location_raw IS NOT NULL").fetchone()[0]
    print(
        f"[{label}] photos={n_photos:,}  distinct_users={n_distinct_users:,}  "
        f"users_by_status={dict(rows)}  with_location_raw={n_with_loc:,}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch Flickr people.getInfo for distinct photo users.")
    p.add_argument("--db", type=Path, default=None,
                   help="Override DB path (default: active destination's db_path from config.yaml).")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Cap users processed this run (debug).")
    p.add_argument("--reset-errors", action="store_true",
                   help="Drop status='error' rows so they re-enter the pending pool.")
    args = p.parse_args()
    if args.db is None:
        args.db = active_db_path()

    if not args.db.exists():
        sys.exit(f"ERROR: {args.db} not found. Run extract_flickr_photos.py first.")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    api_key = os.environ.get("FLICKR_API_KEY")
    if not api_key:
        sys.exit("ERROR: FLICKR_API_KEY not set (expected in .env)")

    try:
        con = duckdb.connect(str(args.db))
    except duckdb.IOException as e:
        sys.exit(
            f"ERROR opening {args.db}: {e}\n"
            "Hint: extract_flickr_photos.py may still be running. DuckDB allows only one writer at a time."
        )
    init_users_table(con)

    if args.reset_errors:
        n = con.execute("DELETE FROM users WHERE status='error' RETURNING user_id").fetchall()
        print(f"-> cleared {len(n)} errored users (will re-fetch)")

    pending = get_pending(con)
    if not pending:
        print("-> no pending users (table already covers all photo authors)")
        status_snapshot(con, "done")
        return 0

    print(f"-> {len(pending):,} users pending  (est. {len(pending)/60:.1f} min at 1 req/s)")
    status_snapshot(con, "start")

    processed = 0
    n_ok = n_nf = n_err = 0
    try:
        for uid in pending:
            if args.max_iters is not None and processed >= args.max_iters:
                print(f"-> reached --max-iters={args.max_iters}, stopping")
                break
            status, payload = flickr_get_info(api_key, uid)
            upsert_user(con, uid, status, payload)
            if status == "ok":
                n_ok += 1
            elif status == "not_found":
                n_nf += 1
            else:
                n_err += 1
            processed += 1
            if processed % PROGRESS_EVERY == 0:
                print(f"  +{processed:>5,}  ok={n_ok}  not_found={n_nf}  error={n_err}")
            time.sleep(RATE_LIMIT_SLEEP)
    except KeyboardInterrupt:
        print("\n-> interrupted; partial state persisted, re-run to resume")

    status_snapshot(con, "end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
