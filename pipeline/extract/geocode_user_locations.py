"""
Resolve Flickr profile `location_raw` strings to ISO3 country codes via Nominatim.

Per SPEC section 5.5:
  - Public Nominatim instance, strict 1 req/sec
  - User-Agent must identify the project
  - Cache aggressively: same string never queries twice
  - Keep only results whose response contains a country_code

Output: user_geocodes table in data/flickr.duckdb keyed on the raw query string.
Downstream Stage T joins on country_iso3 to indicators_panel.

**Run after extract_flickr_users.py** (depends on users.location_raw).

Usage:
    uv run python pipeline/extract/geocode_user_locations.py
    uv run python pipeline/extract/geocode_user_locations.py --max-iters 50
    uv run python pipeline/extract/geocode_user_locations.py --reset-errors
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pycountry
import requests
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def active_db_path() -> Path:
    """DB path of the active destination from config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return Path(cfg["destinations"][cfg["destination"]]["db_path"])


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
RATE_LIMIT_SLEEP = 1.1          # >= 1.0 per Nominatim ToS; 1.1 for safety
RETRY_BACKOFF = [2, 4, 8, 16]   # respectful backoff; Nominatim rate-limits at 429
HTTP_TIMEOUT = 30
PROGRESS_EVERY = 25

# Required by Nominatim ToS. Email is the project owner's contact.
USER_AGENT = "emerging-destinations/0.0.1 (mailto:nn.chetta@gmail.com)"


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def init_geocodes_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS user_geocodes (
            location_raw  VARCHAR PRIMARY KEY,
            country_iso2  VARCHAR,
            country_iso3  VARCHAR,
            country_name  VARCHAR,
            display_name  VARCHAR,
            lat           DOUBLE,
            lon           DOUBLE,
            status        VARCHAR,   -- ok | no_result | no_country | error
            last_error    VARCHAR,
            fetched_at    TIMESTAMP
        );
    """)


def get_pending(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute("""
        SELECT DISTINCT u.location_raw
        FROM users u
        LEFT JOIN user_geocodes g ON g.location_raw = u.location_raw
        WHERE u.location_raw IS NOT NULL
          AND TRIM(u.location_raw) <> ''
          AND g.location_raw IS NULL
        ORDER BY u.location_raw
    """).fetchall()
    return [r[0] for r in rows]


def alpha2_to_iso3(alpha2: str) -> tuple[str, str] | None:
    """Return (alpha_3, country_name) or None for unknown codes."""
    if not alpha2:
        return None
    try:
        c = pycountry.countries.get(alpha_2=alpha2.upper())
    except (LookupError, KeyError):
        return None
    if c is None:
        return None
    return c.alpha_3, c.name


def nominatim_search(query: str) -> tuple[str, dict | str]:
    """Returns (status, payload). status in {'ok', 'no_result', 'error'}."""
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    last_err: str | None = None
    for backoff in [0, *RETRY_BACKOFF]:
        if backoff:
            time.sleep(backoff)
        try:
            r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                last_err = "429 rate-limited"
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                return "no_result", "empty result list"
            return "ok", data[0]
        except (requests.RequestException, ValueError) as e:
            last_err = str(e)
    return "error", (last_err or "unknown")[:500]


def upsert_geocode(con, raw: str, status: str, payload) -> str:
    """Insert/replace a geocode row. Returns the effective status after iso3 lookup."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if status != "ok":
        con.execute("""
            INSERT INTO user_geocodes (location_raw, status, last_error, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (location_raw) DO UPDATE SET
                status=excluded.status, last_error=excluded.last_error,
                fetched_at=excluded.fetched_at
        """, [raw, status, str(payload)[:500] if payload else None, now])
        return status

    hit = payload or {}
    address = hit.get("address") or {}
    iso2 = (address.get("country_code") or "").strip().lower() or None
    iso_pair = alpha2_to_iso3(iso2) if iso2 else None
    iso3 = iso_pair[0] if iso_pair else None
    cname = iso_pair[1] if iso_pair else address.get("country")
    try:
        lat = float(hit.get("lat")) if hit.get("lat") is not None else None
        lon = float(hit.get("lon")) if hit.get("lon") is not None else None
    except (TypeError, ValueError):
        lat = lon = None

    eff_status = "ok" if iso3 else "no_country"
    con.execute("""
        INSERT INTO user_geocodes
            (location_raw, country_iso2, country_iso3, country_name,
             display_name, lat, lon, status, last_error, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
        ON CONFLICT (location_raw) DO UPDATE SET
            country_iso2=excluded.country_iso2, country_iso3=excluded.country_iso3,
            country_name=excluded.country_name, display_name=excluded.display_name,
            lat=excluded.lat, lon=excluded.lon,
            status=excluded.status, last_error=NULL, fetched_at=excluded.fetched_at
    """, [raw, iso2, iso3, cname, hit.get("display_name"), lat, lon, eff_status, now])
    return eff_status


def status_snapshot(con: duckdb.DuckDBPyConnection, label: str) -> None:
    n_users_with_loc = con.execute(
        "SELECT COUNT(DISTINCT location_raw) FROM users WHERE location_raw IS NOT NULL"
    ).fetchone()[0]
    rows = con.execute(
        "SELECT status, COUNT(*) FROM user_geocodes GROUP BY status ORDER BY status"
    ).fetchall()
    n_with_iso3 = con.execute(
        "SELECT COUNT(*) FROM user_geocodes WHERE country_iso3 IS NOT NULL"
    ).fetchone()[0]
    print(
        f"[{label}] distinct_location_strings={n_users_with_loc:,}  "
        f"by_status={dict(rows)}  with_iso3={n_with_iso3:,}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Geocode Flickr location_raw strings via Nominatim.")
    p.add_argument("--db", type=Path, default=None,
                   help="Override DB path (default: active destination's db_path from config.yaml).")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Cap queries this run (debug).")
    p.add_argument("--reset-errors", action="store_true",
                   help="Drop status='error' rows so they re-enter pending pool.")
    args = p.parse_args()
    if args.db is None:
        args.db = active_db_path()

    if not args.db.exists():
        sys.exit(f"ERROR: {args.db} not found. Run extract_flickr_photos.py + extract_flickr_users.py first.")
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    try:
        con = duckdb.connect(str(args.db))
    except duckdb.IOException as e:
        sys.exit(
            f"ERROR opening {args.db}: {e}\n"
            "Hint: another extractor may still be writing to this DB."
        )
    init_geocodes_table(con)

    if args.reset_errors:
        n = con.execute(
            "DELETE FROM user_geocodes WHERE status='error' RETURNING location_raw"
        ).fetchall()
        print(f"-> cleared {len(n)} errored geocodes")

    pending = get_pending(con)
    if not pending:
        print("-> no pending location strings")
        status_snapshot(con, "done")
        return 0

    print(f"-> {len(pending):,} location strings pending  (est. {len(pending) * RATE_LIMIT_SLEEP / 60:.1f} min)")
    status_snapshot(con, "start")

    processed = 0
    n_ok = n_nr = n_nc = n_err = 0
    try:
        for raw in pending:
            if args.max_iters is not None and processed >= args.max_iters:
                print(f"-> reached --max-iters={args.max_iters}, stopping")
                break
            status, payload = nominatim_search(raw)
            eff = upsert_geocode(con, raw, status, payload)
            if eff == "ok":
                n_ok += 1
            elif eff == "no_result":
                n_nr += 1
            elif eff == "no_country":
                n_nc += 1
            else:
                n_err += 1
            processed += 1
            if processed % PROGRESS_EVERY == 0:
                print(f"  +{processed:>5,}  ok={n_ok}  no_result={n_nr}  no_country={n_nc}  error={n_err}")
            time.sleep(RATE_LIMIT_SLEEP)
    except KeyboardInterrupt:
        print("\n-> interrupted; partial state persisted, re-run to resume")

    status_snapshot(con, "end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
