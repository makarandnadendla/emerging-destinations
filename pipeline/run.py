"""
Pipeline orchestrator.

Stage T (Transform) is implemented: it builds a per-destination DuckDB warehouse
by running the numbered SQL files in pipeline/sql/ in order, against a bootstrap
that loads the h3 extension, attaches the raw extraction DB read-only, and sets
the path/threshold variables the SQL reads via getvariable().

Stage E (Extract) is currently run via the individual scripts in
pipeline/extract/ (see their module docstrings); --extract here is a stub.

Usage:
    # Build the warehouse for the active destination (from config.yaml)
    uv run python pipeline/run.py --transform

    # Override destination (e.g. the Georgia comparison set)
    uv run python pipeline/run.py --transform --dest georgia

    # Run only specific steps (by numeric prefix)
    uv run python pipeline/run.py --transform --dest georgia --only 02 03 04 05
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb
import yaml

PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PIPELINE_DIR / "config.yaml"
SQL_DIR = PIPELINE_DIR / "sql"


def load_config(dest_override: str | None,
                h3_res_override: int | None = None,
                warehouse_override: str | None = None) -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    active = dest_override or cfg["destination"]
    if active not in cfg["destinations"]:
        sys.exit(f"ERROR: unknown destination '{active}'. Known: {list(cfg['destinations'])}")
    dest = cfg["destinations"][active]
    y0, y1 = cfg["years"]
    return {
        "active": active,
        "name": dest["name"],
        "iso3": dest["iso3"],
        "raw_db": dest["db_path"],
        "pois_path": dest["osm_pois_parquet"],
        # Overrides let you build an alternate resolution into a side-by-side
        # warehouse without mutating config.yaml (e.g. --h3-res 7).
        "warehouse": warehouse_override or dest["warehouse_path"],
        "indicators_path": cfg["indicators_parquet"],
        "h3_res": h3_res_override if h3_res_override is not None else cfg["h3_res"],
        "min_photos": cfg["min_photos"],
        "min_cells": cfg["min_cells"],
        "years": (y0, y1),
    }


def step_prefix(path: Path) -> str:
    m = re.match(r"^(\d+)", path.name)
    return m.group(1) if m else ""


def table_name(path: Path) -> str:
    # 04_poi_per_cell.sql -> poi_per_cell
    return re.sub(r"^\d+_", "", path.stem)


def split_statements(sql: str) -> list[str]:
    """Split SQL on ';' but ignore separators inside line comments (-- ... ),
    block comments (/* ... */), and single-quoted string literals. A naive
    sql.split(';') breaks statements whose comments contain a semicolon."""
    stmts: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    in_line_comment = in_block_comment = in_string = False
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                i += 2
                in_block_comment = False
                continue
        elif in_string:
            buf.append(ch)
            if ch == "'":
                if nxt == "'":  # escaped quote
                    buf.append(nxt)
                    i += 2
                    continue
                in_string = False
        elif ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
        elif ch == "/" and nxt == "*":
            in_block_comment = True
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        elif ch == "'":
            in_string = True
            buf.append(ch)
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                stmts.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def bootstrap(con: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    con.execute("INSTALL h3 FROM community; LOAD h3;")
    raw_db = Path(cfg["raw_db"])
    if not raw_db.exists():
        sys.exit(f"ERROR: raw DB not found: {raw_db}. Run the extract scripts first.")
    con.execute(f"ATTACH '{raw_db.as_posix()}' AS raw (READ_ONLY);")
    con.execute("SET VARIABLE pois_path = ?;", [Path(cfg["pois_path"]).as_posix()])
    con.execute("SET VARIABLE indicators_path = ?;", [Path(cfg["indicators_path"]).as_posix()])
    con.execute("SET VARIABLE dest_iso3 = ?;", [cfg["iso3"]])
    con.execute("SET VARIABLE h3_res = ?;", [cfg["h3_res"]])
    con.execute("SET VARIABLE min_photos = ?;", [cfg["min_photos"]])
    con.execute("SET VARIABLE min_cells = ?;", [cfg["min_cells"]])


def run_transform(cfg: dict, only: list[str] | None) -> int:
    sql_files = sorted(SQL_DIR.glob("*.sql"))
    if only:
        sql_files = [f for f in sql_files if step_prefix(f) in only]
        if not sql_files:
            sys.exit(f"ERROR: no SQL files matched --only {only}")

    wh = Path(cfg["warehouse"])
    wh.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> destination={cfg['name']} ({cfg['iso3']})")
    print(f"-> warehouse={wh}")
    print(f"-> raw={cfg['raw_db']}  pois={cfg['pois_path']}")
    print(f"-> running {len(sql_files)} SQL step(s)\n")

    con = duckdb.connect(str(wh))
    bootstrap(con, cfg)

    for f in sql_files:
        sql = f.read_text(encoding="utf-8")
        # Split into statements (comment/string-aware) so multi-table steps like
        # 10_aggregates_for_frontend.sql run fully without breaking on a ';'
        # that happens to sit inside a '-- ...' comment.
        statements = split_statements(sql)
        t0 = time.time()
        try:
            for stmt in statements:
                con.execute(stmt)
        except Exception as e:
            print(f"   [{f.name}] FAILED: {e}")
            return 1
        dt = time.time() - t0
        tbl = table_name(f)
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"   [{f.name}] -> {tbl}: {n:,} rows  ({dt:.1f}s)")
        except Exception:
            print(f"   [{f.name}] done  ({dt:.1f}s)")

    con.close()
    print("\n-> transform complete")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Emerging Destinations pipeline orchestrator.")
    p.add_argument("--transform", action="store_true", help="Build the DuckDB warehouse (Stage T).")
    p.add_argument("--extract", action="store_true", help="(stub) Run Stage E extract scripts.")
    p.add_argument("--dest", default=None, help="Override active destination from config.yaml.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these SQL steps by numeric prefix, e.g. --only 02 03.")
    p.add_argument("--h3-res", type=int, default=None,
                   help="Override H3 resolution (e.g. 7) for a side-by-side build.")
    p.add_argument("--warehouse", default=None,
                   help="Override warehouse output path (pair with --h3-res).")
    args = p.parse_args()

    if not (args.transform or args.extract):
        p.error("specify --transform (and/or --extract)")

    cfg = load_config(args.dest, args.h3_res, args.warehouse)

    if args.extract:
        print("-> --extract is a stub; run pipeline/extract/*.py directly for now.")
    if args.transform:
        return run_transform(cfg, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
