"""
Extract B1 — 宿泊旅行統計調査 (Overnight Accommodation Travel Statistics, JTA).

Prefecture-level annual overnight guest counts (total + foreign) for the study
window, pulled from the official e-Stat API (政府統計の総合窓口). This is a
VALIDATION layer for the POI-based remoteness measure, not an input to it:
prefecture tourist volume is far coarser than the H3 grid, so it can only serve
as a convergent-validity check (does prefecture-rolled-up remoteness track
official low-visitor-density?).

Source:    e-Stat API v3.0, government statistics code 00601020 (宿泊旅行統計調査)
Host:      総務省 / 観光庁 via e-Stat (https://www.e-stat.go.jp/)
License:   政府標準利用規約 v2.0 (CC-BY 4.0 compatible) — redistributable with
           attribution. Cite: "出典：政府統計の総合窓口(e-Stat) 宿泊旅行統計調査(観光庁)".

Requires a free e-Stat API key in .env:
    ESTAT_APP_ID=<your appId>
Register at https://www.e-stat.go.jp/api/ (応用ソフトウェア向けAPI機能, free, instant).

API COVERAGE NOTE: the survey is DB-ified (API-queryable via getStatsData) for
ONE batch only — the 2016 年確定値 tables (statsDataId 0003313520 = 第2表 延べ宿泊者数),
whose `time` axis carries 2014–2016. The full 2012–2019 panel therefore comes from
the annual 年確定値 Excel workbooks instead (--download-excel / --excel), which the
API exposes as file resources via getDataCatalog. The Excel `第2表(年計)` total
matches the API to the yen (QA-verified on 2014–2016), so the two routes agree.

Usage (full panel — recommended):
    uv run python pipeline/extract/extract_tourism_b1.py --download-excel
    uv run python pipeline/extract/extract_tourism_b1.py --excel
Both write data/cache/tourism_b1_prefecture.parquet (47 prefectures x 8 years x
{total_overnight, foreign_overnight}).

Usage:
    # 1. find the right table(s) for prefecture annual overnight stays:
    uv run python pipeline/extract/extract_tourism_b1.py --list

    # 2. inspect a candidate table's dimension structure before pulling:
    uv run python pipeline/extract/extract_tourism_b1.py --inspect <statsDataId>

    # 3. pull tidy prefecture x year data to parquet:
    uv run python pipeline/extract/extract_tourism_b1.py --pull <statsDataId> [<statsDataId> ...]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import polars as pl
import requests
from python_calamine import CalamineWorkbook

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
OUT_PATH = ROOT / "data" / "cache" / "tourism_b1_prefecture.parquet"
B1_XLS_DIR = ROOT / "data" / "cache" / "b1_xls"

STATS_CODE = "00601020"          # 宿泊旅行統計調査
YEARS = range(2012, 2020)        # 2012..2019 inclusive
API_BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"
RATE_SLEEP = 1.0

# 47 prefectures in JIS order -> "NN000" area code (matches the e-Stat API area codes).
PREFS = (
    "北海道 青森県 岩手県 宮城県 秋田県 山形県 福島県 茨城県 栃木県 群馬県 埼玉県 千葉県 東京都 "
    "神奈川県 新潟県 富山県 石川県 福井県 山梨県 長野県 岐阜県 静岡県 愛知県 三重県 滋賀県 京都府 "
    "大阪府 兵庫県 奈良県 和歌山県 鳥取県 島根県 岡山県 広島県 山口県 徳島県 香川県 愛媛県 高知県 "
    "福岡県 佐賀県 長崎県 熊本県 大分県 宮崎県 鹿児島県 沖縄県"
).split()
PCODE = {n: f"{i + 1:02d}000" for i, n in enumerate(PREFS)}
_FWID = str.maketrans("０１２３４５６７８９", "0123456789")  # full-width -> half-width digits


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def app_id() -> str:
    load_dotenv(ENV_PATH)
    aid = os.environ.get("ESTAT_APP_ID")
    if not aid:
        sys.exit(
            "ERROR: ESTAT_APP_ID not set (expected in .env).\n"
            "Register a free key at https://www.e-stat.go.jp/api/ and add:\n"
            "    ESTAT_APP_ID=<your appId>"
        )
    return aid


def estat_get(endpoint: str, params: dict) -> dict:
    params = {"appId": app_id(), **params}
    r = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    # e-Stat wraps everything; surface API-level errors clearly.
    root = next(iter(js.values()))
    status = root.get("RESULT", {}).get("STATUS")
    if status not in (0, None):
        sys.exit(f"e-Stat API error {status}: {root.get('RESULT', {}).get('ERROR_MSG')}")
    return root


def _title(t) -> str:
    if isinstance(t, dict):
        return t.get("$", "")
    return str(t) if t is not None else ""


def cmd_list() -> int:
    """List candidate tables under the survey, flagging likely prefecture/annual ones."""
    root = estat_get("getStatsList", {
        "statsCode": STATS_CODE,
        "searchKind": 1,
        "limit": 1000,
    })
    tables = root.get("DATALIST_INF", {}).get("TABLE_INF", [])
    if isinstance(tables, dict):
        tables = [tables]
    print(f"found {len(tables)} tables under statsCode {STATS_CODE}\n")
    print("flag  statsDataId       survey       title")
    print("-" * 100)
    for t in tables:
        sid = t.get("@id", "")
        title = " / ".join(filter(None, [
            _title(t.get("STATISTICS_NAME")),
            _title(t.get("TITLE")),
        ]))
        survey = _title(t.get("SURVEY_DATE"))
        # heuristic: prefecture + overnight + annual signal in the title
        blob = title
        looks_pref = "都道府県" in blob
        looks_year = ("年" in survey) or ("年次" in blob) or len(str(survey)) <= 6
        looks_guest = "宿泊者" in blob or "延べ" in blob
        flag = "".join([
            "P" if looks_pref else " ",
            "G" if looks_guest else " ",
            "Y" if looks_year else " ",
        ])
        print(f"[{flag}] {sid:<17} {str(survey):<12} {title[:70]}")
    print("\nlegend: P=都道府県  G=宿泊者数  Y=annual.  Inspect a promising id with --inspect.")
    return 0


def _class_objs(stat_data: dict) -> dict:
    objs = stat_data.get("CLASS_INF", {}).get("CLASS_OBJ", [])
    if isinstance(objs, dict):
        objs = [objs]
    out = {}
    for o in objs:
        cls = o.get("CLASS", [])
        if isinstance(cls, dict):
            cls = [cls]
        out[o.get("@id")] = {
            "name": o.get("@name"),
            "codes": {c.get("@code"): c.get("@name") for c in cls},
        }
    return out


def cmd_inspect(stats_data_id: str) -> int:
    root = estat_get("getStatsData", {"statsDataId": stats_data_id, "limit": 1})
    sd = root.get("STATISTICAL_DATA", {})
    total = sd.get("RESULT_INF", {}).get("TOTAL_NUMBER")
    print(f"statsDataId {stats_data_id}  TOTAL_NUMBER={total}\n")
    objs = _class_objs(sd)
    for dim_id, info in objs.items():
        codes = info["codes"]
        print(f"dim @id={dim_id!r}  name={info['name']!r}  ({len(codes)} codes)")
        for code, name in list(codes.items())[:8]:
            print(f"     {code:<14} {name}")
        if len(codes) > 8:
            print(f"     ... (+{len(codes) - 8} more)")
        print()
    return 0


PREF_CODE_RE = re.compile(r"^(0[1-9]|[1-3][0-9]|4[0-7])000$")  # 01000..47000


def _is_prefecture(code: str, name: str) -> bool:
    if PREF_CODE_RE.match(str(code)):
        return True
    # fall back to 2-digit JIS codes 01..47 (some tables use these)
    return bool(re.match(r"^(0[1-9]|[1-3][0-9]|4[0-7])$", str(code))) and "全国" not in (name or "")


def _year_of(name: str) -> int | None:
    m = re.search(r"(20\d{2})", str(name))
    return int(m.group(1)) if m else None


def cmd_pull(ids: list[str]) -> int:
    frames = []
    for sid in ids:
        print(f"-> pulling {sid}")
        objs = None
        rows = []
        start = 1
        while True:
            params = {"statsDataId": sid, "limit": 100000}
            if start > 1:
                params["startPosition"] = start
            root = estat_get("getStatsData", params)
            sd = root.get("STATISTICAL_DATA", {})
            if objs is None:
                objs = _class_objs(sd)
            values = sd.get("DATA_INF", {}).get("VALUE", [])
            if isinstance(values, dict):
                values = [values]
            rows.extend(values)
            rinf = sd.get("RESULT_INF", {})
            nxt = rinf.get("NEXT_KEY")
            if not nxt:
                break
            start = int(nxt)
            time.sleep(RATE_SLEEP)

        # locate dimension ids
        area_id = next((d for d in objs if d == "area" or "地域" in (objs[d]["name"] or "")), None)
        time_id = next((d for d in objs if d == "time" or "時間" in (objs[d]["name"] or "")), None)
        tab_id = next((d for d in objs if d == "tab" or "表章" in (objs[d]["name"] or "")), None)
        if not (area_id and time_id):
            print(f"   WARN: could not identify area/time dims in {sid} (dims={list(objs)}); skipping")
            continue

        # The 宿泊旅行統計調査 tables carry the headline metric (延べ宿泊者数 /
        # 実宿泊者数) on `tab`, plus sub-breakdown dims (宿泊目的割合, 従業者数 /
        # 宿泊施設タイプ). One of those category dims doubles as the guest-type
        # axis: it holds a TOTAL member equal to the metric name and a FOREIGN
        # member (うち外国人…). We keep only the grand-total cell per
        # (prefecture, year): every OTHER category dim pinned to 合計, and split
        # the guest-type dim into total vs foreign. Monthly rows are dropped by
        # keeping annual-only time (name == 'YYYY年').
        metric_name = next(iter(objs[tab_id]["codes"].values())) if tab_id else None
        cat_dims = [d for d in objs if d not in (area_id, time_id, tab_id)]
        guest_dim = next(
            (d for d in cat_dims if metric_name in set(objs[d]["codes"].values())), None
        )
        other_dims = [d for d in cat_dims if d != guest_dim]

        kept = []
        for v in rows:
            acode = v.get(f"@{area_id}")
            aname = objs[area_id]["codes"].get(acode, "")
            if not _is_prefecture(acode, aname):
                continue
            tname = objs[time_id]["codes"].get(v.get(f"@{time_id}"), "")
            if not re.match(r"^\d{4}年$", tname):     # annual only, no 月
                continue
            yr = int(tname[:4])
            if yr not in YEARS:
                continue
            # every non-guest category dim must be at its total (合計 / 総数)
            if any(
                ("合計" not in objs[d]["codes"].get(v.get(f"@{d}"), "")
                 and "総数" not in objs[d]["codes"].get(v.get(f"@{d}"), ""))
                for d in other_dims
            ):
                continue
            # guest-type split
            if guest_dim:
                gname = objs[guest_dim]["codes"].get(v.get(f"@{guest_dim}"), "")
                if "外国人" in gname:
                    item = "foreign_overnight"
                elif gname == metric_name:
                    item = "total_overnight"
                else:
                    continue
            else:
                item = "total_overnight"
            try:
                val = float(v.get("$"))
            except (TypeError, ValueError):
                val = None
            kept.append({
                "stats_data_id": sid,
                "pref_code": acode,
                "prefecture": aname,
                "year": yr,
                "item": item,
                "metric": metric_name,
                "unit": v.get("@unit"),
                "value": val,
            })
        print(f"   kept {len(kept):,} prefecture x year x item rows")
        frames.append(pl.DataFrame(kept))

    if not frames:
        sys.exit("ERROR: no rows extracted. Re-check the statsDataId via --inspect.")
    df = pl.concat(frames, how="vertical_relaxed")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    print(f"\n-> wrote {OUT_PATH}  ({df.height:,} rows)")
    # quick summary
    print("\nitems present:")
    for item, n in df.group_by("item").len().sort("len", descending=True).iter_rows():
        print(f"   {n:>5}  {item}")
    print(f"\nprefectures: {df['prefecture'].n_unique()}   years: {sorted(df['year'].unique().to_list())}")
    return 0


# ----------------------------------------------------------------------------
# Excel route (full 2012-2019 panel).
#
# The API DB-ifies this survey for 2016 only (see COVERAGE NOTE). For a full
# year-by-year panel we pull each year's 年確定値 workbook and parse two sheets:
#   第2表(年計)      -> total guest-nights   (col 1)  [== API to the yen, QA-verified]
#   参考第1表(年計)  -> foreign guest-nights (col 1)
# Foreign here is the nationality-based 外国人延べ宿泊者数; it differs ~scale from
# the API's 第2表「うち外国人」 (which includes small facilities) but rank-correlates
# 0.9995, so relative-validation conclusions are unaffected. Prefecture rows are
# matched by NAME (2012-2013 .xls omit the JIS code that 2017+ .xlsx prepend), and
# sheet names are matched after normalising full-width digits (参考第１表 vs 参考第1表).
# ----------------------------------------------------------------------------
def _annual_xls_url(year: int) -> str | None:
    """Find the 年次 (annual confirmed) XLS download URL for a year via getDataCatalog."""
    start = 1
    while True:
        root = estat_get("getDataCatalog", {
            "statsCode": STATS_CODE, "limit": 100, "start": start,
            "surveyYears": f"{year}01-{year}12",
        })
        li = root.get("DATA_CATALOG_LIST_INF", {})
        inf = li.get("DATA_CATALOG_INF", [])
        if isinstance(inf, dict):
            inf = [inf]
        for e in inf:
            if e.get("DATASET", {}).get("TITLE", {}).get("CYCLE") != "年次":
                continue
            res = e.get("RESOURCES", {}).get("RESOURCE", [])
            if isinstance(res, dict):
                res = [res]
            for rr in res:
                if rr.get("FORMAT") == "XLS":
                    return rr.get("URL")
        nxt = li.get("RESULT_INF", {}).get("NEXT_KEY")
        if not nxt:
            return None
        start = int(nxt)


def cmd_download_excel() -> int:
    B1_XLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"-> downloading annual 年確定値 workbooks to {B1_XLS_DIR}")
    for yr in YEARS:
        url = _annual_xls_url(yr)
        if not url:
            print(f"   {yr}: no annual XLS found")
            continue
        content = requests.get(url, timeout=180).content
        ext = ".xlsx" if content[:4].hex().startswith("504b") else ".xls"
        fp = B1_XLS_DIR / f"b1_{yr}{ext}"
        fp.write_bytes(content)
        print(f"   {yr}: {fp.name}  {len(content):,} bytes")
        time.sleep(RATE_SLEEP)
    return 0


def _find_sheet(names: list[str], target: str) -> str | None:
    for n in names:
        if n.translate(_FWID).replace(" ", "") == target:
            return n
    return None


def _pref_of(cell0) -> str | None:
    s = str(cell0).replace("　", "").strip()          # drop full-width spaces
    m = re.match(r"^(\d{2})?(.+)$", s)                     # strip optional leading JIS code
    nm = m.group(2) if m else s
    return nm if nm in PCODE else None                    # exact match excludes 〜運輸局 rows


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_pref_col(rows: list) -> dict:
    """First numeric col-1 value for each prefecture row (the grand total)."""
    out = {}
    for r in rows:
        if not r:
            continue
        p = _pref_of(r[0])
        if p and p not in out and len(r) > 1:
            out[p] = _num(r[1])
    return out


def cmd_excel() -> int:
    files = sorted(B1_XLS_DIR.glob("b1_*.xls*"))
    if not files:
        sys.exit(f"ERROR: no workbooks in {B1_XLS_DIR}. Run --download-excel first.")
    rows = []
    for fp in files:
        yr = int(re.search(r"(\d{4})", fp.name).group(1))
        wb = CalamineWorkbook.from_path(str(fp))
        names = wb.sheet_names
        s_tot = _find_sheet(names, "第2表(年計)")
        s_for = _find_sheet(names, "参考第1表(年計)")
        if not s_tot:
            print(f"   WARN {yr}: 第2表(年計) sheet not found; skipping")
            continue
        tot = _parse_pref_col(wb.get_sheet_by_name(s_tot).to_python())
        forg = _parse_pref_col(wb.get_sheet_by_name(s_for).to_python()) if s_for else {}
        for p in PREFS:
            rows.append({"pref_code": PCODE[p], "prefecture": p, "year": yr,
                         "item": "total_overnight", "metric": "延べ宿泊者数",
                         "unit": "人泊", "value": tot.get(p), "source": fp.name})
            rows.append({"pref_code": PCODE[p], "prefecture": p, "year": yr,
                         "item": "foreign_overnight", "metric": "外国人延べ宿泊者数",
                         "unit": "人泊", "value": forg.get(p), "source": fp.name})
        nt = sum(v is not None for v in tot.values())
        nf = sum(v is not None for v in forg.values())
        print(f"   {yr}: total={nt}/47  foreign={nf}/47")
    if not rows:
        sys.exit("ERROR: parsed no rows.")
    df = pl.DataFrame(rows).sort(["year", "item", "pref_code"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)
    yrs = sorted(df["year"].unique().to_list())
    print(f"\n-> wrote {OUT_PATH}  ({df.height:,} rows; {len(yrs)} years {yrs})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Extract B1 宿泊旅行統計調査 prefecture overnight stats via e-Stat API.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="List candidate DB tables under the survey (API).")
    g.add_argument("--inspect", metavar="ID", help="Dump a DB table's dimension structure (API).")
    g.add_argument("--pull", nargs="+", metavar="ID", help="Pull a DB table to parquet (API; 2016 only).")
    g.add_argument("--download-excel", action="store_true",
                   help="Download annual 年確定値 XLS workbooks 2012-2019 (getDataCatalog).")
    g.add_argument("--excel", action="store_true",
                   help="Parse downloaded XLS into the full 2012-2019 panel parquet.")
    args = p.parse_args()

    if args.inspect:
        return cmd_inspect(args.inspect)
    if args.pull:
        return cmd_pull(args.pull)
    if args.download_excel:
        return cmd_download_excel()
    if args.excel:
        return cmd_excel()
    return cmd_list()  # default


if __name__ == "__main__":
    sys.exit(main())
