"""
Extract OSM POIs from a Geofabrik PBF and write to parquet.

Filters per SPEC.md section 5.3:
  tourism=*
  amenity in {restaurant, cafe, bar, fast_food, hospital, pharmacy, atm, bank, fuel}
  shop    in {supermarket, convenience, bakery, mall}
  public_transport=*

Nodes are kept directly. Ways/relations are assembled into areas and reduced
to the arithmetic mean of their outer-ring vertices (close enough for the
H3 r6 ~36 km^2 cells we aggregate over downstream).

Usage:
    uv run python pipeline/extract/extract_osm_pois.py \\
        --pbf data/cache/georgia-latest.osm.pbf \\
        --out data/cache/pois_raw.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import osmium
import polars as pl

AMENITY_KEEP = {
    "restaurant", "cafe", "bar", "fast_food",
    "hospital", "pharmacy", "atm", "bank", "fuel",
}
SHOP_KEEP = {"supermarket", "convenience", "bakery", "mall"}


def pick_primary(tags) -> tuple[str, str] | None:
    """Return (key, value) of the highest-priority matching POI tag, else None."""
    if "tourism" in tags:
        return ("tourism", tags["tourism"])
    if "amenity" in tags and tags["amenity"] in AMENITY_KEEP:
        return ("amenity", tags["amenity"])
    if "shop" in tags and tags["shop"] in SHOP_KEEP:
        return ("shop", tags["shop"])
    if "public_transport" in tags:
        return ("public_transport", tags["public_transport"])
    return None


class POIHandler(osmium.SimpleHandler):
    def __init__(self, process_areas: bool = True) -> None:
        super().__init__()
        self.rows: list[tuple] = []
        self.process_areas = process_areas

    def node(self, n) -> None:
        if not n.location.valid():
            return
        kv = pick_primary(n.tags)
        if kv is None:
            return
        self.rows.append((
            int(n.id),
            "node",
            float(n.location.lat),
            float(n.location.lon),
            kv[0],
            kv[1],
            n.tags.get("name"),
        ))

    def area(self, a) -> None:
        if not self.process_areas:
            return
        kv = pick_primary(a.tags)
        if kv is None:
            return
        lats: list[float] = []
        lons: list[float] = []
        try:
            for ring in a.outer_rings():
                for ref in ring:
                    lats.append(ref.lat)
                    lons.append(ref.lon)
        except Exception:
            return
        if not lats:
            return
        self.rows.append((
            int(a.orig_id()),
            "area",
            sum(lats) / len(lats),
            sum(lons) / len(lons),
            kv[0],
            kv[1],
            a.tags.get("name"),
        ))


def main() -> int:
    p = argparse.ArgumentParser(description="Extract OSM POIs to parquet.")
    p.add_argument("--pbf", required=True, type=Path, help="Input Geofabrik .osm.pbf")
    p.add_argument("--out", required=True, type=Path, help="Output parquet path")
    p.add_argument("--index", default="flex_mem",
                   help="pyosmium node-location index. 'flex_mem' (default, RAM) is fine for "
                        "country extracts up to ~Georgia. For large files (e.g. Japan 2.3GB) on "
                        "low-RAM machines, use a disk-backed index, e.g. "
                        "'sparse_file_array,data/cache/nodecache.bin'.")
    p.add_argument("--no-areas", action="store_true",
                   help="Skip way/relation area POIs (nodes only). Much faster + lower memory; "
                        "drops the ~8%% of POIs mapped as polygons (large malls, some hotels).")
    args = p.parse_args()

    if not args.pbf.exists():
        sys.exit(f"ERROR: PBF not found: {args.pbf}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    size_mb = args.pbf.stat().st_size / 1e6
    areas = not args.no_areas
    print(f"-> Reading {args.pbf} ({size_mb:.1f} MB)  index={args.index}  areas={areas}")
    t0 = time.time()
    h = POIHandler(process_areas=areas)
    h.apply_file(str(args.pbf), locations=True, idx=args.index)
    print(f"   parsed in {time.time() - t0:.1f}s, {len(h.rows):,} POIs found")

    if not h.rows:
        sys.exit("ERROR: no POIs matched. Check the filter set.")

    df = pl.DataFrame(
        h.rows,
        schema={
            "osm_id": pl.Int64,
            "osm_type": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "primary_key": pl.Utf8,
            "primary_value": pl.Utf8,
            "name": pl.Utf8,
        },
        orient="row",
    )
    print(f"-> Writing {args.out}")
    df.write_parquet(args.out)

    print("\nPOI counts by primary_key:")
    counts = df.group_by("primary_key").len().sort("len", descending=True).to_dicts()
    for row in counts:
        print(f"   {row['primary_key']:<20} {row['len']:>7,}")
    print(f"\nTotal rows written: {len(df):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
