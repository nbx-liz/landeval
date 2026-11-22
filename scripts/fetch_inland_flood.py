"""Ingest the inland-flood zone for the municipalities that publish one.

The conditions this script implements literally:

1. the roster comes first -- without it a 404 tile is ambiguous, so a missing
   roster stops the run rather than defaulting to anything;
2. inside a roster municipality, a 404 tile means the point is outside the
   zone;
3. outside the roster, the cell keeps ``no_data``; and
4. nothing is guessed. A server error is neither 200 nor 404 and stops the run.

The judgement is the same 25 sample points per 500 m cell the vector layers
use, so the ratio means the same thing in every layer. Depth is not read from
the raster: 「かかる／かからない」 only.

Two artefacts come out, and they are separate on purpose. The per-cell status
goes into ``mesh_hazard.parquet`` beside the other layers. The per-point
readings go into their own ``inland_flood_points.npz`` rather than into
``hazard_points.npz``, because availability here is per municipality: folding
it into the main index would make every cell outside the roster read as 区域外,
which is the false-clean answer the roster exists to prevent.

    uv run python scripts/fetch_inland_flood.py --dry-run
    uv run python scripts/fetch_inland_flood.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _root import add_root_argument, resolve_root, under_root

from landeval.hazard import (
    NO_DATA,
    STATUS_SUFFIX,
    SUBGRID,
    covered_column,
    layer_status,
    subcell_lonlat,
)
from landeval.inland_flood import (
    ABSENT,
    DEFAULT_ZOOM,
    InlandFloodError,
    TileClient,
    load_roster,
    publishes,
    save_points,
)

LAYER = "inland_flood"
CELL_SIZE_M = 500


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument(
        "--roster", type=Path, default=under_root("data/raw/hazard/naisui_roster.csv")
    )
    parser.add_argument(
        "--cells", type=Path, default=under_root("data/processed/final_cells.parquet")
    )
    parser.add_argument(
        "--hazard", type=Path, default=under_root("data/processed/mesh_hazard.parquet")
    )
    parser.add_argument("--cache-dir", type=Path, default=under_root("data/cache/naisui"))
    parser.add_argument(
        "--points-out", type=Path, default=under_root("data/processed/inland_flood_points.npz")
    )
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    parser.add_argument("--limit-cells", type=int, default=0, help="0 = every eligible cell")
    parser.add_argument("--dry-run", action="store_true", help="plan only; fetch nothing")
    args = parser.parse_args()
    resolve_root(args)

    try:
        roster = load_roster(args.roster)
    except InlandFloodError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    cells = pd.read_parquet(args.cells, columns=["mesh_id", "lat", "lon", "muni_code", "muni_name"])
    eligible = cells[
        cells["muni_code"].astype(str).map(lambda code: publishes(roster, code))
    ].reset_index(drop=True)
    if args.limit_cells:
        eligible = eligible.head(args.limit_cells)

    print(f"名簿          : {roster.rows} 行 / {len(roster.codes)} 団体（{args.roster}）")
    print(f"公開自治体のセル: {len(eligible):,} / 作業セット {len(cells):,}")
    print(f"標本点        : {len(eligible) * SUBGRID * SUBGRID:,}（1 セル {SUBGRID}x{SUBGRID}）")
    if eligible.empty:
        print("該当セルがありません。全セル no_data のままです")
        return 0

    lon, lat, cell_index = subcell_lonlat(eligible, mesh_size_m=CELL_SIZE_M, subgrid=SUBGRID)
    if args.dry_run:
        from landeval.inland_flood import tile_index

        tiles = {tile_index(float(a), float(o), args.zoom) for a, o in zip(lat, lon, strict=True)}
        print(f"必要なタイル  : {len(tiles):,}（ズーム {args.zoom}）")
        print("--dry-run のため取得しない")
        return 0

    per_cell = SUBGRID * SUBGRID
    membership = np.zeros(len(lat), dtype=np.uint8)
    present = np.zeros(len(eligible), dtype=bool)
    try:
        with TileClient(cache_dir=args.cache_dir, zoom=args.zoom, delay_s=args.delay) as client:
            for index in range(len(lat)):
                state, inside = client.reading_at(float(lat[index]), float(lon[index]))
                membership[index] = 1 if inside else 0
                if state != ABSENT:
                    present[cell_index[index]] = True
                if index % 5000 == 0 or index == len(lat) - 1:
                    print(
                        f"  [{index + 1:,}/{len(lat):,}] 取得 {client.requests:,} 件",
                        flush=True,
                    )
            requests = client.requests
    except InlandFloodError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    hits = membership.reshape(len(eligible), per_cell).sum(axis=1)
    ratio = hits / per_cell
    # A 404 everywhere in the cell still means "outside the zone" here: the
    # municipality publishes, so the absence of a tile is the absence of a
    # drawn area. `present` is kept only to report how many cells had any tile
    # at all, which is what makes the 404-heavy municipalities visible.
    statuses = [
        layer_status(float(value), float("nan"), available=True, designated=True) for value in ratio
    ]

    hazard = pd.read_parquet(args.hazard)
    ratio_column = f"{LAYER}_hazard_ratio"
    status_column = f"{LAYER}{STATUS_SUFFIX}"
    updates = pd.DataFrame(
        {
            "mesh_id": eligible["mesh_id"].astype(str),
            ratio_column: ratio,
            status_column: statuses,
        }
    ).set_index("mesh_id")

    flag_column = covered_column(LAYER)
    if flag_column not in hazard.columns:
        print(
            f"error: {args.hazard} has no {flag_column}; it predates the coverage column, "
            "so rebuild it with scripts/build_hazard.py first",
            file=sys.stderr,
        )
        return 1
    hazard = hazard.set_index(hazard["mesh_id"].astype(str))
    covered = hazard.index.intersection(updates.index)
    if len(covered) != len(updates):
        print(
            f"error: {len(updates) - len(covered)} cells are absent from {args.hazard}",
            file=sys.stderr,
        )
        return 1
    hazard.loc[covered, ratio_column] = updates.loc[covered, ratio_column]
    hazard.loc[covered, status_column] = updates.loc[covered, status_column]
    # Only the roster's cells were read; every other cell keeps the False that
    # scripts/build_hazard.py wrote for the layer.
    hazard.loc[covered, flag_column] = True
    hazard = hazard.reset_index(drop=True)
    hazard.to_parquet(args.hazard, index=False)

    save_points(
        args.points_out,
        layer=LAYER,
        mesh_ids=eligible["mesh_id"].astype(str).tolist(),
        membership=membership,
        points_per_cell=per_cell,
    )

    inside_cells = int((ratio > 0).sum())
    print()
    print(f"取得          : {requests:,} タイル")
    print(f"タイルのあったセル: {int(present.sum()):,} / {len(eligible):,}")
    print(f"区域内のセル  : {inside_cells:,}（面積率 > 0）")
    print(f"区域外のセル  : {len(eligible) - inside_cells:,}")
    print(f"no_data のまま: {int((hazard[status_column] == NO_DATA).sum()):,} セル")
    print(f"wrote {args.hazard}")
    print(f"wrote {args.points_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
