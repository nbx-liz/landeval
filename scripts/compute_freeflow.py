"""Compute free-flow travel time from every candidate mesh to the screening destination.

Writes two artefacts:

* ``freeflow_all.parquet``   -- every cell in the candidate box, including the
  unreachable and off-road ones. This is the raw free-flow record and is never
  filtered in place.
* ``freeflow_horizon.parquet`` -- the working set: reachable, on-road, and
  within ``origin_grid.freeflow_minutes``.

Usage (with an OSRM server built from the extract already running):
    uv run python scripts/compute_freeflow.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _root import add_root_argument, resolve_root, under_root

from landeval.config import load_config
from landeval.freeflow import (
    DEFAULT_MAX_SNAP_DISTANCE_M,
    annotate,
    candidate_bbox,
    candidate_mesh_codes,
    compute_freeflow,
)
from landeval.osrm import DEFAULT_BASE_URL, OsrmClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--config", type=Path, default=under_root("config/destinations.yaml"))
    parser.add_argument("--out", type=Path, default=under_root("data/processed"))
    parser.add_argument("--osrm-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--max-snap-distance-m",
        type=float,
        default=DEFAULT_MAX_SNAP_DISTANCE_M,
        help="beyond this, a mesh centre is treated as having no road (sea, forest, mountain)",
    )
    args = parser.parse_args()
    resolve_root(args)

    config = load_config(args.config)
    group_id, group = config.screening_group()
    destination = (group.points[0].lat, group.points[0].lon)
    horizon = config.origin_grid.freeflow_minutes
    mesh_size = config.origin_grid.mesh_size_m

    south, west, north, east = candidate_bbox(destination, horizon)
    print(f"screening group : {group_id} -> {group.points[0].name} {destination}")
    print(f"horizon         : {horizon} min free-flow")
    print(f"candidate bbox  : S{south:.4f} W{west:.4f} N{north:.4f} E{east:.4f}")

    print("enumerating mesh cells")
    codes = candidate_mesh_codes(destination, horizon, mesh_size)
    print(f"  {len(codes):,} cells of {mesh_size} m")

    started = time.monotonic()
    with OsrmClient(args.osrm_url) as client:
        frame = compute_freeflow(
            client, codes, destination, chunk_size=args.chunk_size, progress=True
        )
    elapsed = time.monotonic() - started
    print(f"routing took {elapsed / 60:.1f} min")

    frame = annotate(frame, horizon_min=horizon, max_snap_distance_m=args.max_snap_distance_m)

    args.out.mkdir(parents=True, exist_ok=True)
    all_path = args.out / "freeflow_all.parquet"
    horizon_path = args.out / "freeflow_horizon.parquet"
    frame.to_parquet(all_path, index=False)
    working = frame[frame["in_horizon"]]
    working.to_parquet(horizon_path, index=False)

    usable = int((frame["is_reachable"] & ~frame["is_off_road"]).sum())
    print()
    print(f"total cells      : {len(frame):,}")
    print(f"  unreachable    : {int((~frame['is_reachable']).sum()):,}")
    print(f"  off-road       : {int(frame['is_off_road'].sum()):,}")
    print(f"  beyond horizon : {usable - len(working):,}")
    print(f"  working set    : {len(working):,}")
    print()
    print(f"wrote {all_path}")
    print(f"wrote {horizon_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
