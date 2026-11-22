"""Free-flow travel times for every mesh centre in the candidate area.

The candidate area is not configured directly. It is derived from the screening
destination and ``origin_grid.freeflow_minutes`` using an upper bound on road
speed, so widening the horizon widens the search area automatically and no
second source of truth for the bounding box exists.

Nothing is filtered out here. Unreachable cells and cells that snap to a far
away road are kept with their reason recorded, because the free-flow output is also the
material for measuring the estimator's error later.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence

import pandas as pd

from landeval.mesh import JAPAN_LAT_RANGE, JAPAN_LON_RANGE, mesh_center, meshes_in_bbox
from landeval.osrm import Coordinate, OsrmClient, TableRow

# No road in Japan sustains this over a long distance, so a straight-line
# radius derived from it cannot exclude a cell that is reachable in time.
MAX_ROAD_SPEED_KMH = 120.0

# A road route is never shorter than the straight line, but interchange spacing
# and coastline detours mean the reachable set is far from circular. The margin
# only has to absorb the difference between "as the crow flies" and "usable
# road distance" near the boundary.
RADIUS_MARGIN = 1.15

_KM_PER_DEG_LAT = 111.32

# Beyond this, a mesh centre is not near any road: it is sea, forest or high
# mountain. Such cells keep their computed time but are marked off-road so that
# downstream steps reject them explicitly rather than by silent omission.
DEFAULT_MAX_SNAP_DISTANCE_M = 1000.0


def candidate_radius_km(minutes: float) -> float:
    """Upper bound on the straight-line distance reachable within ``minutes``."""
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    return MAX_ROAD_SPEED_KMH * (minutes / 60.0) * RADIUS_MARGIN


def candidate_bbox(destination: Coordinate, minutes: float) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) covering everything reachable in ``minutes``.

    The box is clipped to the Japanese grid, which is the domain mesh codes are
    defined over.
    """
    lat, lon = destination
    radius_km = candidate_radius_km(minutes)

    d_lat = radius_km / _KM_PER_DEG_LAT
    # Use the highest latitude of the box for the longitude conversion: the
    # circles of latitude shrink northward, so that is the widest span needed.
    worst_lat = min(abs(lat) + d_lat, 89.0)
    d_lon = radius_km / (_KM_PER_DEG_LAT * math.cos(math.radians(worst_lat)))

    return (
        max(lat - d_lat, JAPAN_LAT_RANGE[0]),
        max(lon - d_lon, JAPAN_LON_RANGE[0]),
        min(lat + d_lat, JAPAN_LAT_RANGE[1]),
        min(lon + d_lon, JAPAN_LON_RANGE[1]),
    )


def candidate_mesh_codes(destination: Coordinate, minutes: float, mesh_size_m: int) -> list[str]:
    """Enumerate every mesh code in the candidate area."""
    south, west, north, east = candidate_bbox(destination, minutes)
    return list(meshes_in_bbox(south, west, north, east, mesh_size_m))


def compute_freeflow(
    client: OsrmClient,
    mesh_codes: Sequence[str],
    destination: Coordinate,
    *,
    chunk_size: int = 5000,
    progress: bool = False,
) -> pd.DataFrame:
    """Compute free-flow travel time from every mesh centre to ``destination``."""
    centres = [mesh_center(code) for code in mesh_codes]
    rows = client.table_to_destination(centres, destination, chunk_size=chunk_size)
    return _to_frame(mesh_codes, centres, _with_progress(rows, len(mesh_codes), progress))


def _with_progress(rows: Iterable[TableRow], total: int, enabled: bool) -> Iterator[TableRow]:
    step = max(total // 20, 1)
    for index, row in enumerate(rows, start=1):
        if enabled and index % step == 0:
            print(f"  {index}/{total} ({100 * index / total:.0f}%)")
        yield row


def _to_frame(
    mesh_codes: Sequence[str], centres: Sequence[Coordinate], rows: Iterable[TableRow]
) -> pd.DataFrame:
    records = []
    for code, (lat, lon), row in zip(mesh_codes, centres, rows, strict=True):
        records.append(
            {
                "mesh_id": code,
                "lat": lat,
                "lon": lon,
                "freeflow_min": row.duration_min,
                "freeflow_distance_m": row.distance_m,
                "snap_distance_m": row.snap_distance_m,
            }
        )

    frame = pd.DataFrame.from_records(
        records,
        columns=["mesh_id", "lat", "lon", "freeflow_min", "freeflow_distance_m", "snap_distance_m"],
    )
    return frame.astype(
        {
            "mesh_id": "string",
            "lat": "float64",
            "lon": "float64",
            "freeflow_min": "Float64",
            "freeflow_distance_m": "Float64",
            "snap_distance_m": "float64",
        }
    )


def annotate(
    frame: pd.DataFrame,
    *,
    horizon_min: float,
    max_snap_distance_m: float = DEFAULT_MAX_SNAP_DISTANCE_M,
) -> pd.DataFrame:
    """Add the reachability and off-road flags used to select the working set.

    Returns a new frame; the input is not modified.
    """
    reachable = frame["freeflow_min"].notna()
    off_road = frame["snap_distance_m"] > max_snap_distance_m
    return frame.assign(
        is_reachable=reachable,
        is_off_road=off_road,
        # Both conditions must hold. A cell that is merely unreachable is not
        # the same as one with no road nearby, so the two flags stay separate
        # and this column is derived from them rather than replacing them.
        in_horizon=reachable & ~off_road & (frame["freeflow_min"] <= horizon_min),
    )
