"""Choosing the representative points that calibrate the free-flow filter.

The filter threshold is set from the relationship between OSRM time and
measured time, so the sample has to span that relationship rather than
be spread evenly over the map. Two things make the relationship vary:

* **Direction.** Which corridor a trip uses decides how much of it is
  expressway, and OSRM's speed bias is largest exactly there.
* **Distance.** The threshold is read off the far end of the range, so the
  sample must reach it; a sample clustered near the destination would leave the
  boundary unmeasured.

Points are therefore drawn from a bearing x OSRM-time grid, one per cell. Within
a cell the point closest to a road is taken: a mesh centre that snaps 800 m away
sits somewhere no one drives from, and calibrating on it would measure the walk
to the road as much as the drive.
"""

from __future__ import annotations

import math

import pandas as pd

# Bearing sectors around the destination. Twelve keeps each sector narrow
# enough that one corridor dominates it.
DEFAULT_BEARING_SECTORS = 12


def sector_key(value: float) -> int:
    """The sector a value names, refusing anything that falls between two.

    Every sector-keyed lookup in the codebase -- a direction's fitted line, its
    fitted range, the check that a block correction stays inside the sector it
    was measured in -- reaches its table through this function, not through
    ``int(value)``, which truncates. A 3.7 arriving from a float column, a merge
    or a hand-built frame would silently become sector 3 and borrow that
    sector's line, its range and its corrections. There is no sector 3.7, so the
    honest answer is an error rather
    than the nearest one (the boundary is not anchored, so an adjacent
    token would be accepted).

    The pipeline writes ``sector`` as int64 for every cell, so this refuses
    nothing it produces. It is a guard on the shape of
    the key, placed where the key is read rather than where it happens to be
    written correctly.
    """
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"sector {value!r} is not a whole sector")
    return int(number)


def bearing_deg(origin_lat: float, origin_lon: float, lat: float, lon: float) -> float:
    """Initial bearing from the destination to a cell, in degrees clockwise from north."""
    phi1, phi2 = math.radians(origin_lat), math.radians(lat)
    delta = math.radians(lon - origin_lon)
    y = math.sin(delta) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta)
    return math.degrees(math.atan2(y, x)) % 360.0


def assign_bins(
    frame: pd.DataFrame,
    destination: tuple[float, float],
    *,
    sectors: int = DEFAULT_BEARING_SECTORS,
    time_bands: tuple[float, ...],
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``bearing_deg``, ``sector`` and ``time_band`` columns."""
    if sectors < 1:
        raise ValueError(f"sectors must be positive, got {sectors}")
    if not time_bands or list(time_bands) != sorted(time_bands):
        raise ValueError(f"time_bands must be non-empty and ascending, got {time_bands}")

    lat0, lon0 = destination
    bearings = [
        bearing_deg(lat0, lon0, lat, lon)
        for lat, lon in zip(frame["lat"], frame["lon"], strict=True)
    ]
    # Sectors are centred on the cardinal directions rather than starting at
    # them. A corridor running due north would otherwise straddle the boundary
    # between the first and last sector and be sampled as two half-corridors.
    width = 360.0 / sectors
    band_index = pd.cut(
        frame["freeflow_min"].astype("float64"),
        bins=[float("-inf"), *time_bands],
        labels=list(range(len(time_bands))),
        right=True,
    )
    return frame.assign(
        bearing_deg=bearings,
        sector=[int(((b + width / 2) % 360.0) // width) % sectors for b in bearings],
        time_band=band_index,
    )


def _extra_from_bin(group: pd.DataFrame, wanted: int, chosen: str) -> pd.DataFrame:
    """Pick ``wanted`` further cells from one bin, spread across its time range.

    Taking the next-closest-to-a-road cells would cluster them on the same
    stretch of the same road, which adds cost without adding information. Rank
    positions spaced evenly through the bin's free-flow range spread the sample
    instead.
    """
    rest = group[group["mesh_id"] != chosen]
    rest = rest[rest["snap_distance_m"] <= MAX_EXTRA_SNAP_DISTANCE_M]
    if rest.empty or wanted < 1:
        return rest.head(0)

    ordered = rest.sort_values(["freeflow_min", "mesh_id"], kind="mergesort")
    step = len(ordered) / (wanted + 1)
    positions = sorted({min(int(step * (i + 1)), len(ordered) - 1) for i in range(wanted)})
    return ordered.iloc[positions]


# Extra points still have to sit on a road; the first pick per bin is chosen by
# minimum snap distance, but the spread picks need their own ceiling.
MAX_EXTRA_SNAP_DISTANCE_M = 100.0


def select_calibration_points(
    frame: pd.DataFrame,
    destination: tuple[float, float],
    *,
    sectors: int = DEFAULT_BEARING_SECTORS,
    time_bands: tuple[float, ...],
    per_bin: int = 1,
) -> pd.DataFrame:
    """Pick ``per_bin`` representative cells per (bearing sector, time band).

    The first pick of every bin is the cell closest to a road, and that choice
    does not change when ``per_bin`` grows. Points already measured therefore
    stay in the sample and their cached measurements keep their value; raising
    ``per_bin`` only costs calls for the cells being added.

    Empty combinations -- a sector pointing out to sea, say -- simply yield no
    point. They are absent from the result rather than filled from a neighbour,
    so a thin sample stays visible instead of looking complete.
    """
    if per_bin < 1:
        raise ValueError(f"per_bin must be at least 1, got {per_bin}")

    binned = assign_bins(frame, destination, sectors=sectors, time_bands=time_bands)
    binned = binned[binned["time_band"].notna()]
    if binned.empty:
        return binned.head(0)

    # Closest to a road wins; mesh_id breaks ties so the choice is reproducible.
    ordered = binned.sort_values(["snap_distance_m", "mesh_id"], kind="mergesort")
    grouped = ordered.groupby(["sector", "time_band"], observed=True, sort=True)
    picked = [grouped.head(1)]

    if per_bin > 1:
        for _, group in grouped:
            first = str(group.iloc[0]["mesh_id"])
            picked.append(_extra_from_bin(group, per_bin - 1, first))

    result = pd.concat(picked, ignore_index=False)
    result = result[~result.index.duplicated(keep="first")]
    return result.sort_values(
        ["sector", "time_band", "freeflow_min", "mesh_id"], kind="mergesort"
    ).reset_index(drop=True)
