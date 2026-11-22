"""Tests for the free-flow layer."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pandas as pd
import pytest

from landeval.freeflow import (
    MAX_ROAD_SPEED_KMH,
    RADIUS_MARGIN,
    annotate,
    candidate_bbox,
    candidate_mesh_codes,
    candidate_radius_km,
    compute_freeflow,
)
from landeval.mesh import JAPAN_LAT_RANGE, JAPAN_LON_RANGE, mesh_center
from landeval.osrm import OsrmClient

DEST = (35.6800, 139.7600)

Handler = Callable[[httpx.Request], httpx.Response]


# --- candidate area --------------------------------------------------------


def test_radius_scales_with_the_horizon() -> None:
    assert candidate_radius_km(60) == MAX_ROAD_SPEED_KMH * RADIUS_MARGIN
    assert candidate_radius_km(120) == pytest.approx(2 * candidate_radius_km(60))


@pytest.mark.parametrize("minutes", [0, -1, -135])
def test_non_positive_horizon_is_rejected(minutes: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        candidate_radius_km(minutes)


def test_bbox_contains_the_destination() -> None:
    south, west, north, east = candidate_bbox(DEST, 135)

    assert south < DEST[0] < north
    assert west < DEST[1] < east


def test_bbox_grows_with_the_horizon() -> None:
    small = candidate_bbox(DEST, 60)
    large = candidate_bbox(DEST, 135)

    assert large[0] < small[0] and large[1] < small[1]
    assert large[2] > small[2] and large[3] > small[3]


def test_bbox_is_clipped_to_the_japanese_grid() -> None:
    """A horizon wide enough to leave the grid must not produce codes outside it."""
    south, west, north, east = candidate_bbox(DEST, 100_000)

    assert (south, north) == JAPAN_LAT_RANGE
    assert (west, east) == JAPAN_LON_RANGE


def test_bbox_covers_a_point_at_the_nominal_maximum_range() -> None:
    """A point exactly at max speed x horizon due north must fall inside the box."""
    radius_km = MAX_ROAD_SPEED_KMH * (135 / 60)
    lat = DEST[0] + radius_km / 111.32
    south, _, north, _ = candidate_bbox(DEST, 135)

    assert south < lat < north


def test_mesh_codes_cover_the_destination_cell() -> None:
    codes = candidate_mesh_codes(DEST, 30, 500)

    assert len(codes) == len(set(codes))
    lats = [mesh_center(c)[0] for c in codes]
    lons = [mesh_center(c)[1] for c in codes]
    assert min(lats) < DEST[0] < max(lats)
    assert min(lons) < DEST[1] < max(lons)


# --- table -> frame --------------------------------------------------------


def _distance_for(duration: float | None) -> float | None:
    return None if duration is None else duration * 12


def table_handler(durations: list[float | None], snap: list[float]) -> Handler:
    """Serve batches out of ``durations`` in request order, as OSRM would.

    Every coordinate is a source and the last one is the destination, so each
    response carries one row more than the batch.
    """
    served = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal served
        n = len(request.url.path.rsplit("/", 1)[-1].split(";")) - 1
        window: list[float | None] = [*durations[served : served + n], 0.0]
        snaps = [*snap[served : served + n], 0.0]
        served += n
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "durations": [[d] for d in window],
                "distances": [[_distance_for(d)] for d in window],
                "sources": [{"distance": s} for s in snaps],
                "destinations": [{"distance": 0.0}],
            },
        )

    return handler


def test_frame_has_one_row_per_mesh_in_input_order() -> None:
    codes = candidate_mesh_codes(DEST, 5, 500)
    durations: list[float | None] = [60.0 * (i + 1) for i in range(len(codes))]
    snap = [0.0] * len(codes)

    with OsrmClient(
        "http://osrm.test", transport=httpx.MockTransport(table_handler(durations, snap))
    ) as client:
        frame = compute_freeflow(client, codes, DEST)

    assert list(frame["mesh_id"]) == codes
    assert frame["freeflow_min"].tolist() == [float(i + 1) for i in range(len(codes))]


def test_unreachable_rows_are_kept_as_null() -> None:
    codes = candidate_mesh_codes(DEST, 5, 500)
    durations: list[float | None] = [None] * len(codes)
    durations[0] = 600.0

    with OsrmClient(
        "http://osrm.test",
        transport=httpx.MockTransport(table_handler(durations, [0.0] * len(codes))),
    ) as client:
        frame = compute_freeflow(client, codes, DEST)

    assert len(frame) == len(codes)
    assert frame["freeflow_min"].isna().sum() == len(codes) - 1


def test_frame_is_empty_but_well_formed_for_no_meshes() -> None:
    with OsrmClient(
        "http://osrm.test", transport=httpx.MockTransport(table_handler([], []))
    ) as client:
        frame = compute_freeflow(client, [], DEST)

    assert len(frame) == 0
    assert list(frame.columns) == [
        "mesh_id",
        "lat",
        "lon",
        "freeflow_min",
        "freeflow_distance_m",
        "snap_distance_m",
    ]


# --- annotation ------------------------------------------------------------


def make_frame(rows: list[tuple[float | None, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mesh_id": pd.array([f"m{i}" for i in range(len(rows))], dtype="string"),
            "lat": [35.0] * len(rows),
            "lon": [139.0] * len(rows),
            "freeflow_min": pd.array([r[0] for r in rows], dtype="Float64"),
            "freeflow_distance_m": pd.array([None] * len(rows), dtype="Float64"),
            "snap_distance_m": [r[1] for r in rows],
        }
    )


def test_annotate_flags_reachability_and_off_road_separately() -> None:
    frame = annotate(make_frame([(30.0, 5.0), (None, 5.0), (30.0, 5000.0)]), horizon_min=135)

    assert frame["is_reachable"].tolist() == [True, False, True]
    assert frame["is_off_road"].tolist() == [False, False, True]


def test_annotate_excludes_off_road_cells_from_the_horizon() -> None:
    """A sea cell snapping to a distant road must not enter the working set."""
    frame = annotate(make_frame([(30.0, 5.0), (30.0, 5000.0)]), horizon_min=135)

    assert frame["in_horizon"].tolist() == [True, False]


def test_annotate_applies_the_horizon() -> None:
    frame = annotate(make_frame([(134.9, 0.0), (135.0, 0.0), (135.1, 0.0)]), horizon_min=135)

    assert frame["in_horizon"].tolist() == [True, True, False]


def test_annotate_treats_unreachable_as_outside_the_horizon() -> None:
    """Null must never be read as 'fast'."""
    frame = annotate(make_frame([(None, 0.0)]), horizon_min=135)

    assert frame["in_horizon"].tolist() == [False]


def test_annotate_does_not_mutate_its_input() -> None:
    original = make_frame([(30.0, 5.0)])
    columns_before = list(original.columns)

    annotate(original, horizon_min=135)

    assert list(original.columns) == columns_before


def test_annotate_honours_a_custom_snap_limit() -> None:
    frame = annotate(make_frame([(30.0, 500.0)]), horizon_min=135, max_snap_distance_m=100.0)

    assert frame["is_off_road"].tolist() == [True]
    assert frame["in_horizon"].tolist() == [False]
