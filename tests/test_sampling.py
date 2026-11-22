"""Tests for calibration-point selection."""

from __future__ import annotations

import pandas as pd
import pytest

from landeval.sampling import assign_bins, bearing_deg, select_calibration_points

# An evenly spaced grid chosen for these tests; the bands a real run uses are the
# caller's choice.
CALIBRATION_TIME_BANDS: tuple[float, ...] = (25.0, 55.0, 85.0, 115.0, 145.0)

DEST = (35.680000, 139.760000)


def make_frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """rows: (mesh_id, lat, lon, freeflow_min, snap_distance_m)."""
    return pd.DataFrame(
        {
            "mesh_id": pd.array([r[0] for r in rows], dtype="string"),
            "lat": [r[1] for r in rows],
            "lon": [r[2] for r in rows],
            "freeflow_min": pd.array([r[3] for r in rows], dtype="Float64"),
            "snap_distance_m": [r[4] for r in rows],
        }
    )


# --- bearing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [
        (36.680000, 139.760000, 0.0),  # due north
        (35.680000, 140.760000, 90.0),  # due east
        (34.680000, 139.760000, 180.0),  # due south
        (35.680000, 138.760000, 270.0),  # due west
    ],
)
def test_cardinal_bearings(lat: float, lon: float, expected: float) -> None:
    assert bearing_deg(*DEST, lat, lon) == pytest.approx(expected, abs=1.0)


def test_bearing_is_always_in_range() -> None:
    for lat in (33.0, 35.0, 38.0):
        for lon in (137.0, 139.7, 142.0):
            assert 0.0 <= bearing_deg(*DEST, lat, lon) < 360.0


# --- binning ---------------------------------------------------------------


def test_sectors_partition_the_compass() -> None:
    frame = make_frame(
        [
            ("a", 36.6, 139.76, 60.0, 5.0),  # north
            ("b", 35.68, 140.7, 60.0, 5.0),  # east
            ("c", 34.7, 139.76, 60.0, 5.0),  # south
            ("d", 35.68, 138.8, 60.0, 5.0),  # west
        ]
    )

    binned = assign_bins(frame, DEST, sectors=4, time_bands=CALIBRATION_TIME_BANDS)

    assert sorted(binned["sector"].tolist()) == [0, 1, 2, 3]


def test_time_bands_are_upper_inclusive() -> None:
    frame = make_frame(
        [
            ("a", 36.0, 139.76, 25.0, 5.0),
            ("b", 36.0, 139.76, 25.1, 5.0),
            ("c", 36.0, 139.76, 145.0, 5.0),
        ]
    )

    binned = assign_bins(frame, DEST, time_bands=(25.0, 55.0, 85.0, 115.0, 145.0))

    assert binned["time_band"].tolist() == [0, 1, 4]


def test_cells_beyond_the_last_band_are_unbinned() -> None:
    frame = make_frame([("a", 36.0, 139.76, 146.0, 5.0)])

    binned = assign_bins(frame, DEST, time_bands=(145.0,))

    assert binned["time_band"].isna().all()


@pytest.mark.parametrize("sectors", [0, -1])
def test_invalid_sector_count_is_rejected(sectors: int) -> None:
    frame = make_frame([("a", 36.0, 139.76, 60.0, 5.0)])

    with pytest.raises(ValueError, match="sectors"):
        assign_bins(frame, DEST, sectors=sectors, time_bands=CALIBRATION_TIME_BANDS)


@pytest.mark.parametrize("bands", [(), (60.0, 30.0)])
def test_invalid_time_bands_are_rejected(bands: tuple[float, ...]) -> None:
    frame = make_frame([("a", 36.0, 139.76, 60.0, 5.0)])

    with pytest.raises(ValueError, match="time_bands"):
        assign_bins(frame, DEST, time_bands=bands)


# --- selection -------------------------------------------------------------


def test_one_point_per_populated_bin() -> None:
    frame = make_frame(
        [
            ("n1", 36.2, 139.76, 40.0, 10.0),
            ("n2", 36.3, 139.76, 45.0, 20.0),
            ("e1", 35.68, 140.3, 40.0, 10.0),
            ("s1", 35.0, 139.76, 100.0, 10.0),
        ]
    )

    picked = select_calibration_points(frame, DEST, sectors=4, time_bands=CALIBRATION_TIME_BANDS)

    assert len(picked) == 3
    assert set(picked["mesh_id"]) == {"n1", "e1", "s1"}


def test_the_cell_closest_to_a_road_wins_its_bin() -> None:
    frame = make_frame(
        [
            ("far", 36.2, 139.76, 40.0, 900.0),
            ("near", 36.3, 139.76, 45.0, 12.0),
        ]
    )

    picked = select_calibration_points(frame, DEST, sectors=4, time_bands=CALIBRATION_TIME_BANDS)

    assert picked["mesh_id"].tolist() == ["near"]


def test_empty_bins_are_absent_rather_than_filled() -> None:
    """A sector pointing out to sea must not borrow a point from its neighbour."""
    frame = make_frame([("n1", 36.2, 139.76, 40.0, 10.0)])

    picked = select_calibration_points(frame, DEST, sectors=12, time_bands=CALIBRATION_TIME_BANDS)

    assert len(picked) == 1


def test_selection_is_deterministic() -> None:
    frame = make_frame(
        [
            ("a", 36.2, 139.76, 40.0, 10.0),
            ("b", 36.3, 139.75, 45.0, 10.0),  # identical snap distance
        ]
    )

    first = select_calibration_points(frame, DEST, sectors=4, time_bands=CALIBRATION_TIME_BANDS)
    second = select_calibration_points(frame, DEST, sectors=4, time_bands=CALIBRATION_TIME_BANDS)

    assert first["mesh_id"].tolist() == second["mesh_id"].tolist()


def test_empty_input_yields_an_empty_selection() -> None:
    picked = select_calibration_points(make_frame([]), DEST, time_bands=CALIBRATION_TIME_BANDS)

    assert len(picked) == 0


def test_selection_spans_the_full_time_range() -> None:
    """The threshold is read off the far end, so the sample must reach it."""
    rows = [(f"m{i}", 35.680000 + 0.1 * i, 139.76, 20.0 + 25 * i, 5.0) for i in range(6)]
    picked = select_calibration_points(
        make_frame(rows), DEST, sectors=4, time_bands=CALIBRATION_TIME_BANDS
    )

    assert picked["freeflow_min"].max() >= 120


# --- multiple points per bin -----------------------------------------------


def bin_frame(n: int) -> pd.DataFrame:
    """One bin's worth of cells, spread over the 25-55 minute band."""
    return make_frame(
        [(f"m{i:03d}", 36.0 + 0.001 * i, 139.76, 31.0 + i * 0.5, 5.0) for i in range(n)]
    )


@pytest.mark.parametrize("per_bin", [0, -1])
def test_non_positive_per_bin_is_rejected(per_bin: int) -> None:
    with pytest.raises(ValueError, match="per_bin"):
        select_calibration_points(
            bin_frame(10), DEST, sectors=4, per_bin=per_bin, time_bands=CALIBRATION_TIME_BANDS
        )


def test_more_points_per_bin_are_returned() -> None:
    picked = select_calibration_points(
        bin_frame(20), DEST, sectors=4, per_bin=3, time_bands=CALIBRATION_TIME_BANDS
    )

    assert len(picked) == 3
    assert picked["mesh_id"].nunique() == 3


def test_raising_per_bin_keeps_the_previous_choice() -> None:
    """Existing measurements must stay in the sample so their cache still counts."""
    frame = bin_frame(20)
    before = set(
        select_calibration_points(
            frame, DEST, sectors=4, per_bin=1, time_bands=CALIBRATION_TIME_BANDS
        )["mesh_id"]
    )
    after = set(
        select_calibration_points(
            frame, DEST, sectors=4, per_bin=4, time_bands=CALIBRATION_TIME_BANDS
        )["mesh_id"]
    )

    assert before <= after


def test_extra_points_spread_across_the_band() -> None:
    """Extra picks must not cluster at one end of the bin's time range."""
    picked = select_calibration_points(
        bin_frame(40), DEST, sectors=4, per_bin=4, time_bands=CALIBRATION_TIME_BANDS
    )
    spread = picked["freeflow_min"].astype(float)

    assert spread.max() - spread.min() > 10.0


def test_extra_points_stay_on_a_road() -> None:
    rows = [("near", 36.0, 139.76, 31.0, 5.0)]
    rows += [(f"far{i}", 36.0 + 0.001 * i, 139.76, 40.0 + i, 5000.0) for i in range(5)]

    picked = select_calibration_points(
        make_frame(rows), DEST, sectors=4, per_bin=4, time_bands=CALIBRATION_TIME_BANDS
    )

    assert picked["mesh_id"].tolist() == ["near"]


def test_a_thin_bin_yields_what_it_has() -> None:
    """Asking for more than the bin holds must not duplicate or borrow."""
    picked = select_calibration_points(
        bin_frame(2), DEST, sectors=4, per_bin=5, time_bands=CALIBRATION_TIME_BANDS
    )

    assert len(picked) == 2
    assert picked["mesh_id"].is_unique


def test_per_bin_selection_is_deterministic() -> None:
    frame = bin_frame(30)
    first = select_calibration_points(
        frame, DEST, sectors=4, per_bin=3, time_bands=CALIBRATION_TIME_BANDS
    )
    second = select_calibration_points(
        frame, DEST, sectors=4, per_bin=3, time_bands=CALIBRATION_TIME_BANDS
    )

    assert first["mesh_id"].tolist() == second["mesh_id"].tolist()
