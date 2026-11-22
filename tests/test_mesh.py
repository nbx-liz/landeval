"""Tests for the JIS X 0410 standard grid square (地域メッシュ) helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from landeval.mesh import (
    _SUBDIVISIONS,
    AREA_UNIT_KM2,
    SUPPORTED_250M_ROWS,
    MeshError,
    _decode,
    _encode,
    _raw_row_area_km2,
    area_km2,
    area_units_of_row,
    cell_area_km2,
    mesh_bounds,
    mesh_center,
    mesh_code,
    mesh_size_of,
    meshes_in_bbox,
)

# A station-sized point. 53354611 is the canonical 1 km mesh for it.
STATION_LAT, STATION_LON = 35.6812, 135.7671


@pytest.mark.parametrize(
    ("size_m", "expected"),
    # 500 m cell 533546113 spans lat 35.679167-35.683333, lon 135.762500-135.768750.
    # The point sits in its south-east quarter, hence the trailing 2.
    [(1000, "53354611"), (500, "533546113"), (250, "5335461132")],
)
def test_known_codes_for_the_station_point(size_m: int, expected: str) -> None:
    assert mesh_code(STATION_LAT, STATION_LON, size_m) == expected


@pytest.mark.parametrize("size_m", [1000, 500, 250])
def test_code_length_matches_level(size_m: int) -> None:
    assert len(mesh_code(STATION_LAT, STATION_LON, size_m)) == {1000: 8, 500: 9, 250: 10}[size_m]


@pytest.mark.parametrize("size_m", [1000, 500, 250])
def test_center_round_trips_to_the_same_code(size_m: int) -> None:
    code = mesh_code(STATION_LAT, STATION_LON, size_m)
    lat, lon = mesh_center(code)

    assert mesh_code(lat, lon, size_m) == code


@pytest.mark.parametrize("size_m", [1000, 500, 250])
def test_bounds_contain_the_originating_point(size_m: int) -> None:
    south, west, north, east = mesh_bounds(mesh_code(STATION_LAT, STATION_LON, size_m))

    assert south <= STATION_LAT < north
    assert west <= STATION_LON < east


@pytest.mark.parametrize(
    ("size_m", "lat_deg", "lon_deg"),
    [
        (1000, 1 / 120, 1 / 80),
        (500, 1 / 240, 1 / 160),
        (250, 1 / 480, 1 / 320),
    ],
)
def test_cell_dimensions(size_m: int, lat_deg: float, lon_deg: float) -> None:
    south, west, north, east = mesh_bounds(mesh_code(STATION_LAT, STATION_LON, size_m))

    assert math.isclose(north - south, lat_deg, rel_tol=1e-9)
    assert math.isclose(east - west, lon_deg, rel_tol=1e-9)


def test_center_is_the_midpoint_of_the_bounds() -> None:
    code = mesh_code(STATION_LAT, STATION_LON, 500)
    south, west, north, east = mesh_bounds(code)
    lat, lon = mesh_center(code)

    assert math.isclose(lat, (south + north) / 2, rel_tol=1e-12)
    assert math.isclose(lon, (west + east) / 2, rel_tol=1e-12)


def test_mesh_size_is_recoverable_from_the_code() -> None:
    for size_m in (1000, 500, 250):
        assert mesh_size_of(mesh_code(STATION_LAT, STATION_LON, size_m)) == size_m


def test_adjacent_points_fall_in_adjacent_cells() -> None:
    """A step of one full cell width must change the code."""
    code = mesh_code(STATION_LAT, STATION_LON, 500)
    east_code = mesh_code(STATION_LAT, STATION_LON + 1 / 160, 500)
    north_code = mesh_code(STATION_LAT + 1 / 240, STATION_LON, 500)

    assert code != east_code
    assert code != north_code
    assert east_code != north_code


# --- bbox enumeration ------------------------------------------------------


def test_bbox_enumeration_covers_a_single_cell() -> None:
    code = mesh_code(STATION_LAT, STATION_LON, 500)
    south, west, north, east = mesh_bounds(code)

    codes = list(meshes_in_bbox(south, west, north, east, 500))

    assert codes == [code]


def test_bbox_enumeration_is_complete_and_unique() -> None:
    south, west = 35.60, 139.70
    north, east = south + 10 / 240, west + 7 / 160  # 10 rows x 7 columns

    codes = list(meshes_in_bbox(south, west, north, east, 500))

    assert len(codes) == 70
    assert len(set(codes)) == 70


def test_every_enumerated_cell_intersects_the_bbox() -> None:
    south, west, north, east = 35.60, 139.70, 35.63, 139.75

    for code in meshes_in_bbox(south, west, north, east, 500):
        cs, cw, cn, ce = mesh_bounds(code)
        assert cn > south and cs < north
        assert ce > west and cw < east


def test_bbox_enumeration_covers_points_inside_the_bbox() -> None:
    south, west, north, east = 35.60, 139.70, 35.63, 139.75
    codes = set(meshes_in_bbox(south, west, north, east, 500))

    for lat in (35.601, 35.615, 35.629):
        for lon in (139.701, 139.725, 139.749):
            assert mesh_code(lat, lon, 500) in codes


# --- rejection matrix ------------------------------------------------


@pytest.mark.parametrize("size_m", [0, 100, 300, 750, 2000, -500])
def test_unsupported_mesh_size_is_rejected(size_m: int) -> None:
    with pytest.raises(MeshError, match="mesh size"):
        mesh_code(STATION_LAT, STATION_LON, size_m)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(0.0, 0.0), (35.0, 0.0), (0.0, 139.0), (-35.6, 139.7), (139.7, 35.6)],
)
def test_coordinates_outside_japan_are_rejected(lat: float, lon: float) -> None:
    with pytest.raises(MeshError, match="Japan"):
        mesh_code(lat, lon, 500)


@pytest.mark.parametrize("code", ["", "533", "5335461", "533546113X", "53354611311", "abcdefgh"])
def test_malformed_code_is_rejected(code: str) -> None:
    with pytest.raises(MeshError):
        mesh_bounds(code)


@pytest.mark.parametrize("code", ["533546115", "533546110", "5335461135"])
def test_out_of_range_subdivision_digit_is_rejected(code: str) -> None:
    """Half/quarter digits are 1-4; anything else must fail loudly."""
    with pytest.raises(MeshError, match="subdivision"):
        mesh_bounds(code)


def test_inverted_bbox_is_rejected() -> None:
    with pytest.raises(MeshError, match="bbox"):
        list(meshes_in_bbox(35.63, 139.70, 35.60, 139.75, 500))


# --- cell area --------------------------------------------------------------
#
# Multiplying a cell count by 0.25 reads a "500 m mesh" as 500 m square. It is
# not: JIS X 0410 defines the half-regional mesh in *arc seconds*, 15" of
# latitude by 22.5" of longitude, which at this latitude is roughly 463 m by 565 m,
# so a total computed that way is several percent low.


def test_a_500m_cell_is_not_a_500m_square() -> None:
    """A cell is not a square: pinned so a tidy-looking constant cannot stand in for
    the computed area."""
    area = cell_area_km2("533546113")

    assert not math.isclose(area, 0.25, rel_tol=0.01)
    assert math.isclose(area, 0.2615, rel_tol=0.001)


def test_cells_shrink_towards_the_pole() -> None:
    """A degree of longitude is shorter further north, so a northern cell is a
    smaller piece of ground. A single constant cannot express that, which is why
    the area is read from the cell rather than assumed."""
    south = cell_area_km2(mesh_code(33.0, 139.0, 500))
    north = cell_area_km2(mesh_code(38.0, 139.0, 500))

    assert north < south


def test_a_kilometre_cell_is_its_four_children() -> None:
    """An independent check on the formula. The four 500 m cells inside a 1 km
    cell tile it exactly, so their areas must sum to its own -- an error in the
    ellipsoid maths would not survive this, whereas comparing the function to
    itself would."""
    parent = mesh_code(35.68, 139.76, 1000)
    min_lat, min_lon, max_lat, max_lon = mesh_bounds(parent)
    children = {
        mesh_code(lat, lon, 500)
        for lat in (min_lat + (max_lat - min_lat) * f for f in (0.25, 0.75))
        for lon in (min_lon + (max_lon - min_lon) * f for f in (0.25, 0.75))
    }

    assert len(children) == 4
    assert sum(cell_area_km2(code) for code in children) == cell_area_km2(parent)


def test_total_area_sums_the_cells_it_is_given() -> None:
    codes = [mesh_code(35.68, 139.76, 500), mesh_code(35.70, 139.76, 500)]

    assert area_km2(codes) == sum(cell_area_km2(code) for code in codes)


# --- the same area on every machine --------------------------------
#
# pyproj's geodesic area differs in its last bit between machines (aarch64
# libproj contracts to FMA), and that bit is enough to turn a round sum of
# cells into a value such as 0.7999999999999999. The 250 m rows are therefore rounded to
# whole units of AREA_UNIT_KM2 and every coarser cell is their exact sum. The
# tests below pin what that design promises.

Q250_FIXTURE = Path(__file__).parent / "fixtures" / "mesh" / "q250_units.json"

# How close to a rounding midpoint a raw area may come, in ulps of the raw km2
# value, before the suite calls it a risk. Machines differ by a few ulps, so
# this leaves a wide margin above that difference.
MIN_MIDPOINT_MARGIN_ULPS = 64


def test_every_supported_row_has_the_recorded_units() -> None:
    """The guarantee itself. Two machines can each be far from a midpoint and
    still land on opposite sides of it, so a margin alone proves nothing across
    machines -- the integers have to equal values recorded once. A failure
    after a pyproj/PROJ upgrade is a real change of area: record it
    and rerun the real-data comparison before regenerating the fixture."""
    recorded = json.loads(Q250_FIXTURE.read_text(encoding="utf-8"))

    assert recorded["first_row"] == SUPPORTED_250M_ROWS.start
    assert len(recorded["units"]) == len(SUPPORTED_250M_ROWS)
    differing = [
        (row, expected, area_units_of_row(row))
        for row, expected in zip(SUPPORTED_250M_ROWS, recorded["units"], strict=True)
        if area_units_of_row(row) != expected
    ]
    assert not differing, f"{len(differing)} rows differ (row, recorded, now): {differing[:5]}"


def test_every_row_is_far_from_a_rounding_midpoint() -> None:
    """An early warning, not the guarantee: a shrinking margin says the next
    library or machine may round some row the other way."""
    margins = []
    for row in SUPPORTED_250M_ROWS:
        raw = _raw_row_area_km2(row)
        scaled = raw / AREA_UNIT_KM2
        distance = abs(scaled - math.floor(scaled) - 0.5) * AREA_UNIT_KM2
        margins.append(distance / math.ulp(raw))

    assert min(margins) >= MIN_MIDPOINT_MARGIN_ULPS


def _children(code: str, size_m: int) -> list[str]:
    """The cells of ``size_m`` that tile ``code``.

    Built from grid indices rather than from centre coordinates: the
    northernmost supported row lies north of 46.5, where `mesh_code` refuses a
    point but the row is still one a 46.5 cell opens."""
    row, col, subdivisions = _decode(code)
    finer = _SUBDIVISIONS[size_m]
    k = finer // subdivisions
    return [_encode(row * k + i, col * k + j, finer) for i in range(k) for j in range(k)]


def test_every_cell_is_exactly_its_children() -> None:
    """At every supported row and both levels. Checking a 1 km cell against its
    500 m and its 250 m children alone would let two 500 m rows err in opposite
    directions and cancel. A cell's area depends only on its row, so one column
    per row is the whole grid."""
    first_km_row = SUPPORTED_250M_ROWS.start // 4
    last_km_row = SUPPORTED_250M_ROWS.stop // 4
    tokyo_column = _decode(mesh_code(STATION_LAT, STATION_LON, 1000))[1]
    for km_row in range(first_km_row, last_km_row):
        parent = _encode(km_row, tokyo_column, _SUBDIVISIONS[1000])
        halves = _children(parent, 500)
        quarters = [_children(half, 250) for half in halves]

        assert len(halves) == 4
        for half, its_quarters in zip(halves, quarters, strict=True):
            assert cell_area_km2(half) == sum(cell_area_km2(code) for code in its_quarters)
        assert cell_area_km2(parent) == sum(cell_area_km2(code) for code in halves)
        assert cell_area_km2(parent) == sum(cell_area_km2(c) for q in quarters for c in q)


@pytest.mark.parametrize(
    "code",
    # 1 km rows 2400..5580 touch JAPAN_LAT_RANGE (46.5 inclusive); these are
    # the rows just outside, at every level: 1 km 2399 / 5581, 500 m 4799 /
    # 11162, 250 m 9599 / 22324.
    ["29397090", "69396010", "293970903", "693960101", "2939709033", "6939601011"],
)
def test_a_code_outside_the_supported_rows_has_no_area(code: str) -> None:
    """The recorded units cover only the supported rows, so an area outside
    them would be one no test has pinned."""
    with pytest.raises(MeshError):
        cell_area_km2(code)


@pytest.mark.parametrize(
    "code",
    # The southernmost and northernmost supported rows: 1 km 2400 / 5580,
    # 250 m 9600 / 22323. The north edge is the 1 km row 46.5 itself opens.
    ["30390000", "69396000", "3039000011", "6939600033"],
)
def test_the_edge_rows_are_supported(code: str) -> None:
    assert cell_area_km2(code) > 0


def test_the_total_of_no_cells_is_zero() -> None:
    """Zero area is an answer. Returning it for an empty selection keeps "no
    cells qualified" distinguishable from a crash, which is what a caller
    printing a total needs."""
    assert area_km2([]) == 0.0


def test_a_malformed_code_has_no_area() -> None:
    """Not a zero. A code the grid cannot place is a broken input, and an area
    of 0 for it would quietly shrink whatever total it lands in."""
    with pytest.raises(MeshError):
        cell_area_km2("533546")
