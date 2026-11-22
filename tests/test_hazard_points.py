"""Reading the 100 m sample point nearest one address.

The cell-level reading answers "of the 25 points in this 500 m square, how many
fall inside a designated zone". That is the right question for area screening
and the wrong one for a plot: a cell at 4% and a cell at 96% both print
「区域内」, and the plot asked about may be at either end of it.

What this layer must not do is overclaim in the other direction. The sample
points sit on a 100 m lattice, so the nearest one is up to about 70 m away and
is not the parcel. The distance travels with the reading for that reason, and a
layer that was never obtained is named as unknown rather than read off a
membership bit that could only ever have been zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from landeval.hazard_points import HazardPointError, LayerOverlay, SamplePoints
from landeval.mesh import mesh_center, mesh_code, mesh_step_deg

MESH = mesh_code(43.4, 142.5, 500)
LAT, LON = mesh_center(MESH)
LAYERS = ("flood_max", "landslide_warning", "inland_flood")
AVAILABLE = ("flood_max", "landslide_warning")
SUBGRID = 5


def write_npz(path: Path, membership: dict[int, int]) -> Path:
    """One cell of 25 points, with ``membership`` giving the set bits by index."""
    mask = np.zeros(SUBGRID * SUBGRID, dtype=np.uint16)
    for index, bits in membership.items():
        mask[index] = bits
    np.savez_compressed(
        path,
        membership=mask,
        cell_index=np.zeros(SUBGRID * SUBGRID, dtype=np.int64),
        mesh_id=np.array([MESH], dtype=str),
        coverage=np.array([0b011], dtype=np.uint16),
        layers=np.array(LAYERS),
        available=np.array(AVAILABLE),
        points_per_cell=np.int64(SUBGRID * SUBGRID),
    )
    return path


@pytest.fixture
def points(tmp_path: Path) -> SamplePoints:
    # Point 12 is the centre of a 5x5 lattice; bit 0 is flood_max.
    return SamplePoints.load(write_npz(tmp_path / "p.npz", {12: 0b001}))


def test_the_reading_is_the_nearest_lattice_point(points: SamplePoints) -> None:
    """The cell centre coincides with the middle sample point, so a reading
    taken there is that point's."""
    reading = points.reading_at(MESH, LAT, LON)

    assert reading.inside == ("flood_max",)
    assert reading.distance_m == pytest.approx(0.0, abs=1.0)


def test_a_point_next_to_the_marked_one_reads_clear(points: SamplePoints) -> None:
    """100 m away is a different sample point, and the difference between the
    two is the whole reason for reporting the point rather than the cell."""
    lat_step, _ = mesh_step_deg(500)

    reading = points.reading_at(MESH, LAT + lat_step / SUBGRID, LON)

    assert reading.inside == ()
    assert reading.distance_m == pytest.approx(0.0, abs=1.0)


def test_the_distance_to_the_lattice_point_travels_with_the_reading(
    points: SamplePoints,
) -> None:
    """A lattice point is not a parcel boundary. How far the reading was taken
    from the address is part of the reading."""
    lat_step, _ = mesh_step_deg(500)

    reading = points.reading_at(MESH, LAT + lat_step / (2 * SUBGRID), LON)

    assert 40.0 < reading.distance_m < 60.0


def test_a_layer_that_was_never_obtained_is_unknown_not_clear(points: SamplePoints) -> None:
    """Its membership bit is zero because nothing was ever read into it. Zero
    means "no polygon covered this point" only for a layer that has polygons."""
    reading = points.reading_at(MESH, LAT, LON)

    assert reading.unknown == ("inland_flood",)
    assert "inland_flood" not in reading.inside


def test_two_layers_at_one_point_are_both_reported(tmp_path: Path) -> None:
    loaded = SamplePoints.load(write_npz(tmp_path / "two.npz", {12: 0b011}))

    reading = loaded.reading_at(MESH, LAT, LON)

    assert reading.inside == ("flood_max", "landslide_warning")


def test_a_cell_the_index_does_not_hold_is_an_error(points: SamplePoints) -> None:
    """Answering with the nearest indexed cell would read one square's ground
    as another's."""
    with pytest.raises(HazardPointError, match="53350"):
        points.reading_at("533500001", LAT, LON)


def test_a_coordinate_outside_the_named_cell_is_an_error(points: SamplePoints) -> None:
    """The caller says which cell it is asking about. A coordinate that is not
    in it means the two disagree, and picking either would answer a question
    nobody asked."""
    with pytest.raises(HazardPointError, match="does not contain"):
        points.reading_at(MESH, 43.06, 141.35)


def test_an_index_whose_point_count_disagrees_with_its_cells_is_an_error(
    tmp_path: Path,
) -> None:
    """A truncated or half-written index would otherwise read the wrong cell's
    points for every cell after the gap."""
    path = tmp_path / "short.npz"
    np.savez_compressed(
        path,
        membership=np.zeros(24, dtype=np.uint16),
        cell_index=np.zeros(24, dtype=np.int64),
        mesh_id=np.array([MESH], dtype=str),
        coverage=np.array([0b011], dtype=np.uint16),
        layers=np.array(LAYERS),
        available=np.array(AVAILABLE),
        points_per_cell=np.int64(25),
    )

    with pytest.raises(HazardPointError, match="24"):
        SamplePoints.load(path)


def test_an_available_layer_the_index_does_not_list_is_an_error(tmp_path: Path) -> None:
    """`available` names which bits mean something. A name that is not among
    the bits means the two were written by different runs."""
    path = tmp_path / "mismatch.npz"
    np.savez_compressed(
        path,
        membership=np.zeros(25, dtype=np.uint16),
        cell_index=np.zeros(25, dtype=np.int64),
        mesh_id=np.array([MESH], dtype=str),
        coverage=np.array([0b011], dtype=np.uint16),
        layers=np.array(LAYERS),
        available=np.array(["tsunami"]),
        points_per_cell=np.int64(25),
    )

    with pytest.raises(HazardPointError, match="tsunami"):
        SamplePoints.load(path)


def test_a_clear_point_says_so_without_saying_safe(points: SamplePoints) -> None:
    """`is_clear` is about the obtained layers only. The unobtained ones stay
    in `unknown`, which is why it is not a claim about the ground."""
    lat_step, _ = mesh_step_deg(500)

    reading = points.reading_at(MESH, LAT + lat_step / SUBGRID, LON)

    assert reading.is_clear
    assert reading.unknown == ("inland_flood",)


def test_an_index_missing_an_array_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    np.savez_compressed(path, membership=np.zeros(25, dtype=np.uint16))

    with pytest.raises(HazardPointError, match="mesh_id"):
        SamplePoints.load(path)


def test_a_lattice_that_is_not_square_is_an_error(tmp_path: Path) -> None:
    """The membership index is read as rows and columns, and a count with no
    square root has no rows and columns to read it as."""
    path = tmp_path / "oblong.npz"
    np.savez_compressed(
        path,
        membership=np.zeros(24, dtype=np.uint16),
        cell_index=np.zeros(24, dtype=np.int64),
        mesh_id=np.array([MESH], dtype=str),
        coverage=np.array([0b011], dtype=np.uint16),
        layers=np.array(LAYERS),
        available=np.array(AVAILABLE),
        points_per_cell=np.int64(24),
    )
    loaded = SamplePoints.load(path)

    with pytest.raises(HazardPointError, match="square"):
        loaded.reading_at(MESH, LAT, LON)


def test_a_cell_listed_twice_in_the_index_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "doubled.npz"
    np.savez_compressed(
        path,
        membership=np.zeros(50, dtype=np.uint16),
        cell_index=np.repeat([0, 1], 25),
        mesh_id=np.array([MESH, MESH], dtype=str),
        coverage=np.array([0b011, 0b011], dtype=np.uint16),
        layers=np.array(LAYERS),
        available=np.array(AVAILABLE),
        points_per_cell=np.int64(25),
    )
    loaded = SamplePoints.load(path)

    with pytest.raises(HazardPointError, match="appears 2 times"):
        loaded.reading_at(MESH, LAT, LON)


# --- a layer published per municipality -----------------


def write_overlay(path: Path, mesh_ids: list[str], membership: list[int]) -> Path:
    np.savez_compressed(
        path,
        layer=np.array("inland_flood"),
        mesh_id=np.array(mesh_ids, dtype=str),
        membership=np.array(membership, dtype=np.uint8),
        points_per_cell=np.int64(SUBGRID * SUBGRID),
    )
    return path


def test_an_overlay_turns_an_unknown_layer_into_a_reading(
    points: SamplePoints, tmp_path: Path
) -> None:
    """Inside a publishing municipality the layer is known, and saying 未取得
    there would be as wrong as saying 区域外 outside one."""
    marked = [0] * 25
    marked[12] = 1
    overlay = LayerOverlay.load(write_overlay(tmp_path / "o.npz", [MESH], marked))

    reading = points.reading_at(MESH, LAT, LON, [overlay])

    assert "inland_flood" in reading.inside
    assert reading.unknown == ()


def test_an_overlay_that_covers_the_cell_can_also_say_outside(
    points: SamplePoints, tmp_path: Path
) -> None:
    overlay = LayerOverlay.load(write_overlay(tmp_path / "o.npz", [MESH], [0] * 25))

    reading = points.reading_at(MESH, LAT, LON, [overlay])

    assert "inland_flood" not in reading.inside
    assert reading.unknown == ()


def test_a_cell_the_overlay_does_not_cover_stays_unknown(
    points: SamplePoints, tmp_path: Path
) -> None:
    """The municipality does not publish. Folding this layer into the main
    index would make exactly these cells read as 区域外."""
    overlay = LayerOverlay.load(write_overlay(tmp_path / "o.npz", ["999999999"], [0] * 25))

    reading = points.reading_at(MESH, LAT, LON, [overlay])

    assert reading.unknown == ("inland_flood",)


def test_an_overlay_whose_point_count_disagrees_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "short.npz"
    np.savez_compressed(
        path,
        layer=np.array("inland_flood"),
        mesh_id=np.array([MESH], dtype=str),
        coverage=np.array([0b011], dtype=np.uint16),
        membership=np.zeros(24, dtype=np.uint8),
        points_per_cell=np.int64(25),
    )

    with pytest.raises(HazardPointError, match="24"):
        LayerOverlay.load(path)


def test_an_overlay_missing_an_array_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "partial.npz"
    np.savez_compressed(path, layer=np.array("inland_flood"))

    with pytest.raises(HazardPointError, match="mesh_id"):
        LayerOverlay.load(path)


def test_an_overlay_listing_a_cell_twice_is_an_error(tmp_path: Path) -> None:
    overlay = LayerOverlay.load(write_overlay(tmp_path / "o.npz", [MESH, MESH], [0] * 50))

    with pytest.raises(HazardPointError, match="appears 2 times"):
        overlay.reading_for(MESH, 0)
