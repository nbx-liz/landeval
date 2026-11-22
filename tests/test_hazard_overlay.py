"""Tests for the archive-walking side of the hazard overlay.

The parts that read real shapefiles are covered by the golden checks against
the produced artefact. What is worth isolating here are the decisions taken on
the way in -- which member of an archive belongs to which layer, what a missing
attribute means, and what happens when the data stops matching the spec --
because each of those has a failure mode that looks like clear ground.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from hazard_pin_support import make_pin
from landeval.hazard import HIT_WITHOUT_DEPTH, HazardDataError, PointIndex
from landeval.hazard_overlay import (
    LayerRun,
    OverlaySpec,
    _apply_filter,
    _severity_for,
    _shapefile_members,
    run_layer,
)
from landeval.hazard_sources import HazardSources, Layer

# The published A33_002 code list; the filtering specs below draw on it.
ZONE_CODES = frozenset({1, 2, 3, 4})


def frame(**columns: object) -> gpd.GeoDataFrame:
    length = len(next(iter(columns.values()))) if columns else 1  # type: ignore[arg-type]
    return gpd.GeoDataFrame(dict(columns), geometry=[box(0, 0, 1, 1)] * length, crs="EPSG:6668")


def archive_with(tmp_path: Path, names: list[str]) -> zipfile.ZipFile:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as writer:
        for name in names:
            writer.writestr(name, "")
    return zipfile.ZipFile(path)


def test_only_the_members_a_layer_names_are_selected(tmp_path: Path) -> None:
    spec = OverlaySpec("x", "X", "flood", ("20_想定最大規模",))
    archive = archive_with(
        tmp_path,
        [
            "10_計画規模\\a.shp",
            "20_想定最大規模\\b.shp",
            "30_浸水継続時間\\c.shp",
            "20_想定最大規模\\b.dbf",
        ],
    )

    assert _shapefile_members(archive, spec) == ["20_想定最大規模\\b.shp"]


def test_a_layer_can_draw_on_several_members(tmp_path: Path) -> None:
    """家屋倒壊等氾濫想定区域 arrives as two files, 氾濫流 and 河岸侵食."""
    spec = OverlaySpec("x", "X", "flood", ("41_", "42_"))
    archive = archive_with(tmp_path, ["41_a.shp", "42_b.shp", "20_c.shp"])

    assert sorted(_shapefile_members(archive, spec)) == ["41_a.shp", "42_b.shp"]


def test_a_depthless_layer_marks_every_polygon_as_hazard() -> None:
    spec = OverlaySpec("x", "X", "landslide", ("Polygon",))

    severity, missing = _severity_for(frame(A33_002=[1, 2, 1]), spec)

    assert (severity == HIT_WITHOUT_DEPTH).all()
    assert missing == 0


def test_rank_codes_become_depths() -> None:
    spec = OverlaySpec("x", "X", "flood", ("20_",), "A31b_201", "rank_code")

    severity, missing = _severity_for(frame(A31b_201=[1, 3, 6]), spec)

    assert list(severity) == [0.0, 3.0, 20.0]
    assert missing == 0


def test_a_missing_depth_is_still_hazard_and_is_counted() -> None:
    """Inside a designated area without a published depth is not outside it."""
    spec = OverlaySpec("x", "X", "tsunami", (".shp",), "A40_003", "range_text")

    severity, missing = _severity_for(frame(A40_003=["1m以上3m未満", None]), spec)

    assert severity[0] == 1.0
    assert severity[1] == HIT_WITHOUT_DEPTH
    assert missing == 1


def test_an_unreadable_depth_stops_the_run() -> None:
    """A changed vocabulary must not be absorbed as 'no water here'."""
    spec = OverlaySpec("x", "X", "tsunami", (".shp",), "A40_003", "range_text")

    with pytest.raises(HazardDataError, match="depth"):
        _severity_for(frame(A40_003=["1m以上3m未満", "しばしば浸水"]), spec)


def test_a_renamed_depth_column_stops_the_run() -> None:
    spec = OverlaySpec("x", "X", "flood", ("20_",), "A31b_201", "rank_code")

    with pytest.raises(HazardDataError, match="A31b_201"):
        _severity_for(frame(A31b_999=[1]), spec)


def test_the_special_zone_filter_keeps_only_red_zones() -> None:
    spec = OverlaySpec(
        "x",
        "X",
        "landslide",
        ("Polygon",),
        filter_column="A33_002",
        filter_values=frozenset({2, 4}),
        filter_domain=ZONE_CODES,
    )

    kept = _apply_filter(frame(A33_002=[1, 2, 3, 4]), spec)

    assert list(kept["A33_002"]) == [2, 4]


def test_codes_arriving_as_strings_still_filter() -> None:
    spec = OverlaySpec(
        "x",
        "X",
        "landslide",
        ("Polygon",),
        filter_column="A33_002",
        filter_values=frozenset({2}),
        filter_domain=ZONE_CODES,
    )

    kept = _apply_filter(frame(A33_002=["1", "2"]), spec)

    assert len(kept) == 1


def test_a_non_code_in_the_filter_column_stops_the_run() -> None:
    """Silently dropping unparseable rows would shrink the hazard area."""
    spec = OverlaySpec(
        "x",
        "X",
        "landslide",
        ("Polygon",),
        filter_column="A33_002",
        filter_values=frozenset({2}),
        filter_domain=ZONE_CODES,
    )

    with pytest.raises(HazardDataError, match="not codes"):
        _apply_filter(frame(A33_002=["2", "特別"]), spec)


def test_a_missing_filter_column_stops_the_run() -> None:
    spec = OverlaySpec(
        "x",
        "X",
        "landslide",
        ("Polygon",),
        filter_column="A33_002",
        filter_values=frozenset({2}),
        filter_domain=ZONE_CODES,
    )

    with pytest.raises(HazardDataError, match="A33_002"):
        _apply_filter(frame(other=[1]), spec)


def test_no_filter_passes_the_frame_through() -> None:
    spec = OverlaySpec("x", "X", "landslide", ("Polygon",))
    original = frame(A33_002=[1, 2])

    assert len(_apply_filter(original, spec)) == 2


def index() -> PointIndex:
    return PointIndex(np.array([139.0]), np.array([35.0]))


def special_zone_spec() -> OverlaySpec:
    return OverlaySpec(
        "x",
        "X",
        "landslide",
        ("Polygon",),
        filter_column="A33_002",
        filter_values=frozenset({2, 4}),
        filter_domain=ZONE_CODES,
    )


def test_an_unknown_numeric_zone_code_stops_the_run() -> None:
    """A code outside the published list is a changed vocabulary, not 'not red'."""
    with pytest.raises(HazardDataError, match="code list"):
        _apply_filter(frame(A33_002=[1, 2, 7]), special_zone_spec())


def test_a_fractional_zone_code_stops_the_run() -> None:
    """2.5 must not be truncated into the red-zone code 2."""
    with pytest.raises(HazardDataError, match="code list"):
        _apply_filter(frame(A33_002=[2.0, 2.5]), special_zone_spec())


def test_a_filtering_spec_without_a_code_list_is_refused() -> None:
    """Without the full list, an unknown code could not be told from an unkept one."""
    with pytest.raises(ValueError, match="code list"):
        OverlaySpec(
            "x",
            "X",
            "landslide",
            ("Polygon",),
            filter_column="A33_002",
            filter_values=frozenset({2}),
        )


def test_whole_number_codes_stored_as_floats_still_filter() -> None:
    kept = _apply_filter(frame(A33_002=[1.0, 2.0, 3.0, 4.0]), special_zone_spec())

    assert list(kept["A33_002"]) == [2.0, 4.0]


def mesh_sources() -> HazardSources:
    return HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234",),
        prefectures={"01": "A"},
        layers={"flood": Layer(label="flood", split="primary_mesh", patterns=("f/f_{mesh}.zip",))},
    )


def test_a_layer_whose_archives_hold_no_matching_member_reads_outside(tmp_path: Path) -> None:
    """The pin says the published archive holds no area for the fragment.

    That is a statement about the data, not a gap in reading it, so the layer
    comes back available and its unit is covered. Without the pin nothing could
    tell "holds no such area" from "the member selection does not match", and
    the run would have to stop.
    """
    sources = mesh_sources()
    (tmp_path / "flood").mkdir()
    with zipfile.ZipFile(tmp_path / "flood" / "f_1234.zip", "w") as writer:
        writer.writestr("30_other/c.shp", "")
    spec = OverlaySpec("x", "X", "flood", ("20_想定最大規模",))
    pin = make_pin(sources, tmp_path, [spec])

    run = run_layer(spec, sources, tmp_path, index(), pin=pin, log=lambda _: None)

    assert run.available
    assert run.covered == frozenset({"1234"})
    assert run.severity is not None and np.isnan(run.severity).all()


def prefecture_sources() -> HazardSources:
    return HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234",),
        prefectures={"01": "A", "02": "B"},
        layers={
            "landslide": Layer(
                label="landslide", split="prefecture", patterns=("l/A33_{pref}.zip",)
            )
        },
    )


def write_archive(path: Path, polygons: gpd.GeoDataFrame | None) -> None:
    """A zipped shapefile named like a landslide member, or an archive with none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as writer:
        if polygons is None:
            writer.writestr("readme.txt", "no shapefile here")
            return
        work = path.parent / f"{path.stem}_shp"
        work.mkdir()
        polygons.to_file(work / "A33_Polygon.shp")
        for part in work.iterdir():
            writer.write(part, part.name)


def red_zone(*geometries: object) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"A33_002": [2] * len(geometries)}, geometry=list(geometries), crs="EPSG:6668"
    )


SPEC = OverlaySpec("landslide", "L", "landslide", ("Polygon",))
AROUND_POINT = box(138.9, 34.9, 139.1, 35.1)


def run_prefectures(tmp_path: Path) -> LayerRun:
    sources = prefecture_sources()
    pin = make_pin(sources, tmp_path, [SPEC])
    return run_layer(SPEC, sources, tmp_path, index(), pin=pin, log=lambda _: None)


def test_a_prefecture_whose_archive_the_pin_declares_empty_is_covered(tmp_path: Path) -> None:
    """Prefecture 02's archive holds no landslide member, and the pin says so.

    The pin was made from the same bytes the build reads, so "no member" is
    what the published archive holds, and the prefecture is covered: its cells
    read outside. A member that is listed and then holds no rows is the case
    that leaves a prefecture uncovered (see test_hazard_matrix_overlay.py).
    """
    write_archive(tmp_path / "landslide" / "A33_01.zip", red_zone(AROUND_POINT))
    write_archive(tmp_path / "landslide" / "A33_02.zip", None)

    run = run_prefectures(tmp_path)

    assert run.available
    assert run.covered == frozenset({"01", "02"})


def test_every_prefecture_that_delivered_features_is_covered(tmp_path: Path) -> None:
    far_away = box(130.0, 30.0, 130.1, 30.1)
    write_archive(tmp_path / "landslide" / "A33_01.zip", red_zone(AROUND_POINT))
    write_archive(tmp_path / "landslide" / "A33_02.zip", red_zone(far_away))

    run = run_prefectures(tmp_path)

    assert run.covered == frozenset({"01", "02"})


def test_coverage_is_empty_unless_a_run_reports_it() -> None:
    """A run that never says what it read must not cover anything."""
    assert LayerRun(SPEC, np.array([np.nan])).covered == frozenset()


def test_a_null_geometry_in_a_shapefile_stops_the_run(tmp_path: Path) -> None:
    write_archive(tmp_path / "landslide" / "A33_01.zip", red_zone(AROUND_POINT, None))
    write_archive(tmp_path / "landslide" / "A33_02.zip", red_zone(AROUND_POINT))

    with pytest.raises(HazardDataError, match="null or empty"):
        run_prefectures(tmp_path)


@pytest.mark.parametrize(
    "geometries",
    [[None], [Polygon()], [Polygon(), None], [AROUND_POINT, Polygon()]],
    ids=["null", "empty", "null-and-empty", "valid-and-empty"],
)
def test_null_or_empty_geometries_never_yield_an_available_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, geometries: list[object]
) -> None:
    """Rows that cannot be placed are not polygons read, and the layer must not come back."""
    write_archive(tmp_path / "landslide" / "A33_01.zip", red_zone(AROUND_POINT))
    write_archive(tmp_path / "landslide" / "A33_02.zip", red_zone(AROUND_POINT))

    def chunks(_path: Path, _size: int) -> Iterator[gpd.GeoDataFrame]:
        yield red_zone(*geometries)

    monkeypatch.setattr("landeval.hazard_overlay._chunks", chunks)

    with pytest.raises(HazardDataError, match="null or empty"):
        run_prefectures(tmp_path)
