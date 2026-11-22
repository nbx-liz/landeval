"""The ingestion table and the per-cell coverage table.

Each row of the ingestion table is run on a prefecture-split layer and on a
mesh-split layer (the filter case only on the first: no mesh-split layer filters). The
archives and the pin are synthetic; the pin is made the way the generator
makes it, then edited where a row needs it.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from hazard_pin_support import make_pin, pin_dict
from landeval.hazard import HazardDataError, PointIndex
from landeval.hazard_overlay import OverlaySpec
from landeval.hazard_sources import HazardPin, HazardSources, Layer

POINT = (139.0, 35.0)
AROUND_POINT = box(138.9, 34.9, 139.1, 35.1)
FAR_AWAY = box(130.0, 30.0, 130.1, 30.1)


def index() -> PointIndex:
    return PointIndex(np.array([POINT[0]]), np.array([POINT[1]]))


def zone(*geometries: object, code: int = 2, column: str = "A33_002") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {column: [code] * len(geometries)}, geometry=list(geometries), crs="EPSG:6668"
    )


def empty(column: str = "A33_002") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({column: []}, geometry=gpd.GeoSeries([], crs="EPSG:6668"))


def write_zip(path: Path, members: dict[str, gpd.GeoDataFrame | bytes]) -> Path:
    """An archive whose `.shp` members carry exactly the given names.

    A frame is written as a shapefile with its sidecars; bytes are written as
    the member itself, which makes an unreadable shapefile.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as writer:
        writer.writestr("readme.txt", "synthetic")
        for position, (name, content) in enumerate(members.items()):
            if isinstance(content, bytes):
                writer.writestr(name, content)
                continue
            work = path.parent / f"{path.stem}_{position}"
            work.mkdir()
            content.to_file(work / "m.shp")
            stem = name[: -len(".shp")]
            for part in work.iterdir():
                writer.write(part, stem + part.suffix)
    return path


PREF = OverlaySpec("landslide", "L", "landslide", ("Polygon",))
PREF_FILTERED = OverlaySpec(
    "landslide_special",
    "LS",
    "landslide",
    ("Polygon",),
    filter_column="A33_002",
    filter_values=frozenset({2, 4}),
    filter_domain=frozenset({1, 2, 3, 4}),
)
MESH = OverlaySpec("flood_max", "F", "flood", ("20_max",), "A31b_201", "rank_code")
COLLAPSE = OverlaySpec("flood_collapse", "C", "flood", ("41_c", "42_c"))


def pref_sources() -> HazardSources:
    return HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234",),
        prefectures={"01": "A", "02": "B"},
        layers={"landslide": Layer(label="l", split="prefecture", patterns=("l/A33_{pref}.zip",))},
    )


def mesh_sources() -> HazardSources:
    return HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234", "1235"),
        prefectures={"01": "A"},
        layers={"flood": Layer(label="f", split="primary_mesh", patterns=("f/f_{mesh}.zip",))},
    )


def run(spec: OverlaySpec, sources: HazardSources, raw: Path, pin: HazardPin | None = None) -> Any:
    from landeval.hazard_overlay import run_layer

    used = make_pin(sources, raw, [spec]) if pin is None else pin
    return run_layer(spec, sources, raw, index(), pin=used, log=lambda _: None)


# Each layout: the two archives of one split, with the member the row is about
# in the first and an ordinary member in the second.


def pref_archives(raw: Path, first: dict[str, gpd.GeoDataFrame | bytes]) -> None:
    write_zip(raw / "landslide" / "A33_01.zip", first)
    write_zip(raw / "landslide" / "A33_02.zip", {"A33_02Polygon.shp": zone(FAR_AWAY)})


def mesh_archives(raw: Path, first: dict[str, gpd.GeoDataFrame | bytes]) -> None:
    write_zip(raw / "flood" / "f_1234.zip", first)
    write_zip(
        raw / "flood" / "f_1235.zip",
        {"20_max/f_1235.shp": zone(FAR_AWAY, code=1, column="A31b_201")},
    )


SPLITS = {
    "prefecture": (PREF, pref_sources, pref_archives, "A33_01Polygon.shp", "A33_002", "01"),
    "primary_mesh": (MESH, mesh_sources, mesh_archives, "20_max/f_1234.shp", "A31b_201", "1234"),
}


def layout(split: str) -> tuple[OverlaySpec, HazardSources, Any, str, str, str]:
    spec, make_sources, write, member, column, unit = SPLITS[split]
    return spec, make_sources(), write, member, column, unit


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_a_missing_archive_stops(tmp_path: Path, split: str) -> None:
    spec, sources, write, member, column, _ = layout(split)
    write(tmp_path, {member: zone(AROUND_POINT, code=1, column=column)})
    pin = make_pin(sources, tmp_path, [spec])
    next(
        tmp_path.rglob("*_1234.zip") if split == "primary_mesh" else tmp_path.rglob("A33_01.zip")
    ).unlink()

    with pytest.raises(HazardDataError, match="not downloaded"):
        run(spec, sources, tmp_path, pin)


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_an_archive_that_differs_from_its_pin_stops(tmp_path: Path, split: str) -> None:
    spec, sources, write, member, column, _ = layout(split)
    write(tmp_path, {member: zone(AROUND_POINT, code=1, column=column)})
    data = pin_dict(sources, tmp_path, [spec])
    key = sorted(data["archives"])[0]
    data["archives"][key]["sha256"] = "f" * 64

    with pytest.raises(HazardDataError, match="sha256"):
        run(spec, sources, tmp_path, HazardPin.model_validate(data))


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_an_absent_entry_resolves_its_pair_without_reading(
    tmp_path: Path, split: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive holds no member for the fragment; the unit is covered, nothing is read."""
    spec, sources, write, _member, column, unit = layout(split)
    write(tmp_path, {"other/unrelated.shp": zone(AROUND_POINT, code=1, column=column)})
    pin = make_pin(sources, tmp_path, [spec])
    key = "landslide/A33_01.zip" if split == "prefecture" else "flood/f_1234.zip"
    assert pin.entry(spec.name, key, spec.member_contains[0]).absent is True

    result = run(spec, sources, tmp_path, pin)

    assert unit in result.covered
    assert result.severity is not None and np.isnan(result.severity).all()


@pytest.mark.parametrize("split", sorted(SPLITS))
@pytest.mark.parametrize("how", ["missing", "unreadable"])
def test_a_listed_member_that_cannot_be_read_stops(tmp_path: Path, split: str, how: str) -> None:
    spec, sources, write, member, column, _ = layout(split)
    if how == "missing":
        write(tmp_path, {member: zone(AROUND_POINT, code=1, column=column)})
        data = pin_dict(sources, tmp_path, [spec])
        key = "landslide/A33_01.zip" if split == "prefecture" else "flood/f_1234.zip"
        data["layers"][spec.name]["entries"][key][spec.member_contains[0]] = {
            "members": [member.replace(".shp", "_gone.shp")]
        }
        pin: HazardPin | None = HazardPin.model_validate(data)
        message = "lacks the pinned"
    else:
        write(tmp_path, {member: b"not a shapefile"})
        pin = None
        message = "cannot read pinned member"

    with pytest.raises(HazardDataError, match=message):
        run(spec, sources, tmp_path, pin)


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_a_null_or_empty_geometry_stops(
    tmp_path: Path, split: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, sources, write, member, column, _ = layout(split)
    write(tmp_path, {member: zone(AROUND_POINT, code=1, column=column)})
    pin = make_pin(sources, tmp_path, [spec])

    def chunks(_path: Path, _size: int) -> Iterator[gpd.GeoDataFrame]:
        yield zone(AROUND_POINT, None, code=1, column=column)

    monkeypatch.setattr("landeval.hazard_overlay._chunks", chunks)

    with pytest.raises(HazardDataError, match="null or empty"):
        run(spec, sources, tmp_path, pin)


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_a_value_outside_the_code_list_stops(tmp_path: Path, split: str) -> None:
    spec, sources, write, member, column, _ = layout(split)
    if split == "prefecture":
        spec = PREF_FILTERED
    write(tmp_path, {member: zone(AROUND_POINT, code=9, column=column)})

    with pytest.raises(HazardDataError, match="code"):
        run(spec, sources, tmp_path)


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_a_member_with_no_rows_leaves_its_unit_uncovered(tmp_path: Path, split: str) -> None:
    spec, sources, write, member, column, unit = layout(split)
    write(tmp_path, {member: empty(column)})

    result = run(spec, sources, tmp_path)

    assert result.available
    assert unit not in result.covered
    assert len(result.covered) == 1  # the other unit read its rows


def test_rows_the_filter_removes_in_full_still_cover_the_unit(tmp_path: Path) -> None:
    """Read and none kept is a legitimate empty, not a gap."""
    pref_archives(tmp_path, {"A33_01Polygon.shp": zone(AROUND_POINT, code=1)})

    result = run(PREF_FILTERED, pref_sources(), tmp_path)

    assert result.covered == frozenset({"01", "02"})
    assert result.severity is not None and np.isnan(result.severity).all()


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_an_ordinary_member_is_read_and_its_hits_are_kept(tmp_path: Path, split: str) -> None:
    spec, sources, write, member, column, unit = layout(split)
    write(tmp_path, {member: zone(AROUND_POINT, code=2, column=column)})

    result = run(spec, sources, tmp_path)

    assert unit in result.covered and len(result.covered) == 2
    assert result.severity is not None and np.isfinite(result.severity).all()


def test_an_unavailable_layer_opens_no_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The column for an unsplit layer: no archive is read and every cell is X."""
    from landeval.hazard_overlay import run_layer

    sources = HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234",),
        prefectures={"01": "A"},
        layers={"inland": Layer(label="i", unavailable="not ingested")},
    )

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an unavailable layer opened an archive")

    monkeypatch.setattr("landeval.hazard_overlay.zipfile.ZipFile", refuse)
    spec = OverlaySpec("inland", "I", "inland", ())
    pin = HazardPin.model_validate(
        {"schema": "hazard-archive-pin/v1", "archives": {}, "layers": {}}
    )

    result = run_layer(spec, sources, tmp_path, index(), pin=pin, log=lambda _: None)

    assert not result.available
    assert result.covered == frozenset()


# --- how the entries are collected ---------------------------------------------


def test_every_member_an_entry_lists_is_read_and_both_hits_count(tmp_path: Path) -> None:
    lon, lat = POINT
    second = PointIndex(np.array([lon, 130.05]), np.array([lat, 30.05]))
    pref_archives(
        tmp_path,
        {
            "a/A33_01Polygon.shp": zone(AROUND_POINT),
            "b/A33_01Polygon.shp": zone(FAR_AWAY),
        },
    )
    sources = pref_sources()
    pin = make_pin(sources, tmp_path, [PREF])
    assert len(pin.entry("landslide", "landslide/A33_01.zip", "Polygon").members or ()) == 2
    from landeval.hazard_overlay import run_layer

    result = run_layer(PREF, sources, tmp_path, second, pin=pin, log=lambda _: None)

    assert result.severity is not None and np.isfinite(result.severity).all()


def test_one_fragment_absent_and_the_other_listed_in_one_archive(tmp_path: Path) -> None:
    sources = mesh_sources()
    write_zip(tmp_path / "flood" / "f_1234.zip", {"42_c/x.shp": zone(AROUND_POINT, column="v")})
    write_zip(tmp_path / "flood" / "f_1235.zip", {"41_c/y.shp": zone(FAR_AWAY, column="v")})
    pin = make_pin(sources, tmp_path, [COLLAPSE])
    assert pin.entry("flood_collapse", "flood/f_1234.zip", "41_c").absent is True

    result = run(COLLAPSE, sources, tmp_path, pin)

    assert result.covered == frozenset({"1234", "1235"})
    assert result.severity is not None and np.isfinite(result.severity).all()


def test_a_layer_whose_every_entry_is_absent_reads_outside_and_does_not_stop(
    tmp_path: Path,
) -> None:
    sources = mesh_sources()
    for mesh in ("1234", "1235"):
        write_zip(tmp_path / "flood" / f"f_{mesh}.zip", {"30_other/x.shp": zone(AROUND_POINT)})

    result = run(COLLAPSE, sources, tmp_path)

    assert result.covered == frozenset({"1234", "1235"})
    assert result.severity is not None and np.isnan(result.severity).all()


# --- table U -----------------------------------------------------------------


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_hazard.py"


def build_script() -> ModuleType:
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_hazard_matrix_u", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_cell_states_decide_one_state_per_cell_in_order() -> None:
    """Unavailable, unknown prefecture, not designated (before covered), a
    prefecture with no expected archive, a primary mesh with none, covered,
    and a unit that is not covered."""
    module = build_script()
    by_pref = Layer(
        label="t",
        split="prefecture",
        patterns=("t/t_{pref}.zip",),
        not_designated={"03": "inland"},
    )
    pref = pd.Series([None, "03", "99", "01", "02"], dtype="string")
    # 03 is not designated; with a pattern it still has an expected archive, and
    # even a run that lists it as covered leaves it D (not designated before covered).
    states = module.cell_states(pref, by_pref, {"01", "03"})
    assert list(states) == ["U", "D", "U", "C", "U"]

    by_mesh = Layer(label="f", split="primary_mesh", patterns=("f/f_{mesh}.zip",))
    meshes = pd.Series(["1234", "9999", "1235"])
    states = module.cell_states(pd.Series([None] * 3), by_mesh, {"1234"}, primary_mesh=meshes)
    assert list(states) == ["C", "U", "U"]

    unavailable = Layer(label="i", unavailable="not ingested")
    assert list(module.cell_states(pref, unavailable, {"01"})) == ["X"] * 5


def test_one_empty_member_of_two_archives_leaves_the_prefecture_uncovered(
    tmp_path: Path,
) -> None:
    sources = HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234",),
        prefectures={"01": "A"},
        layers={
            "landslide": Layer(
                label="l", split="prefecture", patterns=("l/A_{pref}.zip", "l/B_{pref}.zip")
            )
        },
    )
    write_zip(tmp_path / "landslide" / "A_01.zip", {"A_01Polygon.shp": zone(AROUND_POINT)})
    write_zip(tmp_path / "landslide" / "B_01.zip", {"B_01Polygon.shp": empty()})

    result = run(PREF, sources, tmp_path)

    assert result.covered == frozenset()


def test_a_unit_with_no_listed_member_is_covered(tmp_path: Path) -> None:
    pref_archives(tmp_path, {"nothing/here.shp": zone(AROUND_POINT)})

    result = run(PREF, pref_sources(), tmp_path)

    assert "01" in result.covered
