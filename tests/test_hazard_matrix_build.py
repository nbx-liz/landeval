"""``scripts/build_hazard.py`` end to end, on synthetic archives and a synthetic pin.

What is checked: the stops (a missing, malformed or mismatched pin, a missing or
changed archive) leave the earlier artefacts byte for
byte; every cell state reaches every per-layer sink -- status, ratio, the point
membership, the coverage column and the npz coverage word -- as the sink table
says; the column and the word agree; and the summary printed on standard
output counts a cell with an unread layer on its own row.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from hazard_pin_support import pin_dict, write_pin
from landeval.hazard_overlay import OverlaySpec
from landeval.hazard_sources import HazardSources, Layer
from landeval.mesh import mesh_bounds, mesh_center, mesh_code, primary_mesh_of

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_hazard.py"

LANDSLIDE = OverlaySpec("landslide", "L", "landslide", ("Polygon",))
FLOOD = OverlaySpec("flood_max", "F", "flood", ("20_max",), "A31b_201", "rank_code")
INLAND = OverlaySpec("inland_flood", "I", "inland_flood", ())
SPECS = (LANDSLIDE, FLOOD, INLAND)

# (latitude, prefecture). Cells 0-5 lie in the primary mesh the manifest lists, 6-7 in
# the next one north, which it does not list (no archive for it, for the mesh-split layer).
CELLS: tuple[tuple[float, str | None], ...] = (
    (35.00, "01001"),  # landslide C+H, flood C+H
    (35.02, "01001"),  # landslide C.0, flood C.0
    (35.04, None),  # landslide U+H (unknown prefecture), flood C.0
    (35.06, "02001"),  # landslide U.0 (02's member has no rows), flood C.0
    (35.08, "03001"),  # landslide D+H (03 not designated), flood C.0
    (35.10, "03001"),  # landslide D.0, flood C.0
    (36.00, "01001"),  # landslide C.0, flood U+H
    (36.02, "01001"),  # landslide C.0, flood U.0
)
LANDSLIDE_HIT = (0, 2, 4)
LISTED_MESH = primary_mesh_of(mesh_code(CELLS[0][0], 139.0, 500))
FLOOD_HIT = (0, 6)


def load_script() -> ModuleType:
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_hazard_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources() -> HazardSources:
    return HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=(LISTED_MESH,),
        prefectures={"01": "A", "02": "B", "03": "C"},
        layers={
            "landslide": Layer(
                label="l",
                split="prefecture",
                files={"01": "l/A33_01.zip", "02": "l/A33_02.zip"},
                not_designated={"03": "the layer does not apply"},
            ),
            "flood": Layer(label="f", split="primary_mesh", patterns=("f/f_{mesh}.zip",)),
            "inland_flood": Layer(label="i", unavailable="not ingested"),
        },
    )


def cells() -> pd.DataFrame:
    codes = [mesh_code(lat, 139.0, 500) for lat, _ in CELLS]
    centres = [mesh_center(code) for code in codes]
    return pd.DataFrame(
        {"mesh_id": codes, "lat": [c[0] for c in centres], "lon": [c[1] for c in centres]}
    )


def cell_boxes(frame: pd.DataFrame, rows: tuple[int, ...]) -> list[object]:
    out = []
    for row in rows:
        south, west, north, east = mesh_bounds(str(frame["mesh_id"].iloc[row]))
        out.append(box(west, south, east, north))
    return out


def write_member(path: Path, name: str, frame: gpd.GeoDataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    work = path.parent / f"{path.stem}_shp"
    work.mkdir()
    frame.to_file(work / "m.shp")
    with zipfile.ZipFile(path, "w") as writer:
        for part in work.iterdir():
            writer.write(part, name[: -len(".shp")] + part.suffix)


def write_archives(raw: Path, frame: pd.DataFrame) -> None:
    crs = "EPSG:6668"
    landslide_hits = cell_boxes(frame, LANDSLIDE_HIT)
    write_member(
        raw / "landslide" / "A33_01.zip",
        "A33_01Polygon.shp",
        gpd.GeoDataFrame({"A33_002": [2] * len(landslide_hits)}, geometry=landslide_hits, crs=crs),
    )
    write_member(
        raw / "landslide" / "A33_02.zip",
        "A33_02Polygon.shp",
        gpd.GeoDataFrame({"A33_002": []}, geometry=gpd.GeoSeries([], crs=crs)),
    )
    flood_hits = cell_boxes(frame, FLOOD_HIT)
    write_member(
        raw / "flood" / f"f_{LISTED_MESH}.zip",
        f"20_max/f_{LISTED_MESH}.shp",
        gpd.GeoDataFrame({"A31b_201": [2] * len(flood_hits)}, geometry=flood_hits, crs=crs),
    )


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[str]]:
    module = load_script()
    frame = cells()
    cells_path = tmp_path / "cells.parquet"
    frame.to_parquet(cells_path, index=False)
    raw = tmp_path / "raw"
    write_archives(raw, frame)
    write_pin(tmp_path / "pin.json", pin_dict(sources(), raw, SPECS))

    def fake_attach(cells_frame: pd.DataFrame, _boundaries: object) -> pd.DataFrame:
        codes = [code for _, code in CELLS]
        return cells_frame.assign(muni_code=pd.array(codes, dtype="string"))

    monkeypatch.setattr(module, "SPECS", SPECS)
    monkeypatch.setattr(module, "load_hazard_sources", lambda _path: sources())
    monkeypatch.setattr(module, "load_boundaries", lambda _path: None)
    monkeypatch.setattr(module, "attach_municipality", fake_attach)
    argv = ["build_hazard.py", "--root", str(tmp_path), "--cells", str(cells_path)]
    argv += ["--raw", str(raw), "--out", str(tmp_path / "out" / "mesh_hazard.parquet")]
    argv += ["--pin", str(tmp_path / "pin.json")]
    return module, argv


def build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    module, argv = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    frame = pd.read_parquet(tmp_path / "out" / "mesh_hazard.parquet")
    with np.load(tmp_path / "out" / "hazard_points.npz") as data:
        points = {key: np.asarray(data[key]) for key in data.files}
    return frame, points


def test_every_state_reaches_each_per_layer_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, _ = build(tmp_path, monkeypatch)

    assert list(frame["landslide_status"]) == [
        "risk",
        "outside",
        "risk",
        "no_data",
        "risk",
        "not_designated",
        "outside",
        "outside",
    ]
    assert list(frame["landslide_covered"]) == [True, True, False, False, True, True, True, True]
    assert list(frame["landslide_hazard_ratio"]) == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    assert list(frame["flood_max_status"]) == ["risk_0.5m"] + ["outside"] * 5 + [
        "risk_0.5m",
        "no_data",
    ]
    assert list(frame["flood_max_covered"]) == [True] * 6 + [False, False]
    assert list(frame["flood_max_max_depth_m"].fillna(-9)) == [0.5] + [-9] * 5 + [0.5, -9]

    assert set(frame["inland_flood_status"]) == {"no_data"}
    assert not frame["inland_flood_covered"].any()
    assert frame["inland_flood_hazard_ratio"].isna().all()


def test_the_npz_holds_membership_per_point_and_coverage_per_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, points = build(tmp_path, monkeypatch)
    layers = [str(name) for name in points["layers"]]
    per_cell = int(points["points_per_cell"])

    assert points["coverage"].shape == (len(frame),)
    by_cell = points["membership"].reshape(len(frame), per_cell)
    for name in ("landslide", "flood_max", "inland_flood"):
        bit = 1 << layers.index(name)
        hits = (by_cell & bit) != 0
        expected_hit = {"landslide": LANDSLIDE_HIT, "flood_max": FLOOD_HIT}.get(name, ())
        assert [bool(row.all()) for row in hits] == [i in expected_hit for i in range(len(frame))]
        assert [bool(row.any()) for row in hits] == [i in expected_hit for i in range(len(frame))]
    assert list(points["available"]) == ["landslide", "flood_max"]


def test_the_coverage_word_agrees_with_the_coverage_column_for_every_available_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, points = build(tmp_path, monkeypatch)
    layers = [str(name) for name in points["layers"]]

    for name in [str(n) for n in points["available"]]:
        bit = 1 << layers.index(name)
        word = (points["coverage"] & bit) != 0
        assert list(word) == list(frame[f"{name}_covered"]), name
    inland = 1 << layers.index("inland_flood")
    assert not ((points["coverage"] & inland) != 0).any()


def test_the_hazard_free_ratio_counts_every_read_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 - k/n over the obtained layers whatever their coverage."""
    frame, _ = build(tmp_path, monkeypatch)

    hit = set(LANDSLIDE_HIT) | set(FLOOD_HIT)
    assert list(frame["hazard_free_ratio"]) == [0.0 if i in hit else 1.0 for i in range(len(frame))]


def test_a_cell_with_an_unread_layer_is_counted_on_its_own_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines() if "セル" in line and line.strip()}

    # Cells 2, 3, 6 and 7 have an obtained layer that is not covered: they are
    # left out of every band, whatever their ratio (a cell at 1.00 with an unread layer included).
    assert int(lines["判定不能"].split()[1]) == 4
    assert int(lines["1.00"].split()[1]) == 2  # cells 1 and 5; 3 and 7 are unjudged
    assert int(lines["0.00–0.10"].split()[1]) == 2  # cells 0 and 4; 2 and 6 are unjudged


# --- the stops leave the artefacts as they were -----------------------------------


def _existing(tmp_path: Path) -> dict[Path, bytes]:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    files = {out / "mesh_hazard.parquet": b"earlier table", out / "hazard_points.npz": b"earlier"}
    for path, content in files.items():
        path.write_bytes(content)
    return files


def _stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: object) -> None:
    module, argv = prepare(tmp_path, monkeypatch)
    before = _existing(tmp_path)
    assert callable(change)
    change()
    pin = tmp_path / "pin.json"
    pin_before = pin.read_bytes() if pin.exists() else None
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 1
    for path, content in before.items():
        assert path.read_bytes() == content
    # The pin is neither regenerated nor repaired by the build.
    assert (pin.read_bytes() if pin.exists() else None) == pin_before


def test_no_pin_stops_without_writing_or_regenerating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = tmp_path / "pin.json"
    _stopped(tmp_path, monkeypatch, pin.unlink)
    assert not pin.exists()


@pytest.mark.parametrize("damage", ["not-json", "both-keys"])
def test_a_malformed_pin_stops_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    pin = tmp_path / "pin.json"

    def change() -> None:
        if damage == "not-json":
            pin.write_text("{", encoding="utf-8")
            return
        data = json.loads(pin.read_text(encoding="utf-8"))
        entry = data["layers"]["landslide"]["entries"]["landslide/A33_01.zip"]["Polygon"]
        entry["absent"] = True
        pin.write_text(json.dumps(data), encoding="utf-8")

    _stopped(tmp_path, monkeypatch, change)


def test_a_pin_for_other_fragments_stops_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = tmp_path / "pin.json"

    def change() -> None:
        data = json.loads(pin.read_text(encoding="utf-8"))
        data["layers"]["landslide"]["fragments"] = ["Polygon", "Line"]
        pin.write_text(json.dumps(data), encoding="utf-8")

    _stopped(tmp_path, monkeypatch, change)


def test_a_missing_archive_stops_before_any_layer_is_overlaid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stopped(tmp_path, monkeypatch, (tmp_path / "raw" / "flood" / f"f_{LISTED_MESH}.zip").unlink)


def test_an_archive_that_changed_since_the_pin_stops_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "raw" / "landslide" / "A33_02.zip"

    def change() -> None:
        with zipfile.ZipFile(archive, "a") as writer:
            writer.writestr("note.txt", "appended after the pin was made")

    _stopped(tmp_path, monkeypatch, change)


def _rewrite(archive: Path, member: str, rows: gpd.GeoDataFrame) -> None:
    archive.unlink()
    work = archive.parent / f"{archive.stem}_shp"
    for leftover in work.iterdir():
        leftover.unlink()
    work.rmdir()
    write_member(archive, member, rows)


@pytest.mark.parametrize("row", ["missing_member", "unplaceable_row", "bad_depth_code"])
def test_a_member_that_cannot_be_ingested_stops_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: str
) -> None:
    """A listed member that is missing, holds an unplaceable row or a
    depth code outside its list stops the build after the pre-checks pass."""
    pin = tmp_path / "pin.json"
    raw = tmp_path / "raw"

    def change() -> None:
        geometry = cell_boxes(cells(), (0,))
        crs = "EPSG:6668"
        if row == "missing_member":
            data = json.loads(pin.read_text(encoding="utf-8"))
            entry = data["layers"]["landslide"]["entries"]["landslide/A33_01.zip"]["Polygon"]
            entry["members"] = ["A33_01Polygon_gone.shp"]
            pin.write_text(json.dumps(data), encoding="utf-8")
            return
        if row == "unplaceable_row":
            rows = gpd.GeoDataFrame({"A33_002": [2, 2]}, geometry=[geometry[0], None], crs=crs)
            _rewrite(raw / "landslide" / "A33_01.zip", "A33_01Polygon.shp", rows)
        else:
            rows = gpd.GeoDataFrame({"A31b_201": [9]}, geometry=geometry, crs=crs)
            _rewrite(raw / "flood" / f"f_{LISTED_MESH}.zip", f"20_max/f_{LISTED_MESH}.shp", rows)
        # The pin is made again from the new bytes, so only the row is at fault.
        write_pin(pin, pin_dict(sources(), raw, SPECS))

    _stopped(tmp_path, monkeypatch, change)
