"""Tests for the cell statuses ``scripts/build_hazard.py`` writes.

The script is run end to end on synthetic cells, with the overlay and the
municipality join replaced by stand-ins, so what is checked is the status the
script assigns once it knows each cell's prefecture and severity.
"""

from __future__ import annotations

import importlib.util
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
from landeval.hazard import (
    HIT_WITHOUT_DEPTH,
    NO_DATA,
    NOT_DESIGNATED,
    OUTSIDE,
    SUBGRID,
    subcell_lonlat,
)
from landeval.hazard_overlay import SPECS, LayerRun, OverlaySpec
from landeval.hazard_sources import HazardSources, Layer
from landeval.mesh import mesh_center, mesh_code

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_hazard.py"


def load_script() -> ModuleType:
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_hazard_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_cells(count: int) -> pd.DataFrame:
    codes = [mesh_code(35.0 + 0.01 * i, 139.0, 500) for i in range(count)]
    centres = [mesh_center(code) for code in codes]
    return pd.DataFrame(
        {
            "mesh_id": codes,
            "lat": [lat for lat, _ in centres],
            "lon": [lon for _, lon in centres],
        }
    )


def landslide_sources() -> HazardSources:
    return HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234",),
        prefectures={"01": "A"},
        layers={
            "landslide": Layer(label="landslide", split="prefecture", patterns=("l/{pref}.zip",))
        },
    )


# Cell layout: (municipality code, whether a polygon covers the cell).
LAYOUT: tuple[tuple[str | None, bool], ...] = (
    (None, False),  # no prefecture, nothing hit
    (None, True),  # no prefecture, a polygon hit
    ("99001", False),  # a prefecture the manifest does not cover, nothing hit
    ("99001", True),  # the same prefecture, a polygon hit
    ("01001", False),  # a covered prefecture, nothing hit
)


def run_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    module = load_script()
    cells = synthetic_cells(len(LAYOUT))
    cells_path = tmp_path / "cells.parquet"
    cells.to_parquet(cells_path, index=False)
    _, _, cell_index = subcell_lonlat(cells, mesh_size_m=500, subgrid=SUBGRID)
    hit_cells = np.array([hit for _, hit in LAYOUT])

    def fake_run_layer(spec: OverlaySpec, *_args: object, **_kwargs: object) -> LayerRun:
        severity = np.where(hit_cells[cell_index], HIT_WITHOUT_DEPTH, np.nan)
        return LayerRun(spec, severity, covered=frozenset({"01"}))

    def fake_attach(frame: pd.DataFrame, _boundaries: object) -> pd.DataFrame:
        return frame.assign(muni_code=pd.array([code for code, _ in LAYOUT], dtype="string"))

    monkeypatch.setattr(module, "load_hazard_sources", lambda _path: landslide_sources())
    monkeypatch.setattr(module, "load_boundaries", lambda _path: None)
    monkeypatch.setattr(module, "attach_municipality", fake_attach)
    monkeypatch.setattr(module, "run_layer", fake_run_layer)
    # The overlay is a stand-in, so there is no archive to pin or to check.
    monkeypatch.setattr(module, "load_hazard_pin", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "check_pinned_archives", lambda *_args: None)
    out = tmp_path / "out" / "mesh_hazard.parquet"
    argv = [
        "build_hazard.py",
        "--root",
        str(tmp_path),
        "--cells",
        str(cells_path),
        "--out",
        str(out),
        "--layer",
        "landslide",
        "--layer",
        "landslide_special",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 0
    return pd.read_parquet(out)


@pytest.mark.parametrize("layer", ["landslide", "landslide_special"])
def test_a_cell_whose_prefecture_data_was_not_read_is_never_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    """No prefecture, or one the manifest lacks, gives no evidence of clear ground.

    A polygon that does reach such a cell is still a real intersection, so its
    risk is kept; only the absence of a hit is withheld as no_data.
    """
    statuses = list(run_script(tmp_path, monkeypatch)[f"{layer}_status"])

    assert statuses == [NO_DATA, "risk", NO_DATA, "risk", OUTSIDE]


def two_prefecture_sources() -> HazardSources:
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
    """A zipped landslide shapefile; ``None`` writes the member with no rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if polygons is None:
        polygons = gpd.GeoDataFrame({"A33_002": []}, geometry=gpd.GeoSeries([], crs="EPSG:6668"))
    with zipfile.ZipFile(path, "w") as writer:
        work = path.parent / f"{path.stem}_shp"
        work.mkdir()
        polygons.to_file(work / "A33_Polygon.shp")
        for part in work.iterdir():
            writer.write(part, part.name)


# (municipality code, whether the prefecture-01 polygon covers the cell).
TWO_PREFECTURES: tuple[tuple[str, bool], ...] = (
    ("01001", True),
    ("01001", False),
    ("02001", False),  # its archive holds a landslide member with no rows
)


@pytest.mark.parametrize("layer", ["landslide", "landslide_special"])
def test_a_prefecture_that_ingested_nothing_is_never_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    """Coverage is what the overlay read, not what the manifest lists.

    Prefecture 01 supplies polygons, so the layer comes back available; the
    archive of 02 holds the pinned member, but it has no rows, so nothing
    about 02 was read.
    """
    module = load_script()
    cells = synthetic_cells(len(TWO_PREFECTURES))
    cells_path = tmp_path / "cells.parquet"
    cells.to_parquet(cells_path, index=False)
    lat, lon = float(cells["lat"].iloc[0]), float(cells["lon"].iloc[0])
    zone = gpd.GeoDataFrame(
        {"A33_002": [2]},
        geometry=[box(lon - 0.004, lat - 0.004, lon + 0.004, lat + 0.004)],
        crs="EPSG:6668",
    )
    raw = tmp_path / "raw"
    write_archive(raw / "landslide" / "A33_01.zip", zone)
    write_archive(raw / "landslide" / "A33_02.zip", None)

    def fake_attach(frame: pd.DataFrame, _boundaries: object) -> pd.DataFrame:
        codes = [code for code, _ in TWO_PREFECTURES]
        return frame.assign(muni_code=pd.array(codes, dtype="string"))

    monkeypatch.setattr(module, "load_hazard_sources", lambda _path: two_prefecture_sources())
    monkeypatch.setattr(module, "load_boundaries", lambda _path: None)
    monkeypatch.setattr(module, "attach_municipality", fake_attach)
    out = tmp_path / "out" / "mesh_hazard.parquet"
    pin = write_pin(
        tmp_path / "pin.json",
        pin_dict(two_prefecture_sources(), raw, [s for s in SPECS if s.source == "landslide"]),
    )
    argv = ["build_hazard.py", "--root", str(tmp_path), "--cells", str(cells_path)]
    argv += ["--raw", str(raw), "--out", str(out), "--layer", layer, "--pin", str(pin)]
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 0
    statuses = list(pd.read_parquet(out)[f"{layer}_status"])

    assert statuses == ["risk", OUTSIDE, NO_DATA]


def coastal_layer() -> Layer:
    return Layer(
        label="coastal",
        split="prefecture",
        files={"01": "t/t_01.zip"},
        not_designated={"02": "inland"},
    )


# (prefecture, hazard ratio, max depth, expected status). Prefecture 03 is
# outside the manifest altogether.
COASTAL: tuple[tuple[str | None, float, float, str], ...] = (
    (None, 0.4, 2.0, "risk_2.0m"),
    (None, 0.0, np.nan, NO_DATA),
    ("02", 0.2, 1.0, "risk_1.0m"),
    ("02", 0.0, np.nan, NOT_DESIGNATED),
    ("01", 0.0, np.nan, OUTSIDE),
    ("01", 0.6, np.nan, "risk"),
    ("03", 0.0, np.nan, NO_DATA),
    ("03", 0.2, 5.0, "risk_5.0m"),
)


def test_a_polygon_hit_keeps_its_risk_wherever_the_cell_lies() -> None:
    """Unknown, uncovered or not designated, a real intersection is still risk.

    Only the absence of a hit is withheld: no_data where nothing was read,
    not_designated where the prefecture declares the layer does not apply.
    """
    module = load_script()
    pref = pd.Series([p for p, *_ in COASTAL], dtype="string")
    ratio = np.array([r for _, r, _, _ in COASTAL])
    depth = np.array([d for _, _, d, _ in COASTAL])

    statuses = module.cell_statuses(ratio, depth, pref, coastal_layer(), {"01"})

    assert list(statuses) == [expected for *_, expected in COASTAL]


def test_a_layer_that_reports_no_coverage_has_no_outside_cells() -> None:
    """With nothing ingested, even a manifest-listed prefecture reads no_data."""
    module = load_script()
    pref = pd.Series(["01", "01"], dtype="string")

    statuses = module.cell_statuses(
        np.array([0.0, 0.3]), np.array([np.nan, np.nan]), pref, coastal_layer(), set()
    )

    assert list(statuses) == [NO_DATA, "risk"]
