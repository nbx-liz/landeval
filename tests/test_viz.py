"""Tests for the map output."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from landeval.mesh import mesh_bounds, mesh_center, mesh_code
from landeval.viz import (
    BAND_ALPHA,
    READOUT_ABSENT,
    READOUT_ALPHABET,
    ValueGrid,
    _readout_html,
    band_colour,
    build_map,
    rasterize,
    value_grid,
)

# Five bands on an evenly spaced grid with a neutral sequential palette, chosen for
# these tests; the bands a real map uses are the caller's choice.
DEFAULT_BANDS: tuple[tuple[float, str], ...] = (
    (25.0, "#3b4cc0"),
    (55.0, "#8db0fe"),
    (85.0, "#dddcdc"),
    (115.0, "#f4987a"),
    (145.0, "#b40426"),
)

DEST = (35.680000, 139.760000)


def cells(values: list[tuple[float, float, float]]) -> pd.DataFrame:
    """values: (lat, lon, freeflow_min)."""
    return pd.DataFrame(
        {
            "mesh_id": pd.array([mesh_code(v[0], v[1], 500) for v in values], dtype="string"),
            "lat": [v[0] for v in values],
            "lon": [v[1] for v in values],
            "freeflow_min": pd.array([v[2] for v in values], dtype="Float64"),
        }
    )


# --- colour banding --------------------------------------------------------


@pytest.mark.parametrize("minutes", [0.0, 25.0])
def test_first_band_is_upper_inclusive(minutes: float) -> None:
    assert band_colour(minutes, DEFAULT_BANDS) == band_colour(1.0, DEFAULT_BANDS)


def test_bands_are_distinct() -> None:
    colours = {band_colour(m, DEFAULT_BANDS) for m in (15.0, 45.0, 75.0, 105.0, 135.0)}

    assert len(colours) == len(DEFAULT_BANDS)


def test_beyond_the_last_band_has_no_colour() -> None:
    """Out of range must be blank, not clamped to the worst band."""
    assert band_colour(145.1, DEFAULT_BANDS) is None
    assert band_colour(1000.0, DEFAULT_BANDS) is None


# --- rasterising -----------------------------------------------------------


def test_raster_covers_exactly_the_cells_given() -> None:
    lat, lon = DEST
    frame = cells([(lat, lon, 20.0)])

    raster = rasterize(frame, "freeflow_min", bands=DEFAULT_BANDS)

    south, west, north, east = mesh_bounds(mesh_code(lat, lon, 500))
    assert raster.image.shape == (1, 1, 4)
    assert (raster.south, raster.west) == pytest.approx((south, west))
    assert (raster.north, raster.east) == pytest.approx((north, east))


def test_north_is_the_top_row_of_the_image() -> None:
    """An inverted image would put the whole map upside down."""
    lat, lon = DEST
    frame = cells([(lat, lon, 20.0), (lat + 1 / 240, lon, 140.0)])

    raster = rasterize(frame, "freeflow_min", bands=DEFAULT_BANDS)

    assert raster.image.shape[0] == 2
    top = tuple(raster.image[0, 0][:3])
    bottom = tuple(raster.image[1, 0][:3])
    assert top == band_colour(140.0, DEFAULT_BANDS)
    assert bottom == band_colour(20.0, DEFAULT_BANDS)


def test_gaps_stay_transparent() -> None:
    """A missing cell must look like a gap, not like a value."""
    lat, lon = DEST
    frame = cells([(lat, lon, 20.0), (lat, lon + 2 / 160, 20.0)])

    raster = rasterize(frame, "freeflow_min", bands=DEFAULT_BANDS)

    assert raster.image.shape[1] == 3
    assert raster.image[0, 1, 3] == 0  # the skipped column
    assert raster.image[0, 0, 3] == BAND_ALPHA


def test_values_beyond_the_bands_are_transparent() -> None:
    frame = cells([(DEST[0], DEST[1], 400.0)])

    raster = rasterize(frame, "freeflow_min", bands=DEFAULT_BANDS)

    assert raster.image[0, 0, 3] == 0


def test_null_values_are_transparent() -> None:
    frame = cells([(DEST[0], DEST[1], 20.0)])
    frame["freeflow_min"] = pd.array([None], dtype="Float64")

    raster = rasterize(frame, "freeflow_min", bands=DEFAULT_BANDS)

    assert raster.image[0, 0, 3] == 0


def test_every_drawn_pixel_maps_back_to_its_cell() -> None:
    lat, lon = DEST
    frame = cells([(lat + i / 240, lon + j / 160, 20.0) for i in range(3) for j in range(4)])

    raster = rasterize(frame, "freeflow_min", bands=DEFAULT_BANDS)

    assert raster.image.shape[:2] == (3, 4)
    assert np.count_nonzero(raster.image[:, :, 3]) == 12
    for _, row in frame.iterrows():
        centre = mesh_center(str(row["mesh_id"]))
        assert raster.south <= centre[0] <= raster.north
        assert raster.west <= centre[1] <= raster.east


def test_empty_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        rasterize(cells([]), "freeflow_min")


def test_unknown_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a column"):
        rasterize(cells([(DEST[0], DEST[1], 20.0)]), "nope")


# --- html output -----------------------------------------------------------


def contains_text(html: str, text: str) -> bool:
    """Whether the page carries ``text``, raw or JS-escaped.

    Folium writes layer names and attributions into JavaScript object literals,
    where non-ASCII is emitted as ``\\uXXXX``. That renders correctly in a
    browser, so both forms count as present.
    """
    escaped = "".join(f"\\u{ord(char):04x}" if ord(char) > 127 else char for char in text)
    return text in html or escaped in html


def test_map_is_written_and_self_describing(tmp_path: Path) -> None:
    frame = cells([(DEST[0] + i / 240, DEST[1], 20.0 + 30 * i) for i in range(5)])
    out = tmp_path / "map.html"

    build_map(
        frame,
        DEST,
        "Central Station",
        out,
        bands=DEFAULT_BANDS,
        generated_at=datetime(2026, 7, 8, 15, 30),
    )

    html = out.read_text(encoding="utf-8")
    assert out.exists()
    # Attribution is required by the GSI tile terms.
    assert contains_text(html, "国土地理院")
    assert "maps.gsi.go.jp" in html
    assert "2026-07-08 15:30" in html  # the reader must see how old this is
    assert contains_text(html, "Central Station")
    assert "data:image/png;base64," in html


def test_map_includes_the_sampled_points(tmp_path: Path) -> None:
    """The sampled-point payload is invented: round numbers at an arbitrary place."""
    from landeval.mesh import mesh_code

    frame = cells([(DEST[0] + i / 240, DEST[1], 20.0 + 30 * i) for i in range(5)])
    measured = pd.DataFrame(
        {
            "mesh_id": [str(mesh_code(43.5, 142.0, 500))] * 2,
            "lat": [43.5, 43.5],
            "lon": [142.0, 142.0],
            "freeflow_min": [20.0, 20.0],
            "schedule_id": ["arr_0930", "dep_2030"],
            "measured_min": [40.0, 30.0],
        }
    )
    out = tmp_path / "map.html"

    build_map(frame, DEST, "Central Station", out, measured=measured, bands=DEFAULT_BANDS)

    html = out.read_text(encoding="utf-8")
    assert "arr_0930" in html
    assert contains_text(html, "Google 実測")


# --- click readout ---------------------------------------------------------


def num(value: object) -> float:
    """Narrow a pandas scalar to float; its stub type is a wide union."""
    return float(value)  # type: ignore[arg-type]


def readout_frame() -> pd.DataFrame:
    """Three adjacent 500 m cells with known fractions."""
    from landeval.mesh import mesh_center

    codes = ["533545114", "533545113", "533546113"]
    centres = [mesh_center(code) for code in codes]
    return pd.DataFrame(
        {
            "mesh_id": codes,
            "lat": [lat for lat, _ in centres],
            "lon": [lon for _, lon in centres],
            "freeflow_min": [10.0, 20.0, 30.0],
            "ratio": [0.0, 0.48, 1.0],
        }
    )


def read_at(grid: ValueGrid, lat: float, lon: float) -> str:
    """The same arithmetic the emitted JavaScript performs."""
    row = math.floor(lat / grid.lat_step)
    col = math.floor((lon - 100.0) / grid.lon_step)
    if row < grid.row0 or row > grid.row1 or col < grid.col0 or col > grid.col1:
        return READOUT_ABSENT
    return grid.data[(grid.row1 - row) * grid.width + (col - grid.col0)]


def test_the_grid_reads_back_the_value_of_each_cell() -> None:
    frame = readout_frame()
    grid = value_grid(frame, "ratio")

    for _, row in frame.iterrows():
        character = read_at(grid, num(row["lat"]), num(row["lon"]))
        assert READOUT_ALPHABET.index(character) == round(num(row["ratio"]) * 25)


def test_a_point_outside_the_grid_reads_as_absent() -> None:
    grid = value_grid(readout_frame(), "ratio")

    assert read_at(grid, 43.0, 141.3) == READOUT_ABSENT


def test_cells_absent_from_the_frame_stay_absent() -> None:
    """A hole in the data must not borrow a neighbour's value."""
    frame = readout_frame()
    grid = value_grid(frame.iloc[[0, 2]], "ratio")

    assert grid.data.count(READOUT_ABSENT) >= 1


def test_a_missing_value_is_not_packed_as_zero() -> None:
    frame = readout_frame()
    frame.loc[1, "ratio"] = float("nan")
    grid = value_grid(frame, "ratio")

    at = read_at(grid, num(frame.loc[1, "lat"]), num(frame.loc[1, "lon"]))

    assert at == READOUT_ABSENT


def test_an_empty_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        value_grid(readout_frame().iloc[:0], "ratio")


def test_an_unknown_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="ratio_x"):
        value_grid(readout_frame(), "ratio_x")


def test_the_map_embeds_the_readout_when_asked(tmp_path: Path) -> None:
    frame = readout_frame()
    output = build_map(
        frame,
        (35.68, 139.76),
        "目的地",
        tmp_path / "m.html",
        bands=DEFAULT_BANDS,
        class_column=None,
        readout_column="ratio",
    )
    html = output.read_text(encoding="utf-8")

    assert "landeval-readout" in html
    assert "latlng" in html


def test_the_map_omits_the_readout_by_default(tmp_path: Path) -> None:
    output = build_map(
        readout_frame(), (35.68, 139.76), "目的地", tmp_path / "m.html", bands=DEFAULT_BANDS
    )

    assert "landeval-readout" not in output.read_text(encoding="utf-8")


def test_a_value_grid_can_carry_a_scale_other_than_a_fraction() -> None:
    """The measured tables store minutes, not fractions."""
    frame = readout_frame()
    frame["minutes"] = [0.0, 100.0, 200.0]
    grid = value_grid(frame, "minutes", levels=24, scale=200.0)

    read = [
        READOUT_ALPHABET.index(read_at(grid, num(row["lat"]), num(row["lon"])))
        for _, row in frame.iterrows()
    ]

    assert read == [0, 12, 24]


def test_a_value_beyond_the_scale_is_clamped_rather_than_wrapped() -> None:
    frame = readout_frame()
    frame["minutes"] = [0.0, 75.0, 400.0]
    grid = value_grid(frame, "minutes", levels=24, scale=200.0)

    assert (
        READOUT_ALPHABET.index(read_at(grid, num(frame.loc[2, "lat"]), num(frame.loc[2, "lon"])))
        == 24
    )


def test_a_non_positive_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="scale"):
        value_grid(readout_frame(), "ratio", scale=0.0)


def test_the_numeric_readout_reports_the_unit_it_was_given(tmp_path: Path) -> None:
    frame = readout_frame()
    frame["minutes"] = [10.0, 100.0, 200.0]
    output = build_map(
        frame,
        (35.68, 139.76),
        "目的地",
        tmp_path / "m.html",
        bands=DEFAULT_BANDS,
        readout_column="minutes",
        readout_levels=24,
        readout_scale=200.0,
        readout_unit="分",
        readout_style="value",
    )
    html = output.read_text(encoding="utf-8")

    assert 'STYLE = "value"' in html
    assert "SCALE = 200.0" in html


# --- the map-wide caveat ---------------------------------------------------
#
# A `build_map` that discarded `caveat` entirely would leave a suite green that
# only tested the function composing the sentence. The point of the parameter
# is that a limit of the map reaches the reader of the *file*, so these check
# that it is actually written into the HTML.


def classified_frame() -> pd.DataFrame:
    frame = readout_frame()
    frame["klass"] = ["a", "a", "b"]
    return frame


CAVEAT_PALETTE = {"a": "#0b5f2a", "b": "#7a4fb0"}
CAVEAT_LABELS = {"a": "判定できた", "b": "判定不能"}


def test_a_caveat_reaches_the_rendered_file(tmp_path: Path) -> None:
    output = build_map(
        classified_frame(),
        (35.68, 139.76),
        "目的地",
        tmp_path / "m.html",
        class_column="klass",
        palette=CAVEAT_PALETTE,
        labels=CAVEAT_LABELS,
        caveat="このレイヤは全セルで未取得: 内水氾濫",
    )

    assert "内水氾濫" in output.read_text(encoding="utf-8")


def test_a_map_without_a_caveat_says_nothing_extra(tmp_path: Path) -> None:
    output = build_map(
        classified_frame(),
        (35.68, 139.76),
        "目的地",
        tmp_path / "m.html",
        class_column="klass",
        palette=CAVEAT_PALETTE,
        labels=CAVEAT_LABELS,
    )
    html = output.read_text(encoding="utf-8")

    assert "未取得" not in html
    assert "判定不能" in html


def test_a_caveat_on_a_map_that_cannot_show_it_is_rejected(tmp_path: Path) -> None:
    """Silently dropping it is the failure the parameter exists to prevent."""
    with pytest.raises(ValueError, match="caveat"):
        build_map(
            readout_frame(),
            (35.68, 139.76),
            "目的地",
            tmp_path / "m.html",
            bands=DEFAULT_BANDS,
            caveat="このレイヤは全セルで未取得: 内水氾濫",
        )


# --- the readout distinguishes "nothing here" from "no value for this" -------


def run_readout(html: str, points: list[tuple[float, float]]) -> list[str]:
    """Evaluate the emitted readout for each point, in a real JavaScript engine.

    The defect this guards is a control-flow conflation inside the emitted
    script, so asserting on the Python that emits it would be asserting on the
    wrong artefact. Only the variable block and the two pure functions are
    sliced out -- `attach` needs a DOM and answers nothing about the reading.
    """
    body = html[html.index("  var DATA = ") : html.index("  function attach()")]
    probe = "console.log(JSON.stringify(POINTS.map(function (p) {" + (
        " return describe(lookup(p[0], p[1])); })));"
    )
    script = f"var POINTS = {json.dumps(points)};\n{body}\n{probe}"
    node = shutil.which("node")
    assert node is not None, "the skipif guard should have run first"
    result = subprocess.run(  # noqa: S603 - resolved argv, no shell
        [node, "-e", script], capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return list(json.loads(result.stdout))


def interior_hole(grid: ValueGrid) -> tuple[float, float]:
    """A point inside the grid rectangle whose cell carries no value.

    Not `read_at`: that helper mirrors the arithmetic the readout does,
    and it answers READOUT_ABSENT for a point off the rectangle as readily as
    for a hole inside it -- which is the conflation under test. The slot is
    picked out of the packed string, where the two cannot be confused.
    """
    index = grid.data.index(READOUT_ABSENT)
    row = grid.row1 - index // grid.width
    col = grid.col0 + index % grid.width
    return (row + 0.5) * grid.lat_step, 100.0 + (col + 0.5) * grid.lon_step


@pytest.mark.skipif(shutil.which("node") is None, reason="no JavaScript engine to run the readout")
def test_a_cell_with_no_value_is_not_reported_as_outside_the_target_area() -> None:
    """Neither miss may read as 「対象範囲外」, which is a statement about
    scope -- true for only one of them. On an estimate map that puts
    cells the map itself paints and classifies, some of which passed the
    transport screen, under the same words as a click on the sea.
    """
    frame = readout_frame()
    grid = value_grid(frame, "ratio")
    hole = interior_hole(grid)

    off_map, no_value, has_value = run_readout(
        _readout_html(grid, "t"),
        [(43.0, 141.3), hole, (num(frame["lat"][2]), num(frame["lon"][2]))],
    )

    assert off_map == "対象範囲外"
    assert no_value == "圏域内・値なし（凡例の分類を参照）"
    assert no_value != off_map
    assert "100%" in has_value


@pytest.mark.skipif(shutil.which("node") is None, reason="no JavaScript engine to run the readout")
def test_a_character_outside_the_alphabet_is_reported_rather_than_counted() -> None:
    """`indexOf` answers -1 for an unknown character, and -1 must not go straight
    into the arithmetic: a negative percentage would render in the same confident
    bold as a real one. The packed string is written from the alphabet alone, so an
    unknown character is a fact about the data."""
    grid = value_grid(readout_frame(), "ratio")
    corrupt = ValueGrid(
        data="?" * len(grid.data),
        row0=grid.row0,
        row1=grid.row1,
        col0=grid.col0,
        col1=grid.col1,
        lat_step=grid.lat_step,
        lon_step=grid.lon_step,
    )
    centre = mesh_center("533545114")

    (reading,) = run_readout(_readout_html(corrupt, "t"), [centre])

    assert reading == "読み出しデータが壊れています"


# --- the bands are the caller's -------------------------


def test_colouring_by_value_without_bands_is_refused() -> None:
    """The library keeps no default bands: they are a decision of the caller."""
    frame = cells([(DEST[0], DEST[1], 20.0)])

    with pytest.raises(ValueError, match="bands are required"):
        rasterize(frame, "freeflow_min")


def test_a_value_map_without_bands_is_refused(tmp_path: Path) -> None:
    frame = cells([(DEST[0], DEST[1], 20.0)])

    with pytest.raises(ValueError, match="bands are required"):
        build_map(frame, DEST, "目的地", tmp_path / "m.html")
