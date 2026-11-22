"""Reading the small-area (町丁・字等) boundaries.

The traps here are all about a file that reads *successfully* while saying
something other than what it holds:

- the DBF is CP932, and a wrong encoding produces mojibake town names rather
  than an error, so the encoding is asserted rather than detected;
- `KEY_CODE` is not the key -- a file can carry more rows than codes, because
  a 町丁字 split by a river is two polygons -- so anything keyed on it before
  dissolving is silently counting parts as places; and
- some rows have no name at all (water-survey districts among them), and
  dropping them quietly turns "these rows were not places" into a number
  nobody ever sees.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from landeval.smallarea import (
    MUNI_CODE,
    WATER_HCODE,
    SmallAreaError,
    SmallAreas,
    archive_for,
    load_small_areas,
)


def write_archive(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    code: str = "13",
    cpg: bool = True,
) -> Path:
    """Build a shapefile archive shaped like e-Stat's.

    `cpg=False` matches the real files, which carry no encoding sidecar.
    """
    frame = gpd.GeoDataFrame(
        rows,
        geometry=[
            Polygon([(139.7, 35.6), (139.8, 35.6), (139.8, 35.7), (139.7, 35.7)]) for _ in rows
        ],
        crs="EPSG:4612",
    )
    stem = f"r2ka{code}"
    folder = tmp_path / stem
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_file(folder / f"{stem}.shp", encoding="cp932")
    if not cpg:
        (folder / f"{stem}.cpg").unlink()

    archive = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for part in sorted(folder.iterdir()):
            bundle.write(part, arcname=part.name)
    return archive


def row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "KEY_CODE": "13101001",
        "PREF": "13",
        "CITY": "13101",
        "PREF_NAME": "東京都",
        "CITY_NAME": "千代田区",
        "S_NAME": "丸の内一丁目",
        "HCODE": 8101,
        "AREA": 1000.0,
        "JINKO": 42,
    }
    base.update(overrides)
    return base


# --- the encoding -----------------------------------------------------------


def test_the_names_survive_the_read(tmp_path: Path) -> None:
    """CP932 in, Japanese out. This is the assertion that fails when the
    encoding is guessed wrong, because the wrong guess does not raise -- it
    returns 「荳ｸ縺ｮ蜀・ｸ譚｡逶ｮ」."""
    archive = write_archive(tmp_path, [row()])

    areas = load_small_areas(archive)

    assert areas.frame.iloc[0]["S_NAME"] == "丸の内一丁目"
    assert areas.frame.iloc[0]["CITY_NAME"] == "千代田区"


def test_an_archive_without_a_cpg_still_decodes(tmp_path: Path) -> None:
    """The case this argument exists for.

    A shapefile may carry a `.cpg` sidecar naming its encoding, and GDAL honours
    it. **e-Stat ships none** -- the archive holds `prj/shp/shx/dbf` and nothing
    else -- so the encoding has to come from the caller. Dropping that argument
    does not raise; it returns mojibake where the town name was.

    A UTF-8 archive is no fixture for a refusal: geopandas writes a `.cpg`, so
    the file describes itself and reads back correctly. Such a test would
    measure the fixture, not the reader.
    """
    archive = write_archive(tmp_path, [row()], cpg=False)

    assert load_small_areas(archive).frame.iloc[0]["S_NAME"] == "丸の内一丁目"


# --- the key ----------------------------------------------------------------


def test_parts_of_one_area_become_one_row(tmp_path: Path) -> None:
    """A file can carry more rows than codes: a 町丁字 cut by a river is two
    polygons, and counting them as two places
    inflates every count taken from the table."""
    archive = write_archive(tmp_path, [row(), row(), row(KEY_CODE="13101002")])

    areas = load_small_areas(archive)

    assert len(areas.frame) == 2
    assert areas.frame["KEY_CODE"].is_unique
    assert areas.parts_merged == 1


def test_the_key_is_unique_after_loading(tmp_path: Path) -> None:
    archive = write_archive(tmp_path, [row(KEY_CODE=f"1310100{index}") for index in range(4)])

    assert load_small_areas(archive).frame["KEY_CODE"].is_unique


# --- what is left out, and said ---------------------------------------------


def test_unnamed_rows_are_dropped_and_counted(tmp_path: Path) -> None:
    """Some rows have no `S_NAME`. Dropping them is right -- they are not
    places anyone can be told to search in -- but dropping them silently makes
    "we excluded some" indistinguishable from "there were none"."""
    archive = write_archive(tmp_path, [row(), row(KEY_CODE="13101002", S_NAME=None)])

    areas = load_small_areas(archive)

    assert len(areas.frame) == 1
    assert areas.unnamed == 1


def test_water_districts_are_dropped_and_counted(tmp_path: Path) -> None:
    """HCODE 8154 is a water-survey district. It is a real polygon and not a
    place to live, and the two facts are reported separately."""
    archive = write_archive(
        tmp_path,
        [row(), row(KEY_CODE="13101002", S_NAME="東京湾", HCODE=WATER_HCODE)],
    )

    areas = load_small_areas(archive)

    assert len(areas.frame) == 1
    assert areas.water == 1


def test_a_clean_archive_reports_zero_exclusions(tmp_path: Path) -> None:
    """The counters must distinguish "nothing was excluded" from "nothing
    checked". Zero is an answer here, not an absence."""
    areas = load_small_areas(write_archive(tmp_path, [row()]))

    assert (areas.unnamed, areas.water, areas.parts_merged) == (0, 0, 0)


# --- finding the archives ---------------------------------------------------


def test_the_archive_is_found_by_prefecture_code(tmp_path: Path) -> None:
    write_archive(tmp_path, [row()], code="13")
    write_archive(tmp_path, [row(PREF="12")], code="12")

    assert archive_for(tmp_path, "13").name == "r2ka13.zip"


def test_a_missing_prefecture_is_an_error_not_an_empty_frame(tmp_path: Path) -> None:
    """An absent prefecture means the fetch did not finish. Returning nothing
    would make a partial download look like a region with no towns."""
    with pytest.raises(SmallAreaError, match="09"):
        archive_for(tmp_path, "09")


def test_loading_several_prefectures_keeps_them_apart(tmp_path: Path) -> None:
    write_archive(tmp_path, [row()], code="13")
    write_archive(
        tmp_path,
        [row(KEY_CODE="12101001", PREF="12", PREF_NAME="千葉県", CITY="12101")],
        code="12",
    )

    areas = SmallAreas.load_all(tmp_path, ("13", "12"))

    assert set(areas.frame["PREF"]) == {"13", "12"}
    assert len(areas.frame) == 2


# --- the join key -----------------------------------------------------------


def test_the_municipality_code_is_the_five_digit_one(tmp_path: Path) -> None:
    """`CITY` is three digits *within a prefecture*, so several prefectures
    together carry far fewer distinct values than municipalities. Joining a
    town to a municipality on it alone puts a city of prefecture 08 (08201) and any other
    prefecture's 201 in one bucket -- a boundary that is not anchored.
    N03 keys on five digits, so this does too.
    """
    archive = write_archive(tmp_path, [row(PREF="08", CITY="201")])

    areas = load_small_areas(archive)

    assert areas.frame.iloc[0][MUNI_CODE] == "08201"


def test_the_same_city_number_in_two_prefectures_stays_apart(tmp_path: Path) -> None:
    write_archive(tmp_path, [row(PREF="13", CITY="101", KEY_CODE="131010010")], code="13")
    write_archive(tmp_path, [row(PREF="12", CITY="101", KEY_CODE="121010010")], code="12")

    areas = SmallAreas.load_all(tmp_path, ("13", "12"))

    assert sorted(areas.frame[MUNI_CODE]) == ["12101", "13101"]
