"""Tests for attaching municipalities to cells.

The name a cell carries is what a reader types into a property search, and the
search form for a designated city lists its wards separately. So the name has to
resolve to the same unit as the code: N03 puts the city in ``N03_004`` and the
ward in ``N03_005``, and dropping the second collapses all 24 wards of Osaka
onto one label while their codes stay distinct.

The join itself is deliberately unforgiving -- a cell outside every polygon
keeps a null rather than being snapped to a neighbour -- and that is pinned here
too, because a silent snap moves a border cell into a municipality it is not in.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from landeval.admin import (
    CITY_COLUMN,
    CODE_COLUMN,
    COUNTY_COLUMN,
    PREF_COLUMN,
    WARD_COLUMN,
    AdminError,
    alias_map,
    attach_municipality,
    load_boundaries,
    municipality_name,
    summarize_by_municipality,
    without_prefecture,
)


def row(pref: str, county: object, city: object, ward: object) -> pd.Series:
    return pd.Series(
        {PREF_COLUMN: pref, COUNTY_COLUMN: county, CITY_COLUMN: city, WARD_COLUMN: ward}
    )


# --- naming ----------------------------------------------------------------


def test_a_plain_city_is_prefecture_plus_city() -> None:
    assert municipality_name(row("愛知県", None, "岡崎市", None)) == "愛知県岡崎市"


def test_a_town_keeps_its_county() -> None:
    """Without the county, several 町 in different counties share a name."""
    assert municipality_name(row("愛知県", "額田郡", "幸田町", None)) == "愛知県額田郡幸田町"


def test_a_designated_city_ward_keeps_the_ward() -> None:
    assert municipality_name(row("大阪府", None, "大阪市", "西区")) == "大阪府大阪市西区"


def test_a_special_ward_of_tokyo_is_unchanged() -> None:
    """Special wards sit in N03_004 and have no N03_005. The row is synthetic: the
    only real units of this shape are in Tokyo, outside this public fixture set."""
    assert municipality_name(row("愛知県", None, "中区", None)) == "愛知県中区"


def test_the_designated_city_name_is_not_dropped_for_the_ward() -> None:
    """`大阪市西区` and `名古屋市西区` are different places."""
    osaka = municipality_name(row("大阪府", None, "大阪市", "西区"))
    nagoya = municipality_name(row("愛知県", None, "名古屋市", "西区"))

    assert osaka != nagoya
    assert "大阪市" in osaka
    assert "名古屋市" in nagoya


def test_a_county_that_is_not_a_county_is_ignored() -> None:
    """N03_003 also carries the city name for designated cities; only 郡 counts."""
    assert municipality_name(row("大阪府", "堺市", "堺市", "北区")) == "大阪府堺市北区"


@pytest.mark.parametrize("blank", [None, "", float("nan")])
def test_a_blank_ward_adds_nothing(blank: object) -> None:
    assert municipality_name(row("愛知県", None, "岡崎市", blank)) == "愛知県岡崎市"


def test_every_ward_of_one_city_gets_its_own_name() -> None:
    names = {
        municipality_name(row("愛知県", None, "名古屋市", ward))
        for ward in ("西区", "北区", "中区", "東区", "港区")
    }

    assert len(names) == 5


# --- loading ---------------------------------------------------------------


def test_an_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(AdminError, match="no N03 archives"):
        load_boundaries(tmp_path)


def test_the_ward_column_is_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A vintage that renames N03_005 must fail, not silently merge the wards."""
    archive = tmp_path / "N03-20240101_14_GML.zip"
    archive.write_bytes(b"")
    frame = gpd.GeoDataFrame(
        {
            PREF_COLUMN: ["大阪府"],
            COUNTY_COLUMN: [None],
            CITY_COLUMN: ["大阪市"],
            CODE_COLUMN: ["27106"],
        },
        geometry=[box(0.0, 0.0, 1.0, 1.0)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr("landeval.admin.gpd.read_file", lambda _: frame)

    with pytest.raises(AdminError, match=WARD_COLUMN):
        load_boundaries(tmp_path)


# --- joining ---------------------------------------------------------------


def boundaries() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            PREF_COLUMN: ["大阪府", "大阪府"],
            COUNTY_COLUMN: [None, None],
            CITY_COLUMN: ["大阪市", "大阪市"],
            WARD_COLUMN: ["西区", "中央区"],
            CODE_COLUMN: ["27106", "27128"],
        },
        geometry=[box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        crs="EPSG:4326",
    )


def test_wards_of_one_city_stay_distinguishable_after_the_join() -> None:
    cells = pd.DataFrame({"mesh_id": ["a", "b"], "lat": [0.5, 0.5], "lon": [0.5, 1.5]})

    joined = attach_municipality(cells, boundaries())

    assert list(joined["muni_code"]) == ["27106", "27128"]
    assert list(joined["muni_name"]) == ["大阪府大阪市西区", "大阪府大阪市中央区"]


def test_a_cell_outside_every_polygon_keeps_a_null() -> None:
    cells = pd.DataFrame({"mesh_id": ["a"], "lat": [9.0], "lon": [9.0]})

    joined = attach_municipality(cells, boundaries())

    assert pd.isna(joined["muni_code"].iloc[0])
    assert pd.isna(joined["muni_name"].iloc[0])


def test_summaries_do_not_merge_two_wards() -> None:
    cells = pd.DataFrame(
        {
            "muni_code": ["27106", "27128"],
            "muni_name": ["大阪府大阪市西区", "大阪府大阪市中央区"],
            "predicted_min": [60.0, 70.0],
        }
    )

    summary = summarize_by_municipality(cells, "predicted_min")

    assert len(summary) == 2


# --- the spellings an address may legitimately arrive in --------------------


def alias_frame() -> gpd.GeoDataFrame:
    """Four units, one of them a name two prefectures both use."""
    rows = [
        ("愛知県", None, "豊橋市", None, "23201"),
        ("愛知県", "額田郡", "幸田町", None, "23441"),
        ("北海道", None, "伊達市", None, "01233"),
        ("福島県", None, "伊達市", None, "07213"),
    ]
    return gpd.GeoDataFrame(
        [
            {
                PREF_COLUMN: pref,
                COUNTY_COLUMN: county,
                CITY_COLUMN: city,
                WARD_COLUMN: ward,
                CODE_COLUMN: code,
                "geometry": box(0, 0, 1, 1),
            }
            for pref, county, city, ward, code in rows
        ],
        crs="EPSG:4326",
    )


def test_the_alias_map_offers_the_spelling_without_the_prefecture() -> None:
    """`municipality_aliases` alone produces no prefecture-less alias, so
    without this a query typed 〇〇市〇〇町1-2 matches no municipality on the
    query side. The plot is refused even though the geocoder resolved it in
    full, and the check that refuses an answer in a different municipality
    switches itself off for exactly those queries."""
    aliases = alias_map(alias_frame())

    assert aliases["豊橋市"] == "愛知県豊橋市"
    assert aliases["愛知県豊橋市"] == "愛知県豊橋市"


def test_the_alias_map_keeps_offering_the_county_less_spelling() -> None:
    """The geocoder drops the county, so all four spellings of one town mean it."""
    aliases = alias_map(alias_frame())

    assert aliases["愛知県額田郡幸田町"] == "愛知県額田郡幸田町"
    assert aliases["愛知県幸田町"] == "愛知県額田郡幸田町"
    assert aliases["額田郡幸田町"] == "愛知県額田郡幸田町"
    assert aliases["幸田町"] == "愛知県額田郡幸田町"


def test_a_bare_name_two_municipalities_claim_is_left_out() -> None:
    """伊達市 exists in Hokkaido and in Fukushima. Admitting it would resolve to
    whichever row was read first and put every stored figure -- commute, hazard,
    the lot -- on the wrong municipality code. It keeps needing its prefecture."""
    aliases = alias_map(alias_frame())

    assert "伊達市" not in aliases
    assert aliases["北海道伊達市"] == "北海道伊達市"
    assert aliases["福島県伊達市"] == "福島県伊達市"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("愛知県名古屋市港区", "名古屋市港区"),
        ("北海道札幌市中央区", "札幌市中央区"),
        ("京都府京都市北区", "京都市北区"),
        ("大阪府大阪市中央区", "大阪市中央区"),
        ("和歌山県和歌山市", "和歌山市"),
        ("東京都山手区", "山手区"),
        ("港区", None),
    ],
)
def test_the_prefecture_is_read_off_the_name_rather_than_listed(
    name: str, expected: str | None
) -> None:
    """Three or four characters ending in 都道府県 -- so a list that could go
    stale is not needed, and a name carrying no prefecture says so."""
    assert without_prefecture(name) == expected
