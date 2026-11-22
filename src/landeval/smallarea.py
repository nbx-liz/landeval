"""The 町丁・字等 boundaries, read from e-Stat's archives.

N03 says which municipality a point is in. This says which *town* -- and that is
the difference between 「〇〇区」 and 「〇〇区〇〇町三丁目」 for an address the
geocoder could not resolve to a street number.

Three things about the files decide the shape of this module:

- **the DBF is CP932.** Reading it as UTF-8 raises; reading it as something else
  plausible does not raise, it returns mojibake. So the encoding is stated and
  a file that will not decode is refused, rather than detected and hoped for.
- **`KEY_CODE` is not the key of the file.** There are more rows than codes,
  because a town split by a river is two polygons. Anything keyed on it before
  the parts are merged counts polygons and calls them places.
- **some rows are not places.** Some have no name, and some are water-survey
  districts. They are dropped, and how many were dropped travels with the
  result -- a silent drop makes "some were excluded" look exactly like "there
  were none".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

# e-Stat ships Shift-JIS. Stated rather than sniffed: the failure mode of a
# wrong guess is a wrong name, not an exception.
ENCODING = "cp932"

# 8154 is a water-survey district: a real polygon, not a place to live.
WATER_HCODE = 8154

# The columns this module promises its callers. The archives carry thirty.
COLUMNS = (
    "KEY_CODE",
    "PREF",
    "CITY",
    "PREF_NAME",
    "CITY_NAME",
    "S_NAME",
    "AREA",
    "JINKO",
    "geometry",
)

# The column this module adds. `CITY` is three digits *within a prefecture* --
# 水戸市 and 千代田区 are both "201" and "101" style codes -- so joining on it
# alone merges towns from different prefectures. N03 keys on the five-digit
# code, and so does everything downstream, so the join key is built here once
# rather than at each call site.
MUNI_CODE = "muni_code"


class SmallAreaError(Exception):
    """Raised for a missing archive, an undecodable one, or a missing column."""


@dataclass(frozen=True)
class SmallAreas:
    """Town-level polygons, and what was left out of them."""

    frame: gpd.GeoDataFrame
    unnamed: int
    water: int
    parts_merged: int

    @classmethod
    def load_all(cls, directory: Path, prefectures: tuple[str, ...]) -> SmallAreas:
        """Every named prefecture, in one frame, with the exclusions summed."""
        loaded = [load_small_areas(archive_for(directory, code)) for code in prefectures]
        if not loaded:
            raise SmallAreaError("no prefectures requested")
        merged = pd.concat([areas.frame for areas in loaded], ignore_index=True)
        return cls(
            frame=gpd.GeoDataFrame(merged, geometry="geometry", crs=loaded[0].frame.crs),
            unnamed=sum(areas.unnamed for areas in loaded),
            water=sum(areas.water for areas in loaded),
            parts_merged=sum(areas.parts_merged for areas in loaded),
        )


def archive_for(directory: Path, prefecture: str) -> Path:
    """The archive holding one prefecture, or an error naming the missing one.

    An absent prefecture means the download did not finish. Returning an empty
    frame instead would make a partial fetch look like a region with no towns,
    which is the reading that must be prevented.
    """
    archive = directory / f"r2ka{prefecture}.zip"
    if not archive.is_file():
        raise SmallAreaError(
            f"no small-area archive for prefecture {prefecture} in {directory}; "
            "run scripts/fetch_estat.py"
        )
    return archive


def load_small_areas(archive: Path) -> SmallAreas:
    """Read one prefecture's archive into one row per 町丁字."""
    # No try/except around the encoding. There is nothing to catch: CP932
    # decodes almost any byte sequence, so a wrong encoding does not raise -- it
    # returns mojibake where the town name was. A handler
    # here would be a branch that never runs, and its presence would suggest the
    # case is covered. What protects the names is the argument below,
    # pinned by `test_an_archive_without_a_cpg_still_decodes`.
    raw = gpd.read_file(f"zip://{archive}", encoding=ENCODING)

    missing = [column for column in COLUMNS if column not in raw.columns]
    if missing:
        raise SmallAreaError(f"{archive} lacks {missing}; it has {sorted(raw.columns)}")

    # Names arriving as bytes-shaped nonsense is the failure this catches: the
    # read succeeded, so only the content can say whether it was decoded right.
    named = raw[raw["S_NAME"].notna()]
    unnamed = len(raw) - len(named)

    dry = named[named["HCODE"] != WATER_HCODE]
    water = len(named) - len(dry)

    kept = gpd.GeoDataFrame(dry[list(COLUMNS)], geometry="geometry", crs=raw.crs)
    kept[MUNI_CODE] = kept["PREF"].astype(str).str.zfill(2) + kept["CITY"].astype(str).str.zfill(3)
    if kept.empty:
        return SmallAreas(frame=kept, unnamed=unnamed, water=water, parts_merged=0)

    # One town, one row. `dissolve` unions the parts and keeps the first of the
    # other columns, which are identical across parts of the same KEY_CODE.
    merged = kept.dissolve(by="KEY_CODE", aggfunc="first").reset_index()
    return SmallAreas(
        frame=gpd.GeoDataFrame(merged, geometry="geometry", crs=raw.crs),
        unnamed=unnamed,
        water=water,
        parts_merged=len(kept) - len(merged),
    )
