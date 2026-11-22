"""Attaching municipalities to mesh cells, from MLIT N03 boundaries.

The municipality-level measurement uses one representative point per
municipality, so a cell has to know which municipality it belongs to before
those points can be chosen.

A cell whose centre falls outside every polygon keeps a null municipality
rather than being attached to the nearest one. Snapping would silently move a
coastal or border cell into a municipality it is not in, and the count of
unmatched cells is itself the signal that the boundary set is incomplete.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

# N03 attribute names. N03_007 is the 5-digit municipality code; the wards of
# the designated cities each carry their own. N03_005 holds the ward name and is
# null for everywhere else, so it is required to be present but not populated:
# a vintage that renames it would otherwise collapse every ward code of a
# designated city onto the city's name without any error.
PREF_COLUMN = "N03_001"
COUNTY_COLUMN = "N03_003"
CITY_COLUMN = "N03_004"
WARD_COLUMN = "N03_005"
CODE_COLUMN = "N03_007"

WGS84 = "EPSG:4326"


class AdminError(Exception):
    """Raised when the boundary set cannot be loaded or is missing a column."""


def load_boundaries(directory: Path) -> gpd.GeoDataFrame:
    """Load every N03 prefecture archive in ``directory`` into one frame."""
    archives = sorted(directory.glob("*_GML.zip"))
    if not archives:
        raise AdminError(f"no N03 archives found in {directory}; run scripts/fetch_admin.py")

    frames = []
    for archive in archives:
        frame = gpd.read_file(f"zip://{archive}")
        missing = {PREF_COLUMN, CITY_COLUMN, WARD_COLUMN, CODE_COLUMN} - set(frame.columns)
        if missing:
            raise AdminError(f"{archive.name} is missing columns {sorted(missing)}")
        frames.append(frame)

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return merged.to_crs(WGS84)


def municipality_name(row: pd.Series) -> str:
    """Human-readable name of the unit the code identifies.

    The county is kept for towns and villages, because several 町 in different
    counties share a name. The ward is kept for the designated cities, because
    each ward has its own code and its own entry in a property search -- the
    city name alone would name one place where the codes name several.

    N03_003 carries the county for towns but the city name for designated
    cities, so only a value ending in 郡 is treated as a county.
    """
    parts = [row[PREF_COLUMN]]
    for column in (COUNTY_COLUMN, CITY_COLUMN, WARD_COLUMN):
        value = row.get(column)
        if not isinstance(value, str) or not value:
            continue
        if column is COUNTY_COLUMN and not value.endswith("郡"):
            continue
        parts.append(value)
    return "".join(parts)


def municipality_aliases(row: pd.Series) -> tuple[str, ...]:
    """Every spelling of this unit that an address may legitimately arrive in.

    The GSI geocoder omits the county: 〇〇県〇〇郡〇〇町 comes back as
    〇〇県〇〇町. Matching an answer against the canonical name alone rejects
    that as "no municipality in the reference set", which is a false negative
    dressed as a strict check -- the address resolved fine, only the spelling
    differs. Both forms are offered and both mean the same code.
    """
    canonical = municipality_name(row)
    county = row.get(COUNTY_COLUMN)
    if isinstance(county, str) and county.endswith("郡"):
        return (canonical, canonical.replace(county, "", 1))
    return (canonical,)


PREFECTURE_SUFFIX = "都道府県"


def without_prefecture(name: str) -> str | None:
    """``name`` with its prefecture removed, or None when it carries none.

    Prefecture names are three or four characters and all of them end in one of
    都道府県, so the cut is read off the name rather than kept in a list that
    could go stale.
    """
    for cut in (3, 4):
        if len(name) > cut and name[cut - 1] in PREFECTURE_SUFFIX:
            return name[cut:]
    return None


def alias_map(boundaries: gpd.GeoDataFrame) -> dict[str, str]:
    """Every accepted spelling of every municipality, mapped to its canonical name.

    ``municipality_aliases`` alone yields no prefecture-less spelling. Without
    one, a user typing an address such as 〇〇市〇〇町1-2 matches no municipality
    on the query side, and the consequences are both halves of a false-clean failure -- the plot
    is refused even though the geocoder resolved it in full, and the check that
    refuses an answer in a *different* municipality silently switches itself off
    for exactly the queries that most need it.

    So the prefecture-less form is offered too, and only where it is
    unambiguous across the whole set. A bare name that more than one
    municipality claims (village names recur across prefectures, and so does
    N03's 所属未定地) keeps requiring the prefecture rather than resolving to
    whichever row was read first. Ambiguity is decided over the finished set,
    which is why this cannot live in ``municipality_aliases`` -- one row cannot
    see the others.
    """
    canonical_of: dict[str, str] = {}
    claimants: dict[str, set[str]] = {}
    for _, row in boundaries.iterrows():
        canonical = municipality_name(row)
        for alias in municipality_aliases(row):
            canonical_of[alias] = canonical
            bare = without_prefecture(alias)
            if bare:
                claimants.setdefault(bare, set()).add(canonical)
    for bare, owners in claimants.items():
        if len(owners) == 1 and bare not in canonical_of:
            canonical_of[bare] = next(iter(owners))
    return canonical_of


def attach_municipality(cells: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return ``cells`` with ``muni_code`` and ``muni_name`` columns.

    Cells outside every polygon keep nulls; they are not snapped to a neighbour.
    """
    points = gpd.GeoDataFrame(
        cells.copy(),
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs=WGS84,
    )
    named = boundaries.assign(
        muni_code=boundaries[CODE_COLUMN],
        muni_name=boundaries.apply(municipality_name, axis=1),
    )[["muni_code", "muni_name", "geometry"]]

    joined = gpd.sjoin(points, named, how="left", predicate="within")
    # A cell on a shared border can match more than one polygon; keep the first
    # deterministically rather than dropping or duplicating the cell.
    joined = joined[~joined.index.duplicated(keep="first")]
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def summarize_by_municipality(cells: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """One row per municipality: cell count and the distribution of ``value_column``.

    ``best`` is the minimum, which is what the municipality-level measurement uses: a
    municipality is only ruled out when even its best-placed cell fails.
    """
    grouped = cells.dropna(subset=["muni_code"]).groupby(["muni_code", "muni_name"])
    summary = grouped[value_column].agg(cells="size", best="min", median="median", worst="max")
    return summary.reset_index().sort_values("best").reset_index(drop=True)
