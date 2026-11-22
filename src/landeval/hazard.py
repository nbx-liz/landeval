"""Overlaying the published hazard areas onto the mesh.

A 500 m cell is 0.25 km2 and routinely straddles the edge of an inundation
zone, so a centre-point test would either delete ground that is fine or keep
ground that is not. Each cell is therefore sampled on a 5x5 sub-grid and the
result is a *fraction*, which is also what lets the exclusion threshold stay a
display-time parameter rather than something baked into the data.

Nothing here is allowed to say "safe". The published polygons cover designated
rivers, designated slopes and designated coastlines; ground outside them is
ground outside them, which is a statement about the survey rather than about
the ground. The vocabulary is `outside` / `risk...` / `not_designated`
/ `no_data`, and a test asserts the word "safe" never appears among them.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely import STRtree
from shapely import points as shapely_points

from landeval.mesh import mesh_step_deg

# Sub-grid resolution per cell edge. 5 gives 25 samples of a 500 m cell, i.e.
# 100 m spacing and a fraction resolved to 4%, which is the granularity the
# regional question needs; anything finer belongs to the per-property lookup.
SUBGRID = 5

# Status vocabulary. `OUTSIDE` deliberately does not claim safety.
OUTSIDE = "outside"
NOT_DESIGNATED = "not_designated"
NO_DATA = "no_data"

# Severity sentinel for a layer that designates areas but publishes no depth
# (家屋倒壊等氾濫想定区域, 土砂災害警戒区域). Negative so it is never mistaken
# for a measured depth, but not NaN, which is reserved for "outside".
HIT_WITHOUT_DEPTH = -1.0

# 浸水深ランクコード (A31b_101 / A31b_201), mapped to the lower bound of each
# band in metres. Verified against the MLIT code list for the 2025 vintage.
DEPTH_RANK_LOWER_M: dict[int, float] = {
    1: 0.0,  # 0m以上0.5m未満
    2: 0.5,  # 0.5m以上3.0m未満
    3: 3.0,  # 3.0m以上5.0m未満
    4: 5.0,  # 5.0m以上10.0m未満
    5: 10.0,  # 10.0m以上20.0m未満
    6: 20.0,  # 20.0m以上
}

# A33_002 区域区分. 2 and 4 are 特別警戒区域 (red); 3 and 4 are survey results
# published ahead of formal designation, which are still hazard.
LANDSLIDE_SPECIAL_CODES = frozenset({2, 4})

_NUMBER = r"\d+(?:\.\d+)?"
_BETWEEN = re.compile(rf"^({_NUMBER})m以上({_NUMBER})m未満$")
_ABOVE = re.compile(rf"^({_NUMBER})m以上$")
_BELOW = re.compile(rf"^({_NUMBER})m未満$")


class HazardDataError(Exception):
    """Raised when source data does not match the vocabulary it is supposed to use."""


def depth_from_rank_code(code: object) -> float:
    """Lower bound in metres of an A31b depth rank code.

    An unrecognised code raises. Mapping it to zero would put deep water into
    the shallowest band and mapping it to "outside" would erase it entirely.
    """
    try:
        key = int(str(code).strip())
    except (TypeError, ValueError) as error:
        raise HazardDataError(f"unknown depth rank code {code!r}") from error
    if key not in DEPTH_RANK_LOWER_M:
        raise HazardDataError(
            f"unknown depth rank code {code!r}; expected one of {sorted(DEPTH_RANK_LOWER_M)}"
        )
    return DEPTH_RANK_LOWER_M[key]


def depth_from_range_text(text: object) -> float:
    """Lower bound in metres of a depth range written as text.

    Each prefecture writes its own 津波浸水想定 and 高潮浸水想定区域, so the same
    band is spelled several ways across the published sources:

        0.5m以上1m未満        0.5m以上 1.0m未満      0.3m以上 ～ 1m未満
        20.0m以上             0.3m未満               ～0.3m未満

    Those shapes form a closed set: the separators are normalised away and the
    result is matched against it. ``scripts/survey_depth_vocabulary.py`` lists
    every distinct value in a set of source files, so a new vintage can be
    checked against the set before it is read. Anything outside it raises rather than widening
    the parser one string at a time -- a range the parser cannot read is not an
    absence of water.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        raise HazardDataError("cannot read depth range from a missing value")
    normalised = unicodedata.normalize("NFKC", str(text))
    # Whitespace and wave dashes are separators here and nothing else; removing
    # them collapses the spellings without changing which numbers are present.
    normalised = re.sub(r"[\s~〜～]+", "", normalised)
    for pattern in (_BETWEEN, _ABOVE):
        match = pattern.match(normalised)
        if match:
            return float(match.group(1))
    if _BELOW.match(normalised):
        return 0.0
    raise HazardDataError(f"cannot read depth range from {text!r}")


def subcell_lonlat(
    cells: pd.DataFrame, *, mesh_size_m: int, subgrid: int = SUBGRID
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Sample points for every cell, as (lon, lat, cell index).

    Points sit at the centres of the sub-squares, so they stay strictly inside
    the cell and no point is shared with a neighbour.
    """
    if subgrid < 1:
        raise HazardDataError(f"subgrid must be at least 1, got {subgrid}")
    lat_step, lon_step = mesh_step_deg(mesh_size_m)
    offsets = (np.arange(subgrid, dtype=np.float64) + 0.5) / subgrid - 0.5

    centre_lat = cells["lat"].to_numpy(dtype=np.float64)
    centre_lon = cells["lon"].to_numpy(dtype=np.float64)

    lat_grid, lon_grid = np.meshgrid(offsets * lat_step, offsets * lon_step, indexing="ij")
    lat = (centre_lat[:, None] + lat_grid.ravel()[None, :]).ravel()
    lon = (centre_lon[:, None] + lon_grid.ravel()[None, :]).ravel()
    index = np.repeat(np.arange(len(cells), dtype=np.int64), subgrid**2)
    return lon, lat, index


class PointIndex:
    """A spatial index over the sample points, built once and reused.

    The index goes over the points rather than the polygons because the points
    are fixed for the whole run while the polygons arrive in chunks -- a single
    source file can hold a very large number of them and cannot be held in
    memory at once. Inverting the query this way means one tree for every layer
    instead of one tree per chunk.
    """

    def __init__(self, lon: NDArray[np.float64], lat: NDArray[np.float64]) -> None:
        if len(lon) != len(lat):
            raise HazardDataError(f"got {len(lon)} longitudes but {len(lat)} latitudes")
        self.size = len(lon)
        self._tree = STRtree(shapely_points(lon, lat))

    def empty_severity(self) -> NDArray[np.float64]:
        """A fresh accumulator, NaN meaning 'no polygon covers this point'."""
        return np.full(self.size, np.nan, dtype=np.float64)

    def accumulate(
        self, geometries: Any, severities: NDArray[np.float64], out: NDArray[np.float64]
    ) -> int:
        """Fold one batch of polygons into ``out``, keeping the worst severity.

        ``intersects`` rather than ``within`` so a point exactly on a boundary
        counts as inside -- the direction that keeps hazard visible. Returns
        the number of point/polygon hits, which the caller logs so a layer that
        silently matched nothing is visible rather than indistinguishable from
        a layer with no hazard in range.
        """
        geometry_list = list(geometries)
        if len(geometry_list) != len(severities):
            raise HazardDataError(
                f"got {len(geometry_list)} geometries but {len(severities)} severities"
            )
        if out.shape != (self.size,):
            raise HazardDataError(f"accumulator has shape {out.shape}, expected ({self.size},)")
        if not geometry_list:
            return 0

        geometry_index, point_index = self._tree.query(geometry_list, predicate="intersects")
        if not len(point_index):
            return 0
        np.fmax.at(out, point_index, np.asarray(severities, dtype=np.float64)[geometry_index])
        return len(point_index)


def max_severity_at_points(
    lon: NDArray[np.float64],
    lat: NDArray[np.float64],
    geometries: Any,
    severities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Worst severity covering each point; NaN where no polygon covers it."""
    index = PointIndex(lon, lat)
    result = index.empty_severity()
    index.accumulate(geometries, severities, result)
    return result


def summarise_layer(
    cell_index: NDArray[np.int64],
    severity: NDArray[np.float64],
    *,
    n_cells: int,
    points_per_cell: int,
) -> pd.DataFrame:
    """Per-cell hazard fraction and worst depth.

    A cell that received no sample points is NaN rather than zero: never having
    looked is not the same as having looked and found nothing.
    """
    sampled = np.bincount(cell_index, minlength=n_cells)[:n_cells]
    inside = np.isfinite(severity)
    hits = np.bincount(cell_index, weights=inside.astype(np.float64), minlength=n_cells)[:n_cells]

    ratio = np.divide(hits, float(points_per_cell), out=np.full(n_cells, np.nan), where=sampled > 0)

    depth = np.full(n_cells, np.nan, dtype=np.float64)
    with_depth = severity >= 0.0
    if with_depth.any():
        np.fmax.at(depth, cell_index[with_depth], severity[with_depth])

    return pd.DataFrame({"hazard_ratio": ratio, "max_depth_m": depth})


def union_ratio(
    mask: NDArray[np.unsignedinteger[Any]],
    bits: int,
    cell_index: NDArray[np.int64],
    *,
    n_cells: int,
    points_per_cell: int,
    unknown: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Per-cell fraction covered by *any* of the layers selected in ``bits``.

    A union over a subset of layers cannot be recovered from the per-layer
    fractions -- two layers covering the same half of a cell give 0.5, not 1.0 --
    so the per-point membership is kept and the union is taken here. That makes
    the choice of which layers to apply a display-time parameter, the same
    status the thresholds have.

    ``unknown`` marks cells for which at least one selected layer was never
    obtained, and those come back NaN unless every sample point of the cell is
    already inside a zone of the layers that were read -- such a cell is 1.0,
    because an unread layer cannot make it any more covered. It is
    required rather than optional because a caller that forgets it would get
    the false-clean answer back silently, which is the shape of gate that
    disables itself when an input is absent. The mask
    records where polygons *were* found and has no way to distinguish a cell
    that was checked and found dry from one that was never covered by the
    source at all -- a cell outside every prefecture the tsunami layer was
    fetched for sets no bit, exactly like dry ground. Folding the second into
    the first reports "no hazard here" about ground nobody looked at, which is
    what must never happen; :func:`layer_status` takes the same care per layer.
    """
    if bits == 0:
        raise HazardDataError("no layers selected")
    if len(mask) != len(cell_index):
        raise HazardDataError(f"got {len(mask)} mask entries but {len(cell_index)} cell indices")
    if len(unknown) != n_cells:
        raise HazardDataError(f"got {len(unknown)} unknown flags but {n_cells} cells")
    hit = (mask & np.uint64(bits)) != 0
    ratio = summarise_layer(
        cell_index,
        np.where(hit, 0.0, np.nan),
        n_cells=n_cells,
        points_per_cell=points_per_cell,
    )["hazard_ratio"].to_numpy()
    # One exception: a cell whose every sample point is inside a zone of a
    # layer that *was* read is wholly covered, whatever the unread layer would
    # have added. Its points cannot become any more covered, so the ratio is
    # an answer rather than a lower bound.
    hits = np.bincount(cell_index, weights=hit.astype(np.float64), minlength=n_cells)[:n_cells]
    whole = hits >= points_per_cell
    return np.where(unknown & ~whole, np.nan, ratio)


STATUS_SUFFIX = "_status"


def status_layers(hazard: pd.DataFrame) -> list[str]:
    """Every layer the frame carries a status column for, in column order."""
    names = [c[: -len(STATUS_SUFFIX)] for c in hazard.columns if c.endswith(STATUS_SUFFIX)]
    if not names:
        raise HazardDataError(
            f"frame carries no *{STATUS_SUFFIX} column, so which layers were obtained "
            "cannot be determined; regenerate it with scripts/build_hazard.py"
        )
    return names


APPLIED_LAYERS_COLUMN = "applied_layers"
_APPLIED_SEPARATOR = "|"


def record_applied_layers(layers: Sequence[str]) -> str:
    """Encode which layers a selection was made over, for storage beside it."""
    if len(layers) == 0:
        raise HazardDataError("no layers selected; a selection over no layers is vacuously clean")
    return _APPLIED_SEPARATOR.join(layers)


def applied_layers(cells: pd.DataFrame) -> tuple[str, ...]:
    """Which hazard layers the stored selection was actually made over.

    Which layers to apply is a display-time choice, made by whatever builds
    the selection. A downstream script that hardcoded its own copy
    of the default would disagree with the others about the same cells as soon
    as the documented ``--layer`` option was used, without anything failing.

    Reading it from the frame raises when the column is absent, rather than
    falling back to a guess: a parquet written before the choice was recorded
    cannot tell us what it was, and assuming the default is how a selection
    made over other layers would be read as if it were the usual one.
    """
    if APPLIED_LAYERS_COLUMN not in cells.columns:
        raise HazardDataError(
            f"{APPLIED_LAYERS_COLUMN} is absent, so which hazard layers this selection "
            "was made over is unknown; rebuild the selection so that it records them"
        )
    distinct = set(cells[APPLIED_LAYERS_COLUMN].dropna().unique())
    if len(distinct) != 1:
        raise HazardDataError(
            f"{APPLIED_LAYERS_COLUMN} is not constant across the frame: {sorted(distinct)}"
        )
    return tuple(distinct.pop().split(_APPLIED_SEPARATOR))


def covered_column(layer: str) -> str:
    """The name of a layer's coverage column: True where the layer was read."""
    return f"{layer}_covered"


def _read_flags(hazard: pd.DataFrame, name: str) -> NDArray[np.bool_]:
    """One layer's per-cell coverage, refusing a frame that cannot say.

    The status column must be there and must not be null: a null status arrives
    when a cell has no row in the hazard table at all -- a left merge that did
    not match -- and reading it as anything is the original fault one level up.
    The coverage column must be there as well. A status of ``risk`` comes from a
    real intersection even where the layer was not read, so the status alone
    cannot tell read ground from unread ground; a table written before the
    coverage column existed is refused rather than guessed at.
    """
    column = f"{name}{STATUS_SUFFIX}"
    if column not in hazard.columns:
        raise HazardDataError(
            f"{column} is absent, so whether {name} was obtained cannot be told from "
            f"whether it found nothing; frame has {sorted(status_layers(hazard))}"
        )
    column_values = hazard[column]
    if column_values.isna().any():
        raise HazardDataError(
            f"{column} is null for {int(column_values.isna().sum())} cells; "
            "the hazard table does not cover them, so their status is unknown "
            "rather than clear"
        )
    flag = covered_column(name)
    if flag not in hazard.columns:
        raise HazardDataError(
            f"{flag} is absent, so which cells {name} was read for is unknown; this "
            "table predates the coverage column -- rebuild it with scripts/build_hazard.py"
        )
    flags = hazard[flag]
    if flags.isna().any():
        raise HazardDataError(
            f"{flag} is null for {int(flags.isna().sum())} cells; whether the layer was "
            "read there is unknown rather than yes"
        )
    if not pd.api.types.is_bool_dtype(flags):
        raise HazardDataError(f"{flag} must hold booleans, got {flags.dtype}")
    return flags.to_numpy(dtype=bool)


def split_by_coverage(hazard: pd.DataFrame, layers: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``layers`` into those read for at least one cell, and those for none.

    Two absences look alike and are not alike. A layer read for some cells and
    not this one is a gap *in this cell*: the neighbour got an answer and it
    did not, and only the cell-level flag can say so. A layer read for no cell
    at all is a statement about the whole map, and flagging every cell for it
    marks all of them as unanswerable, which distinguishes nothing and hides
    the first kind completely.

    The split is on the coverage column, not on the statuses and not on why a
    layer is absent: a layer that was not read anywhere can still carry a
    ``risk`` status where a polygon reached a cell. A layer can be unread
    everywhere because it was never ingested: a 404 tile cannot distinguish
    "this municipality does not publish" from "it publishes and this point is
    outside the zone", and telling those apart needs the roster of
    publishing municipalities, which is the condition set on ingesting it.

    Both have to be reported; they must not be reported as the same thing. The
    caller states the second once, in words, and passes only the first to
    :func:`unobtained_mask`.
    """
    covered: list[str] = []
    absent: list[str] = []
    for name in layers:
        (covered if _read_flags(hazard, name).any() else absent).append(name)
    return covered, absent


def unobtained_mask(hazard: pd.DataFrame, layers: Sequence[str]) -> NDArray[np.bool_]:
    """Per-cell flag: at least one of ``layers`` was not read for that cell.

    This is the rule that separates "looked and found nothing" from "never
    looked", and it belongs in one place because a call site that re-derives it
    can omit it -- and then a block classification paints unlooked-at ground as
    candidate area, or ``hazard_free_ratio`` reads 1.00 for a cell with an
    unread layer. Both are the same false-clean answer.

    It reads each layer's coverage column (see :func:`covered_column`). A layer
    without one, or without a status column, raises rather than being treated
    as read: a question the frame cannot answer must not come back as "no
    problem". Passing no layers raises for the same reason -- a union over
    nothing is vacuously clean.
    """
    if len(layers) == 0:
        raise HazardDataError("no layers selected; a union over no layers is vacuously clean")
    mask = np.zeros(len(hazard), dtype=bool)
    for name in layers:
        mask |= ~_read_flags(hazard, name)
    return mask


def coverage_class(ratio: float) -> str:
    """Split a cell into none / partly / wholly covered.

    ``partly`` is deliberately its own outcome rather than being folded into
    either neighbour: a cell that is half designated still contains ground that
    is not, which is a question for a site visit and not something a 500 m grid
    can answer either way.
    """
    if math.isnan(ratio):
        return NO_DATA
    if ratio <= 0.0:
        return "clear"
    if ratio >= 1.0:
        return "covered"
    return "partial"


def layer_status(
    hazard_ratio: float, max_depth_m: float, *, available: bool, designated: bool
) -> str:
    """Classify one cell for one layer.

    The order matters: a layer that could not be obtained is `no_data` whatever
    the geometry says, because the absence of polygons in that case carries no
    information at all.
    """
    if not available:
        return NO_DATA
    if not designated:
        return NOT_DESIGNATED
    if math.isnan(hazard_ratio):
        return NO_DATA
    if hazard_ratio <= 0.0:
        return OUTSIDE
    if math.isnan(max_depth_m):
        return "risk"
    return f"risk_{max_depth_m:.1f}m"
