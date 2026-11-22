"""Choosing one measurable point to stand for each municipality.

The municipality-level measurement uses a single point per municipality, so the choice decides what
the measurement means. Two rules follow from that.

Ground outside every designated hazard zone comes first, ahead of a shorter
commute -- a point in the middle of an inundation zone would report the travel
time to somewhere nobody in this search would buy. Where a municipality has no
such ground the least-covered cell is used instead and the fallback is recorded
in ``basis``, because "the best clear point" and "the least bad point" are
different claims and only one of them should be read as an endorsement.

The order is then fully determined, down to the tie-break on ``mesh_id``. The
measurement cache is keyed on the coordinate, so a rule that reshuffled between
runs would re-bill every point it moved.
"""

from __future__ import annotations

import pandas as pd

BASIS_CLEAR = "hazard_clear"
BASIS_LEAST_HAZARD = "least_hazard"

DEFAULT_ADOPTED = ("pass", "hold")

REQUIRED_COLUMNS = (
    "mesh_id",
    "muni_code",
    "muni_name",
    "screening_class",
    "selected_hazard_ratio",
    "predicted_min",
    "lat",
    "lon",
    "freeflow_min",
    "sector",
)


def choose_representatives(
    cells: pd.DataFrame,
    *,
    group_column: str = "muni_code",
    adopted: tuple[str, ...] = DEFAULT_ADOPTED,
    hazard_column: str = "selected_hazard_ratio",
    value_column: str = "predicted_min",
) -> pd.DataFrame:
    """One row per group: the point the measurement should use.

    ``group_column`` is the unit being represented -- a municipality, or a
    coarse grid block. The rule is the same either way, because the
    question is the same: what is the best point in here that stands on ground
    the hazard layers do not cover.

    Cells the transport filter removed are not eligible, and cells with no group
    are dropped -- they have nothing to represent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in cells.columns]
    if missing:
        raise KeyError(f"cells is missing {missing}")
    if group_column not in cells.columns:
        raise KeyError(f"cells is missing {[group_column]}")

    eligible = cells[cells["screening_class"].isin(adopted)].dropna(subset=[group_column])
    if eligible.empty:
        return pd.DataFrame(
            columns=[*REQUIRED_COLUMNS, group_column, "basis", "adopted_cells", "clear_cells"]
        )

    counts = eligible.groupby(group_column, observed=True).agg(
        adopted_cells=("mesh_id", "size"),
        clear_cells=(hazard_column, lambda values: int((values <= 0).sum())),
    )

    # Least hazard first, then the shortest commute, then the code -- the last
    # only so that two identical candidates always resolve the same way.
    ordered = eligible.sort_values(
        [hazard_column, value_column, "mesh_id"], ascending=True, kind="mergesort"
    )
    chosen = ordered.groupby(group_column, observed=True, as_index=False).head(1)

    chosen = chosen.merge(counts, left_on=group_column, right_index=True, how="left")
    chosen = chosen.assign(
        basis=chosen[hazard_column].map(
            lambda ratio: BASIS_CLEAR if ratio <= 0 else BASIS_LEAST_HAZARD
        )
    )
    return chosen.sort_values(group_column).reset_index(drop=True)
