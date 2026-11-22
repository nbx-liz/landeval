"""What `data/processed/` promises: which tables exist and what identifies a row.

This is the SSOT the schema contract pins. Every table is declared with the
columns that identify one of its rows, and sits in exactly one of two sets that
say where its checked copy is read from (`contract_path`; the caller names both
directories):

- **`TRACKED_TABLES`**: the checked copy is the table itself, in the processed
  directory; and
- **`SAMPLED_TABLES`**, the rest: the checked copy is a `SAMPLE_ROWS`-row sample
  of the table, taken with `sample_of`, in the fixture directory.

A table is never in both: a full copy and a sample of one table would give the
contract two sources to disagree about.

**How this package validates the contract.** Its tests run on synthetic frames,
not on stored data: the two sets partition the declared tables, `contract_path`
follows that split, `sample_of` keeps the columns, the dtypes and a unique,
non-null key, and every declared table, built synthetically from its expected
column order and dtypes, keeps them through a parquet round trip. Checking
stored tables against this declaration is left to whoever holds them.

**Why the keys are declared here rather than discovered.** Probing the data
for "which column combination happens to be unique" answers a different question:
a table with one row per municipality and one measured schedule has `mesh_id`
alone unique for as long as that lasts. Declaring that as the key would pin a
property of the current input distribution rather than of the design, and the
pin would hold until a second schedule is measured -- at which point the table changes meaning
and nothing fails. Every table carrying a `schedule_id` therefore names it in the
key, whether or not it currently holds more than one value.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Rows kept per sample. Large enough that a dtype survives the round trip and a
# duplicate key would plausibly appear, small enough that a sample stays a small
# file. The count of tables is
# deliberately not written here: it moves whenever one is added or retired, and
# a comment restating a derived number goes stale silently.
SAMPLE_ROWS = 300

# Every parquet under `data/processed/`, mapped to the columns that identify one
# row. Two of them (`stability_check`, `stability_control`) are stability evidence
# rather than pipeline output; they are declared all the same, so that their
# schema is under the same contract as every other table.
TABLE_KEYS: dict[str, tuple[str, ...]] = {
    # Cell tables: one row per 500 m mesh, at each stage of the funnel.
    "combined_cells": ("mesh_id",),
    "final_cells": ("mesh_id",),
    "freeflow_all": ("mesh_id",),
    "freeflow_horizon": ("mesh_id",),
    "mesh_hazard": ("mesh_id",),
    "screened_cells": ("mesh_id",),
    "selected_cells": ("mesh_id",),
    # Which municipality each habitable cell is in, for every cell
    # rather than only those inside the horizon. Per-municipality
    # denominators are taken from here.
    "cell_municipality": ("mesh_id",),
    # Blocks and representatives: one row per sampled point.
    "blocks": ("mesh_id",),
    "representatives": ("mesh_id",),
    # Measurements. Long format -- one row per point per schedule.
    "block_measurements": ("mesh_id", "schedule_id"),
    "calibration": ("mesh_id", "schedule_id"),
    "municipality_measured": ("mesh_id", "schedule_id"),
    "municipality_rep_target_times": ("mesh_id", "schedule_id"),
    "municipality_rep_schedule_times": ("mesh_id", "schedule_id"),
    "block_rep_target_times": ("mesh_id", "schedule_id"),
    "block_rep_afternoon_outbound_times": ("mesh_id", "schedule_id"),
    "block_rep_schedule_times": ("mesh_id", "schedule_id"),
    # Group destinations. The only key that names a destination:
    # one group's points share one `schedule_id`, so (mesh, schedule) is
    # not unique here the way it is for the single-destination commute.
    "group_destination_times": ("mesh_id", "schedule_id", "destination_name"),
    # Sheets and summaries.
    "commute_sheet": ("mesh_id",),
    # One row per municipality, ranked by the share of its
    # habitable land that is usable.
    "areas_sheet": ("muni_code",),
    # Provenance: one row per cache entry, keyed by the digest itself.
    "fetch_provenance": ("cache_digest",),
    # Stability evidence: one row per schedule per quantile.
    "stability_check": ("mesh_id", "schedule_id", "quantile"),
    "stability_control": ("mesh_id", "schedule_id", "quantile"),
}

# Only tables the pipeline produces are declared. A declaration of a name that no
# script writes and no module reads could never be satisfied: a correct gate
# reading a declaration no real input can meet.

# Declared before the artefact exists. A name here is keyed and reviewed but no
# pipeline step writes it yet, so it stays out of `TABLE_KEYS`: every entry there
# is a table the pipeline produces, and whoever checks stored tables against
# `TABLE_KEYS` would see a name with no table fail for a reason that has
# nothing to do with the data.
PLANNED_TABLE_KEYS: dict[str, tuple[str, ...]] = {}


# The tables whose checked copy is the table itself rather than a sample (see
# `contract_path`). Every other declared table is in `SAMPLED_TABLES`; the
# synthetic contract tests check that the two sets partition `TABLE_KEYS`.
TRACKED_TABLES = frozenset(
    {
        "block_measurements",
        "calibration",
        "commute_sheet",
        "fetch_provenance",
        "group_destination_times",
        "stability_check",
        "stability_control",
        "municipality_rep_target_times",
        "municipality_rep_schedule_times",
        "block_rep_target_times",
        "block_rep_afternoon_outbound_times",
        "block_rep_schedule_times",
    }
)

SAMPLED_TABLES = frozenset(TABLE_KEYS) - TRACKED_TABLES


def contract_path(name: str, *, processed_dir: Path, fixture_dir: Path) -> Path:
    """Where the checked copy of ``name`` is read from: the table itself, or its sample.

    The caller names both directories; the library does not know where they are.
    """
    if name in TRACKED_TABLES:
        return processed_dir / f"{name}.parquet"
    return fixture_dir / f"{name}.parquet"


def column_types(frame: pd.DataFrame) -> dict[str, str]:
    """The frame's columns and dtypes, in order, as the manifest records them."""
    return {str(column): str(frame[column].dtype) for column in frame.columns}


def sample_of(frame: pd.DataFrame, key: tuple[str, ...], rows: int = SAMPLE_ROWS) -> pd.DataFrame:
    """A deterministic subset of ``frame``, spread across its key order.

    Taking the head instead would sample one corner of the country: `mesh_id`
    sorts geographically, so the first 300 rows of a large table are one
    cluster of cells with one municipality's null pattern. An even stride costs
    nothing and gives the dtype checks values from the whole range.
    """
    ordered = frame.sort_values(list(key), kind="stable").reset_index(drop=True)
    if len(ordered) <= rows:
        return ordered
    stride = len(ordered) // rows
    return ordered.iloc[::stride].head(rows).reset_index(drop=True)
