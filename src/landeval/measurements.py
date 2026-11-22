"""Merging measurement vintages into one table.

Tables written before the ``direction`` column existed carry no ``direction``
column; later rows carry one. Stacking them needs a rule for the gap, and
the rule cannot be "assume outbound" -- assuming outbound is precisely the
mistake the ``direction`` column exists to prevent.

So the configuration's declaration is the authority, and a stored value that
disagrees with it is an error rather than something to reconcile. That is the
guard this class of mistake needs: an id whose direction changed
between vintages puts two different quantities in one column, and no consumer
downstream can separate them afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

DIRECTION_COLUMN = "direction"
# What identifies one measurement, per unit of measurement. The municipality-level
# tables hold one point per municipality and the block-level ones one per block, so
# there is no single right answer -- and guessing it wrong means duplicates go
# undetected, which is the whole thing this function is for. The caller states it.
BLOCK_KEY = ("block_id", "schedule_id")
MUNICIPALITY_KEY = ("muni_code", "schedule_id")


class MeasurementError(Exception):
    """Raised when vintages cannot be merged without losing or mixing meaning."""


def with_direction(frame: pd.DataFrame, directions: Mapping[str, str]) -> pd.DataFrame:
    """Return ``frame`` with a ``direction`` column agreeing with ``directions``.

    A frame that never had the column gets it filled in. A frame that has it
    must agree: the declaration is the contract, and a row that contradicts it
    is not a row to trust in either reading.
    """
    undeclared = sorted(set(frame["schedule_id"].astype(str)) - set(directions))
    if undeclared:
        raise MeasurementError(
            f"the configuration declares no direction for {undeclared}; "
            "a measurement whose direction is unknown cannot be merged"
        )

    declared = frame["schedule_id"].astype(str).map(directions)
    if DIRECTION_COLUMN not in frame.columns:
        return frame.assign(**{DIRECTION_COLUMN: declared})

    stored = frame[DIRECTION_COLUMN]
    if stored.isna().any():
        raise MeasurementError(
            f"{int(stored.isna().sum())} rows carry a null {DIRECTION_COLUMN}; "
            "a writer meant to record one and did not, which is not the same as "
            "a vintage that never had the column"
        )
    disagree = frame[stored.astype(str) != declared]
    if not disagree.empty:
        ids = sorted(set(disagree["schedule_id"].astype(str)))
        raise MeasurementError(
            f"{len(disagree)} stored rows disagree with the declared direction for {ids}; "
            "an id that changes direction between vintages mixes two quantities in one column"
        )
    return frame


def consolidate(
    frames: Sequence[pd.DataFrame],
    directions: Mapping[str, str],
    *,
    key: Sequence[str],
) -> pd.DataFrame:
    """Stack measurement vintages, one row per (block, schedule).

    An empty frame raises rather than contributing nothing: a vintage that was
    meant to add rows and added none is a failed run, and it looks exactly like
    a successful merge once it has been concatenated away.
    """
    if not frames:
        raise MeasurementError("no frames to consolidate")
    empty = [i for i, frame in enumerate(frames) if frame.empty]
    if empty:
        raise MeasurementError(f"frame(s) {empty} are empty; a vintage that added no rows failed")

    merged = pd.concat([with_direction(frame, directions) for frame in frames], ignore_index=True)
    missing = [column for column in key if column not in merged.columns]
    if missing:
        raise MeasurementError(
            f"the identity columns {missing} are absent, so duplicates cannot be detected; "
            f"frame has {sorted(merged.columns)}"
        )
    duplicated = merged.duplicated(subset=list(key), keep=False)
    if duplicated.any():
        sample = merged.loc[duplicated, list(key)].head(3).to_dict("records")
        raise MeasurementError(
            f"{int(duplicated.sum())} duplicate rows for the same {tuple(key)}: {sample}; "
            "two answers to one question, and keeping either silently discards a measurement"
        )
    return merged
