"""Merging measurement vintages without losing or mixing what they mean."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import pytest

from landeval.measurements import (
    BLOCK_KEY,
    MeasurementError,
    consolidate,
    with_direction,
)

DIRECTIONS = {"arr_0930": "outbound", "ret_1730": "inbound"}


def consolidate_blocks(
    frames: Sequence[pd.DataFrame], directions: Mapping[str, str]
) -> pd.DataFrame:
    return consolidate(frames, directions, key=BLOCK_KEY)


def rows(schedule_id: str, blocks: list[str], **extra: str | None) -> pd.DataFrame:
    frame = pd.DataFrame({"block_id": blocks, "schedule_id": schedule_id, "measured_min": 90.0})
    for name, value in extra.items():
        frame[name] = value
    return frame


# --- with_direction --------------------------------------------------------


def test_a_vintage_without_the_column_is_filled_from_the_declaration() -> None:
    """A vintage written before the direction column carries no direction of its own."""
    out = with_direction(rows("arr_0930", ["a", "b"]), DIRECTIONS)

    assert list(out["direction"]) == ["outbound", "outbound"]


def test_a_stored_direction_that_contradicts_the_declaration_is_rejected() -> None:
    """The missing-direction class of mistake is guarded.

    An id whose direction changed between vintages puts two quantities in one
    column, and nothing downstream can tell them apart afterwards -- which is
    why a return trip gets a new id rather than reusing an outbound one.
    """
    stored = rows("ret_1730", ["a"], direction="outbound")

    with pytest.raises(MeasurementError, match="ret_1730"):
        with_direction(stored, DIRECTIONS)


def test_a_matching_stored_direction_is_accepted() -> None:
    out = with_direction(rows("ret_1730", ["a"], direction="inbound"), DIRECTIONS)

    assert list(out["direction"]) == ["inbound"]


def test_a_schedule_the_configuration_does_not_declare_is_rejected() -> None:
    """Filling it from a neighbour, or leaving it null, both read as outbound
    somewhere downstream."""
    with pytest.raises(MeasurementError, match="depart_2100"):
        with_direction(rows("depart_2100", ["a"]), DIRECTIONS)


def test_a_null_stored_direction_is_rejected_rather_than_filled() -> None:
    """A column that exists but is empty is not the same as no column: it means
    a writer meant to record something and did not."""
    stored = rows("arr_0930", ["a"], direction=None)

    with pytest.raises(MeasurementError, match="null"):
        with_direction(stored, DIRECTIONS)


# --- consolidate -----------------------------------------------------------


def test_vintages_are_stacked_and_nothing_is_dropped() -> None:
    merged = consolidate(
        [rows("arr_0930", ["a", "b"]), rows("ret_1730", ["a", "b"])], DIRECTIONS, key=BLOCK_KEY
    )

    assert len(merged) == 4
    assert set(merged["schedule_id"]) == {"arr_0930", "ret_1730"}


def test_the_same_point_measured_twice_for_one_schedule_is_rejected() -> None:
    """Two rows for one (block, schedule) is two answers to one question, and
    silently keeping either one is how a re-run overwrites a measurement."""
    with pytest.raises(MeasurementError, match="duplicate"):
        consolidate_blocks([rows("arr_0930", ["a"]), rows("arr_0930", ["a"])], DIRECTIONS)


def test_the_same_point_at_the_same_clock_in_two_directions_is_kept() -> None:
    """`dep_1730` and `ret_1730` are the same clock, opposite ways. They
    are different measurements and both must survive."""
    merged = consolidate(
        [rows("arr_0930", ["a"]), rows("ret_1730", ["a"])],
        DIRECTIONS,
        key=BLOCK_KEY,
    )

    assert len(merged) == 2


def test_consolidating_nothing_is_rejected() -> None:
    with pytest.raises(MeasurementError, match="no frames"):
        consolidate([], DIRECTIONS, key=BLOCK_KEY)


def test_an_empty_vintage_is_rejected_rather_than_silently_contributing_nothing() -> None:
    with pytest.raises(MeasurementError, match="empty"):
        consolidate_blocks([rows("arr_0930", ["a"]), rows("arr_0930", [])], DIRECTIONS)


def test_the_identity_key_must_exist_in_the_frames() -> None:
    """The municipality-level frame has no block_id and the block-level one has no
    reason to share the municipality key; asking for a column that is not there must fail rather
    than let every row look unique."""
    with pytest.raises(MeasurementError, match="block_id"):
        consolidate([rows("arr_0930", ["a"]).drop(columns=["block_id"])], DIRECTIONS, key=BLOCK_KEY)
