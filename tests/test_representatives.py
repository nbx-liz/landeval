"""Tests for representative-point selection.

One point stands in for a whole municipality, so what it stands on matters as
much as how far it is. The rule prefers ground outside every designated hazard
zone even when that ground commutes worse, because a representative sitting in
the middle of an inundation zone would report a commute nobody would take.

The selection also has to be stable: the measurement cache is keyed on the
coordinate, so a rule that reshuffles under a tie would re-bill every run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from landeval.representatives import (
    BASIS_CLEAR,
    BASIS_LEAST_HAZARD,
    choose_representatives,
)


def cells(rows: list[tuple[str, str, str, float, float, str]]) -> pd.DataFrame:
    """rows: (mesh_id, muni_code, screening_class, hazard_ratio, predicted_min, selection_class)."""
    return pd.DataFrame(
        {
            "mesh_id": [r[0] for r in rows],
            "muni_code": pd.array([r[1] for r in rows], dtype="string"),
            "muni_name": pd.array([f"市{r[1]}" for r in rows], dtype="string"),
            "screening_class": [r[2] for r in rows],
            "selected_hazard_ratio": [r[3] for r in rows],
            "predicted_min": [r[4] for r in rows],
            "selection_class": [r[5] for r in rows],
            "lat": [35.0 + 0.01 * i for i in range(len(rows))],
            "lon": [139.0 + 0.01 * i for i in range(len(rows))],
            "freeflow_min": [r[4] * 0.6 for r in rows],
            "sector": [3] * len(rows),
        }
    )


def test_one_representative_per_municipality() -> None:
    frame = cells(
        [
            ("a", "13101", "pass", 0.0, 50.0, "clear"),
            ("b", "13101", "pass", 0.0, 60.0, "clear"),
            ("c", "13102", "hold", 0.0, 95.0, "clear"),
        ]
    )

    chosen = choose_representatives(frame)

    assert len(chosen) == 2
    assert set(chosen["muni_code"]) == {"13101", "13102"}


def test_the_best_commute_wins_among_equally_clear_ground() -> None:
    frame = cells(
        [
            ("far", "13101", "pass", 0.0, 80.0, "clear"),
            ("near", "13101", "pass", 0.0, 50.0, "clear"),
        ]
    )

    chosen = choose_representatives(frame)

    assert chosen.iloc[0]["mesh_id"] == "near"
    assert chosen.iloc[0]["basis"] == BASIS_CLEAR


def test_clear_ground_beats_a_closer_point_inside_a_hazard_zone() -> None:
    """A representative in the middle of an inundation zone reports a commute
    to somewhere nobody would buy."""
    frame = cells(
        [
            ("clear_far", "13101", "pass", 0.0, 80.0, "clear"),
            ("wet_near", "13101", "pass", 0.6, 40.0, "partial"),
        ]
    )

    chosen = choose_representatives(frame)

    assert chosen.iloc[0]["mesh_id"] == "clear_far"


def test_a_municipality_without_clear_ground_still_gets_a_representative() -> None:
    """Dropping it would silently remove a candidate area from the run."""
    frame = cells(
        [
            ("wet", "13101", "hold", 0.6, 100.0, "partial"),
            ("wetter", "13101", "hold", 0.9, 70.0, "partial"),
        ]
    )

    chosen = choose_representatives(frame)

    assert len(chosen) == 1
    assert chosen.iloc[0]["mesh_id"] == "wet"
    assert chosen.iloc[0]["basis"] == BASIS_LEAST_HAZARD


def test_the_basis_is_recorded_rather_than_inferred() -> None:
    """Downstream must be able to tell the two kinds of representative apart."""
    frame = cells(
        [
            ("dry", "13101", "pass", 0.0, 50.0, "clear"),
            ("wet", "13102", "hold", 0.4, 50.0, "partial"),
        ]
    )

    chosen = choose_representatives(frame).set_index("muni_code")

    assert chosen.loc["13101", "basis"] == BASIS_CLEAR
    assert chosen.loc["13102", "basis"] == BASIS_LEAST_HAZARD


def test_cells_the_transport_filter_removed_are_not_eligible() -> None:
    frame = cells(
        [
            ("kept", "13101", "hold", 0.0, 110.0, "clear"),
            ("cut", "13101", "exclude", 0.0, 30.0, "transport_excluded"),
        ]
    )

    chosen = choose_representatives(frame)

    assert chosen.iloc[0]["mesh_id"] == "kept"


def test_a_municipality_with_no_adopted_cell_yields_nothing() -> None:
    frame = cells([("cut", "13101", "exclude", 0.0, 30.0, "transport_excluded")])

    assert choose_representatives(frame).empty


def test_cells_outside_every_municipality_are_dropped() -> None:
    frame = cells([("a", "13101", "pass", 0.0, 50.0, "clear")])
    frame.loc[0, "muni_code"] = None

    assert choose_representatives(frame).empty


def test_ties_are_broken_deterministically_so_the_cache_survives() -> None:
    """Same inputs in a different order must pick the same point."""
    rows = [
        ("b", "13101", "pass", 0.0, 50.0, "clear"),
        ("a", "13101", "pass", 0.0, 50.0, "clear"),
    ]
    forward = choose_representatives(cells(rows))
    backward = choose_representatives(cells(list(reversed(rows))))

    assert forward.iloc[0]["mesh_id"] == backward.iloc[0]["mesh_id"] == "a"


def test_the_counts_behind_each_choice_are_carried_along() -> None:
    frame = cells(
        [
            ("a", "13101", "pass", 0.0, 50.0, "clear"),
            ("b", "13101", "hold", 0.5, 95.0, "partial"),
            ("c", "13101", "exclude", 0.0, 30.0, "transport_excluded"),
        ]
    )

    chosen = choose_representatives(frame).iloc[0]

    assert chosen["adopted_cells"] == 2
    assert chosen["clear_cells"] == 1


def test_a_missing_column_is_reported_rather_than_guessed() -> None:
    frame = cells([("a", "13101", "pass", 0.0, 50.0, "clear")]).drop(columns=["predicted_min"])

    with pytest.raises(KeyError, match="predicted_min"):
        choose_representatives(frame)


# --- grouping by something other than a municipality -----------------------


def test_the_same_rule_applies_to_any_grouping() -> None:
    """Block-level points group by grid block; only the key changes, not the rule."""
    frame = cells(
        [
            ("a", "13101", "pass", 0.0, 80.0, "clear"),
            ("b", "13101", "pass", 0.0, 50.0, "clear"),
        ]
    ).assign(block_id=["north", "north"])

    chosen = choose_representatives(frame, group_column="block_id")

    assert len(chosen) == 1
    assert chosen.iloc[0]["mesh_id"] == "b"


def test_one_municipality_can_span_several_blocks() -> None:
    frame = cells(
        [
            ("a", "13101", "pass", 0.0, 50.0, "clear"),
            ("b", "13101", "pass", 0.0, 60.0, "clear"),
        ]
    ).assign(block_id=["north", "south"])

    assert len(choose_representatives(frame, group_column="block_id")) == 2


def test_an_unknown_grouping_column_is_reported() -> None:
    with pytest.raises(KeyError, match="block_id"):
        choose_representatives(
            cells([("a", "13101", "pass", 0.0, 50.0, "clear")]), group_column="block_id"
        )
