"""The storage-table contract, checked on synthetic frames instead of on real data.

A contract test run against real data can pin the column order, dtypes and keys
of every table on a recorded sample, but a copy of the repository without data
would then have nothing checking these guarantees. These tests pin the same
three guarantees on hand-made frames, through the same
functions (`column_types`, `sample_of`, `contract_path`) and through a parquet
round trip, because a column that changes order or turns from int64 to float64
on the way to disk reaches a consumer as a formatting difference rather than as
an error. The per-table contract -- every declared table's column order, dtypes
and key columns -- is pinned by `EXPECTED_SCHEMA` below, on synthetic tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from landeval.provenance import PROVENANCE_COLUMNS
from landeval.schema import (
    SAMPLE_ROWS,
    SAMPLED_TABLES,
    TABLE_KEYS,
    TRACKED_TABLES,
    column_types,
    contract_path,
    sample_of,
)

# A frame with one column of each dtype family the stored tables use, in a
# deliberately non-alphabetical order, so a reader that sorts columns is caught.
EXPECTED_COLUMNS = {
    "mesh_id": "string",
    "schedule_id": "string",
    "duration_s": "float64",
    "attempts": "int64",
    "snapped": "bool",
    "fetched_at": "datetime64[ns, UTC]",
}


def synthetic_frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mesh_id": pd.array([f"M{index:05d}" for index in range(rows)], dtype="string"),
            "schedule_id": pd.array(["s1"] * rows, dtype="string"),
            "duration_s": [60.0 * index for index in range(rows)],
            "attempts": list(range(rows)),
            "snapped": [index % 2 == 0 for index in range(rows)],
            "fetched_at": pd.date_range("2030-01-01", periods=rows, tz="UTC", unit="ns"),
        }
    )


def test_the_declared_keys_are_non_empty_and_name_distinct_columns() -> None:
    for name, key in TABLE_KEYS.items():
        assert key, f"{name} declares no key"
        assert len(set(key)) == len(key), f"{name} repeats a key column: {key}"


def test_every_table_is_either_tracked_or_sampled_and_never_both() -> None:
    assert set(TABLE_KEYS) >= TRACKED_TABLES
    assert set(TABLE_KEYS) - TRACKED_TABLES == SAMPLED_TABLES
    assert not TRACKED_TABLES & SAMPLED_TABLES


def test_column_types_reports_names_and_dtypes_in_column_order() -> None:
    types = column_types(synthetic_frame())

    assert list(types.items()) == list(EXPECTED_COLUMNS.items())
    assert list(types) == list(EXPECTED_COLUMNS)


def test_a_reordered_frame_reports_a_different_column_order() -> None:
    """Order is part of the contract: a reader that takes columns positionally
    silently reads a different column when it moves."""
    frame = synthetic_frame()
    swapped = frame[["schedule_id", "mesh_id", *frame.columns[2:]]]

    assert list(column_types(swapped).items()) != list(column_types(frame).items())


def test_a_dtype_change_is_visible_in_the_recorded_types() -> None:
    """A column that turns from int64 to float64 because a null appeared."""
    frame = synthetic_frame()
    frame.loc[0, "attempts"] = None
    frame["attempts"] = frame["attempts"].astype("float64")

    assert column_types(frame)["attempts"] == "float64"
    assert column_types(frame) != EXPECTED_COLUMNS


def test_a_parquet_round_trip_keeps_column_order_and_dtypes(tmp_path: Path) -> None:
    frame = synthetic_frame()
    path = tmp_path / "table.parquet"

    frame.to_parquet(path, index=False)
    read_back = pd.read_parquet(path)

    assert list(read_back.columns) == list(frame.columns)
    assert column_types(read_back) == column_types(frame)


def test_sample_of_keeps_the_columns_dtypes_and_key_uniqueness() -> None:
    frame = synthetic_frame(rows=SAMPLE_ROWS * 3 + 7)
    key = ("mesh_id", "schedule_id")

    sample = sample_of(frame, key)

    assert len(sample) == SAMPLE_ROWS
    assert list(sample.columns) == list(frame.columns)
    assert column_types(sample) == column_types(frame)
    assert not sample.duplicated(list(key)).any()


def test_sample_of_spreads_across_the_key_order_and_is_deterministic() -> None:
    frame = synthetic_frame(rows=SAMPLE_ROWS * 4)
    key = ("mesh_id",)

    first = sample_of(frame, key)
    again = sample_of(frame.sample(frac=1.0, random_state=1), key)

    assert first.equals(again)
    ids = list(first["mesh_id"])
    # The head of a sorted table is one corner of it; a stride reaches the far end.
    assert ids[0] == "M00000"
    assert ids[-1] > f"M{SAMPLE_ROWS * 3:05d}"


def test_a_small_frame_is_returned_whole_and_sorted_by_key() -> None:
    frame = synthetic_frame(rows=5).iloc[::-1].reset_index(drop=True)

    sample = sample_of(frame, ("mesh_id",))

    assert list(sample["mesh_id"]) == sorted(frame["mesh_id"])


@pytest.mark.parametrize("name", sorted(TABLE_KEYS))
def test_the_contract_path_follows_where_a_table_is_kept(name: str) -> None:
    processed, fixtures = Path("processed-dir"), Path("fixture-dir")

    path = contract_path(name, processed_dir=processed, fixture_dir=fixtures)

    expected_dir = processed if name in TRACKED_TABLES else fixtures
    assert path == expected_dir / f"{name}.parquet"


# --- the per-table contract, without real data --------------------------------
#
# What each declared table stores: its columns, in order, and their dtypes. It is
# written out here by hand, not read from any recorded sample. A test over recorded samples can
# compare them with this same literal, so the two cannot drift apart silently, and a
# table added to or dropped from `TABLE_KEYS` fails here.
#
# `commute_sheet` is wide: between `basis` and `outbound_gap_min` it holds one
# float64 column per configured commute schedule, named by the schedule id. The
# ids are configuration rather than schema, so the literal names the slot.
SCHEDULE_SLOT = "<one float64 column per configured schedule>"
SYNTHETIC_SCHEDULE_IDS = ("dep_a", "dep_b", "ret_a")

EXPECTED_SCHEMA: dict[str, dict[str, str]] = {
    "areas_sheet": {
        "muni_code": "str",
        "muni_name": "str",
        "habitable_cells": "int64",
        "habitable_km2": "float64",
        "usable_cells": "int64",
        "usable_km2": "float64",
        "unjudged_cells": "int64",
        "unjudged_km2": "float64",
        "unjudged_below_cells": "int64",
        "unjudged_below_km2": "float64",
        "unjudged_above_cells": "int64",
        "unjudged_above_km2": "float64",
        "clear_km2": "float64",
        "partial_km2": "float64",
        "median_min": "float64",
        "share": "float64",
        "tier": "str",
        "reachable_share": "float64",
        "d_reason": "str",
    },
    "block_measurements": {
        "block_id": "str",
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "basis": "str",
        "adopted_cells": "int64",
        "clear_cells": "int64",
        "selected_hazard_ratio": "float64",
        "freeflow_min": "float64",
        "predicted_min": "float64",
        "schedule_id": "str",
        "measured_min": "float64",
        "measured_distance_m": "float64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "float64",
        "route_error": "str",
        "source_stage": "str",
        "direction": "str",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
    "blocks": {
        "mesh_id": "string",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
        "bearing_deg": "float64",
        "sector": "int64",
        "osrm_good_max": "float64",
        "osrm_exclude": "float64",
        "predicted_min": "float64",
        "screening_class": "str",
        "selected_hazard_ratio": "float64",
        "selection_class": "str",
        "muni_code": "string",
        "muni_name": "string",
        "muni_measured_class": "str",
        "muni_measured_min": "float64",
        "block_id": "str",
        "adopted_cells": "Int64",
        "clear_cells": "int64",
        "basis": "str",
    },
    "calibration": {
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "time_band": "int64",
        "freeflow_min": "float64",
        "schedule_id": "str",
        "measured_min": "float64",
        "measured_distance_m": "int64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "float64",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
    "cell_municipality": {
        "mesh_id": "string",
        "muni_code": "str",
        "muni_name": "str",
    },
    "combined_cells": {
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
        "bearing_deg": "float64",
        "sector": "int64",
        "osrm_good_max": "float64",
        "osrm_exclude": "float64",
        "predicted_min": "float64",
        "screening_class": "str",
        "flood_planned_hazard_ratio": "float64",
        "flood_planned_max_depth_m": "float64",
        "flood_planned_status": "str",
        "flood_planned_covered": "bool",
        "flood_max_hazard_ratio": "float64",
        "flood_max_max_depth_m": "float64",
        "flood_max_status": "str",
        "flood_max_covered": "bool",
        "flood_collapse_hazard_ratio": "float64",
        "flood_collapse_max_depth_m": "float64",
        "flood_collapse_status": "str",
        "flood_collapse_covered": "bool",
        "landslide_hazard_ratio": "float64",
        "landslide_max_depth_m": "float64",
        "landslide_status": "str",
        "landslide_covered": "bool",
        "landslide_special_hazard_ratio": "float64",
        "landslide_special_max_depth_m": "float64",
        "landslide_special_status": "str",
        "landslide_special_covered": "bool",
        "tsunami_hazard_ratio": "float64",
        "tsunami_max_depth_m": "float64",
        "tsunami_status": "str",
        "tsunami_covered": "bool",
        "storm_surge_hazard_ratio": "float64",
        "storm_surge_max_depth_m": "float64",
        "storm_surge_status": "str",
        "storm_surge_covered": "bool",
        "inland_flood_hazard_ratio": "float64",
        "inland_flood_max_depth_m": "float64",
        "inland_flood_status": "str",
        "inland_flood_covered": "bool",
        "hazard_free_ratio": "float64",
        "fetched_at": "str",
    },
    "commute_sheet": {
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "basis": "str",
        SCHEDULE_SLOT: "float64",
        "outbound_gap_min": "float64",
        "outbound_sensitivity": "float64",
        "inbound_gap_min": "float64",
        "inbound_sensitivity": "float64",
        "round_trip_peak_min": "float64",
        "unsearchable_reason": "str",
        "searchable": "bool",
    },
    "fetch_provenance": {
        "cache_digest": "str",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "source": "str",
    },
    "final_cells": {
        "mesh_id": "string",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
        "bearing_deg": "float64",
        "sector": "int64",
        "osrm_good_max": "float64",
        "osrm_exclude": "float64",
        "predicted_min": "float64",
        "screening_class": "str",
        "applied_layers": "str",
        "selected_hazard_ratio": "float64",
        "selection_class": "str",
        "muni_code": "string",
        "muni_name": "string",
        "block_id": "str",
        "block_measured_min": "float64",
        "block_attempted": "bool",
        "block_class": "str",
        "block_measured_scaled": "float64",
    },
    "freeflow_all": {
        "mesh_id": "string",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
    },
    "freeflow_horizon": {
        "mesh_id": "string",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
    },
    "group_destination_times": {
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "schedule_id": "str",
        "direction": "str",
        "measured_min": "float64",
        "measured_distance_m": "int64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "object",
        "route_error": "object",
        "source_stage": "str",
        "batch_id": "string",
    },
    "mesh_hazard": {
        "mesh_id": "str",
        "flood_planned_hazard_ratio": "float64",
        "flood_planned_max_depth_m": "float64",
        "flood_planned_status": "str",
        "flood_planned_covered": "bool",
        "flood_max_hazard_ratio": "float64",
        "flood_max_max_depth_m": "float64",
        "flood_max_status": "str",
        "flood_max_covered": "bool",
        "flood_collapse_hazard_ratio": "float64",
        "flood_collapse_max_depth_m": "float64",
        "flood_collapse_status": "str",
        "flood_collapse_covered": "bool",
        "landslide_hazard_ratio": "float64",
        "landslide_max_depth_m": "float64",
        "landslide_status": "str",
        "landslide_covered": "bool",
        "landslide_special_hazard_ratio": "float64",
        "landslide_special_max_depth_m": "float64",
        "landslide_special_status": "str",
        "landslide_special_covered": "bool",
        "tsunami_hazard_ratio": "float64",
        "tsunami_max_depth_m": "float64",
        "tsunami_status": "str",
        "tsunami_covered": "bool",
        "storm_surge_hazard_ratio": "float64",
        "storm_surge_max_depth_m": "float64",
        "storm_surge_status": "str",
        "storm_surge_covered": "bool",
        "inland_flood_hazard_ratio": "float64",
        "inland_flood_max_depth_m": "float64",
        "inland_flood_status": "str",
        "inland_flood_covered": "bool",
        "hazard_free_ratio": "float64",
        "fetched_at": "str",
    },
    "municipality_measured": {
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "basis": "str",
        "adopted_cells": "int64",
        "clear_cells": "int64",
        "selected_hazard_ratio": "float64",
        "freeflow_min": "float64",
        "predicted_min": "float64",
        "schedule_id": "str",
        "measured_min": "float64",
        "measured_distance_m": "int64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "float64",
        "source_stage": "str",
        "measured_class": "str",
        "predicted_class": "str",
        "pref": "str",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
    "representatives": {
        "mesh_id": "string",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
        "bearing_deg": "float64",
        "sector": "int64",
        "osrm_good_max": "float64",
        "osrm_exclude": "float64",
        "predicted_min": "float64",
        "screening_class": "str",
        "selected_hazard_ratio": "float64",
        "selection_class": "str",
        "muni_code": "string",
        "muni_name": "string",
        "adopted_cells": "Int64",
        "clear_cells": "int64",
        "basis": "str",
    },
    "screened_cells": {
        "mesh_id": "string",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
        "bearing_deg": "float64",
        "sector": "int64",
        "osrm_good_max": "float64",
        "osrm_exclude": "float64",
        "predicted_min": "float64",
        "screening_class": "str",
    },
    "selected_cells": {
        "mesh_id": "string",
        "lat": "float64",
        "lon": "float64",
        "freeflow_min": "Float64",
        "freeflow_distance_m": "Float64",
        "snap_distance_m": "float64",
        "is_reachable": "bool",
        "is_off_road": "bool",
        "in_horizon": "boolean",
        "bearing_deg": "float64",
        "sector": "int64",
        "osrm_good_max": "float64",
        "osrm_exclude": "float64",
        "predicted_min": "float64",
        "screening_class": "str",
        "applied_layers": "str",
        "selected_hazard_ratio": "float64",
        "selection_class": "str",
    },
    "stability_check": {
        "schedule_id": "str",
        "quantile": "float64",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "direction": "str",
        "departure_time": "str",
        "cached_duration_s": "int64",
        "fresh_duration_s": "int64",
        "delta_s": "int64",
        "cached_distance_m": "int64",
        "fresh_distance_m": "int64",
        "delta_distance_m": "int64",
        "refetched_on": "str",
        "population_tables": "str",
        "min_lead_days": "int64",
    },
    "stability_control": {
        "schedule_id": "str",
        "quantile": "float64",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "direction": "str",
        "departure_time": "str",
        "cached_duration_s": "int64",
        "fresh_duration_s": "int64",
        "delta_s": "int64",
        "cached_distance_m": "int64",
        "fresh_distance_m": "int64",
        "delta_distance_m": "int64",
        "refetched_on": "str",
        "population_tables": "str",
        "min_lead_days": "int64",
    },
    "municipality_rep_target_times": {
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "basis": "str",
        "adopted_cells": "int64",
        "clear_cells": "int64",
        "selected_hazard_ratio": "float64",
        "freeflow_min": "float64",
        "predicted_min": "float64",
        "schedule_id": "str",
        "measured_min": "float64",
        "measured_distance_m": "int64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "float64",
        "source_stage": "str",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
    "municipality_rep_schedule_times": {
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "basis": "str",
        "adopted_cells": "int64",
        "clear_cells": "int64",
        "selected_hazard_ratio": "float64",
        "freeflow_min": "float64",
        "predicted_min": "float64",
        "schedule_id": "str",
        "direction": "str",
        "measured_min": "float64",
        "measured_distance_m": "int64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "object",
        "route_error": "object",
        "source_stage": "str",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
    "block_rep_target_times": {
        "block_id": "str",
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "basis": "str",
        "adopted_cells": "int64",
        "clear_cells": "int64",
        "selected_hazard_ratio": "float64",
        "freeflow_min": "float64",
        "predicted_min": "float64",
        "schedule_id": "str",
        "measured_min": "float64",
        "measured_distance_m": "float64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "float64",
        "route_error": "str",
        "source_stage": "str",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
    "block_rep_afternoon_outbound_times": {
        "block_id": "str",
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "basis": "str",
        "adopted_cells": "int64",
        "clear_cells": "int64",
        "selected_hazard_ratio": "float64",
        "freeflow_min": "float64",
        "predicted_min": "float64",
        "schedule_id": "str",
        "direction": "str",
        "measured_min": "float64",
        "measured_distance_m": "int64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "object",
        "route_error": "object",
        "source_stage": "str",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
    "block_rep_schedule_times": {
        "block_id": "str",
        "muni_code": "str",
        "muni_name": "str",
        "mesh_id": "str",
        "lat": "float64",
        "lon": "float64",
        "sector": "int64",
        "basis": "str",
        "adopted_cells": "int64",
        "clear_cells": "int64",
        "selected_hazard_ratio": "float64",
        "freeflow_min": "float64",
        "predicted_min": "float64",
        "schedule_id": "str",
        "direction": "str",
        "measured_min": "float64",
        "measured_distance_m": "float64",
        "departure_time": "str",
        "arrival_time": "str",
        "solver_converged": "bool",
        "solver_iterations": "int64",
        "solver_error_s": "object",
        "route_error": "str",
        "source_stage": "str",
        "destination_group": "str",
        "destination_name": "str",
        "destination_lat": "float64",
        "destination_lon": "float64",
        "fetched_at": "datetime64[us, UTC+09:00]",
        "batch_id": "string",
    },
}

# The dtype families the stored tables use. Closed: a dtype outside it is refused
# by the builder below rather than approximated, so a new family cannot pass by
# being built as some other one.
NULLABLE_FAMILIES = frozenset({"Float64", "Int64", "boolean"})
TIMESTAMP_FAMILIES = frozenset({"datetime64[us, UTC+09:00]"})
DTYPE_FAMILIES = frozenset({"float64", "int64", "bool", "str", "string", "object"}) | (
    NULLABLE_FAMILIES | TIMESTAMP_FAMILIES
)
JST_OFFSET = timezone(timedelta(hours=9))


def expand_schema(columns: dict[str, str], schedule_ids: Sequence[str]) -> dict[str, str]:
    """``columns`` with the schedule slot replaced, in place, by one column per id."""
    expanded: dict[str, str] = {}
    for column, dtype in columns.items():
        if column == SCHEDULE_SLOT:
            expanded.update(dict.fromkeys(schedule_ids, dtype))
        else:
            expanded[column] = dtype
    return expanded


def synthetic_column(dtype: str, rows: int, *, is_key: bool) -> pd.Series:
    """One column of ``dtype`` whose value differs on every row (bool aside), with
    nulls on odd rows for the nullable families unless it is part of a key."""
    index = range(rows)
    gaps = not is_key
    if dtype == "float64":
        return pd.Series([0.5 * i for i in index], dtype=dtype)
    if dtype == "int64":
        return pd.Series(list(index), dtype=dtype)
    if dtype == "bool":
        return pd.Series([i % 2 == 0 for i in index], dtype=dtype)
    if dtype in {"str", "string"}:
        return pd.Series([f"v{i:05d}" for i in index], dtype=dtype)
    if dtype == "object":
        return pd.Series([None] * rows, dtype=dtype)
    if dtype == "Float64":
        return pd.Series([None if gaps and i % 2 else 0.25 * i for i in index], dtype=dtype)
    if dtype == "Int64":
        return pd.Series([None if gaps and i % 2 else i for i in index], dtype=dtype)
    if dtype == "boolean":
        return pd.Series([None if gaps and i % 2 else i % 3 == 0 for i in index], dtype=dtype)
    if dtype == "datetime64[us, UTC+09:00]":
        stamps = pd.date_range("2030-01-01", periods=rows, freq="h", tz=JST_OFFSET, unit="us")
        return pd.Series(stamps)
    raise AssertionError(f"no synthetic builder for dtype {dtype!r}")


def synthetic_table(name: str, rows: int = 12) -> pd.DataFrame:
    columns = expand_schema(EXPECTED_SCHEMA[name], SYNTHETIC_SCHEDULE_IDS)
    key = set(TABLE_KEYS[name])
    return pd.DataFrame(
        {
            column: synthetic_column(dtype, rows, is_key=column in key)
            for column, dtype in columns.items()
        }
    )


def test_the_expected_schema_covers_exactly_the_declared_tables() -> None:
    assert set(EXPECTED_SCHEMA) == set(TABLE_KEYS)


def test_every_expected_dtype_is_a_family_the_builder_knows() -> None:
    used = {dtype for columns in EXPECTED_SCHEMA.values() for dtype in columns.values()}

    assert used <= DTYPE_FAMILIES, sorted(used - DTYPE_FAMILIES)
    # The families most easily lost on the way to disk are present, so the
    # round trip below exercises them rather than only the plain ones.
    assert used >= NULLABLE_FAMILIES | TIMESTAMP_FAMILIES


def test_the_provenance_writer_writes_the_expected_columns() -> None:
    """`fetch_provenance` is written by this package, from a declared column
    tuple. Its column order is the writer's, so the literal must follow it."""
    assert list(EXPECTED_SCHEMA["fetch_provenance"]) == list(PROVENANCE_COLUMNS)


def test_only_the_wide_sheet_has_a_schedule_slot() -> None:
    slotted = {name for name, columns in EXPECTED_SCHEMA.items() if SCHEDULE_SLOT in columns}

    assert slotted == {"commute_sheet"}


def test_the_schedule_slot_expands_in_place() -> None:
    expanded = expand_schema(EXPECTED_SCHEMA["commute_sheet"], SYNTHETIC_SCHEDULE_IDS)
    names = list(expanded)
    start = names.index("basis") + 1

    assert names[start : start + len(SYNTHETIC_SCHEDULE_IDS)] == list(SYNTHETIC_SCHEDULE_IDS)
    assert names[start + len(SYNTHETIC_SCHEDULE_IDS)] == "outbound_gap_min"
    assert all(expanded[column] == "float64" for column in SYNTHETIC_SCHEDULE_IDS)


@pytest.mark.parametrize("name", sorted(TABLE_KEYS))
def test_the_declared_key_columns_are_in_the_expected_schema(name: str) -> None:
    assert set(TABLE_KEYS[name]) <= set(EXPECTED_SCHEMA[name]), name


@pytest.mark.parametrize("name", sorted(TABLE_KEYS))
def test_a_table_keeps_its_column_order_and_dtypes_through_parquet(
    name: str, tmp_path: Path
) -> None:
    expected = expand_schema(EXPECTED_SCHEMA[name], SYNTHETIC_SCHEDULE_IDS)
    frame = synthetic_table(name)
    path = tmp_path / f"{name}.parquet"

    assert list(column_types(frame).items()) == list(expected.items())
    frame.to_parquet(path, index=False)
    read_back = pd.read_parquet(path)

    assert list(column_types(read_back).items()) == list(expected.items())


@pytest.mark.parametrize("name", sorted(TABLE_KEYS))
def test_a_sample_of_a_table_keeps_its_key_unique_and_non_null(name: str) -> None:
    key = TABLE_KEYS[name]
    frame = synthetic_table(name, rows=SAMPLE_ROWS * 2 + 3)

    sample = sample_of(frame, key)

    assert len(sample) == SAMPLE_ROWS
    assert column_types(sample) == column_types(frame)
    assert not sample.duplicated(list(key)).any()
    assert not sample[list(key)].isna().any().any()
