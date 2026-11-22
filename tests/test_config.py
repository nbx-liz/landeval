"""Tests for the destination/threshold configuration loader.

The configuration is human-edited YAML, i.e. an external open grammar. Every
off-contract input must be rejected loudly rather than tolerated, so this module
documents the full accept/reject matrix.

Schema-level cases are built by mutating a base mapping and dumping it, so they
do not depend on the indentation of a literal YAML fixture. Parse-level cases
(malformed, empty, non-mapping) necessarily use raw text.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from landeval.config import ConfigError, load_config

BASE: dict[str, Any] = {
    "origin_grid": {
        "mesh_size_m": 500,
        "bbox_source": "freeflow_minutes",
        "freeflow_minutes": 100,
    },
    "thresholds": {
        "commute": {
            "target_schedule": "arr_0930",
            "good_max_min": 45,
            "hard_max_min": 70,
            "exclude_min": 100,
        }
    },
    "destinations": {
        "commute": {
            "aggregate": "single",
            "used_for_screening": True,
            "points": [{"name": "Central Station", "lat": 35.6800, "lon": 139.7600}],
            "schedules": [
                {
                    "id": "arr_0930",
                    "type": "arrive",
                    "time": "09:30",
                    "daytype": "friday",
                    "direction": "outbound",
                },
                {
                    "id": "dep_1730",
                    "type": "depart",
                    "time": "17:30",
                    "daytype": "friday",
                    "direction": "outbound",
                },
            ],
        },
        "group_b": {
            "aggregate": "min",
            "used_for_screening": False,
            "points": [
                {"name": "Site D", "lat": 43.5800, "lon": 141.9000},
                {"name": "Site E", "lat": 43.9000, "lon": 141.8000},
            ],
            "schedules": [
                {
                    "id": "arrive_0600",
                    "type": "arrive",
                    "time": "06:00",
                    "daytype": ["weekday", "weekend"],
                    "direction": "outbound",
                }
            ],
        },
    },
}


def write_raw(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "destinations.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def write(tmp_path: Path, mutate: Any = None) -> Path:
    doc = copy.deepcopy(BASE)
    if mutate is not None:
        mutate(doc)
    return write_raw(tmp_path, yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))


# --- valid form ------------------------------------------------------------


def test_loads_valid_config(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path))

    assert cfg.origin_grid.mesh_size_m == 500
    assert cfg.origin_grid.freeflow_minutes == 100
    assert cfg.thresholds.commute.target_schedule == "arr_0930"
    assert set(cfg.destinations) == {"commute", "group_b"}


def test_screening_group_is_resolvable(tmp_path: Path) -> None:
    group_id, group = load_config(write(tmp_path)).screening_group()

    assert group_id == "commute"
    assert group.used_for_screening is True


def test_daytype_is_normalised_to_a_list(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path))

    assert cfg.destinations["commute"].schedules[0].daytype == ["friday"]
    assert cfg.destinations["group_b"].schedules[0].daytype == ["weekday", "weekend"]


def test_aggregate_accepts_scalar_and_list(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["aggregate"] = ["each", "mean", "max"]

    cfg = load_config(write(tmp_path, mutate))
    assert cfg.destinations["group_b"].aggregate == ["each", "mean", "max"]
    assert cfg.destinations["commute"].aggregate == ["single"]


def test_arrival_anchored_schedules_are_identifiable(tmp_path: Path) -> None:
    """Arrival anchors need departure-time back-solving."""
    cfg = load_config(write(tmp_path))

    commute = cfg.destinations["commute"]
    assert [s.id for s in commute.schedules if s.is_arrival_anchored] == ["arr_0930"]


def test_schedule_exposes_hour_and_minute(tmp_path: Path) -> None:
    schedule = load_config(write(tmp_path)).destinations["group_b"].schedules[0]

    assert (schedule.hour, schedule.minute) == (6, 0)


# --- direction ----------------------------------------------------
#
# `type` anchors the clock, not the direction of travel. Leaving direction
# implicit would let an evening schedule be measured as an outbound trip
# without anything failing, so it is stated.


def test_a_schedule_that_states_outbound_is_outbound(tmp_path: Path) -> None:
    schedule = load_config(write(tmp_path)).destinations["commute"].schedules[0]

    assert schedule.direction == "outbound"
    assert schedule.is_outbound


def test_a_schedule_that_omits_direction_is_rejected(tmp_path: Path) -> None:
    """No default, deliberately.

    A default is what a schedule added later silently inherits when nobody
    considered the question, and inheriting ``outbound`` would bill a return
    trip as a trip *towards* the destination because the direction was implied
    by the time anchor. So the config refuses a schedule that is silent about
    it.
    """

    def mutate(doc: dict[str, Any]) -> None:
        del doc["destinations"]["commute"]["schedules"][1]["direction"]

    with pytest.raises(ConfigError, match="direction"):
        load_config(write(tmp_path, mutate))


def test_direction_can_be_declared_inbound(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][1]["direction"] = "inbound"

    schedule = load_config(write(tmp_path, mutate)).destinations["commute"].schedules[1]

    assert schedule.direction == "inbound"
    assert not schedule.is_outbound


def test_an_unknown_direction_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][0]["direction"] = "return"

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


def test_direction_is_independent_of_the_time_anchor(tmp_path: Path) -> None:
    """A departure anchor says nothing about which way the trip goes."""

    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][0]["type"] = "depart"
        doc["destinations"]["commute"]["schedules"][0]["direction"] = "outbound"

    schedule = load_config(write(tmp_path, mutate)).destinations["commute"].schedules[0]

    assert schedule.type == "depart"
    assert schedule.is_outbound


def test_endpoints_swap_for_an_inbound_schedule(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][0]["direction"] = "inbound"

    schedule = load_config(write(tmp_path, mutate)).destinations["commute"].schedules[0]
    cell, destination = (35.0, 139.0), (35.5, 139.5)

    assert schedule.endpoints(cell, destination) == (destination, cell)


def test_endpoints_keep_their_order_for_an_outbound_schedule(tmp_path: Path) -> None:
    schedule = load_config(write(tmp_path)).destinations["commute"].schedules[0]
    cell, destination = (35.0, 139.0), (35.5, 139.5)

    assert schedule.endpoints(cell, destination) == (cell, destination)


# --- unparseable / absent input must fail loudly ---------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="parse"):
        load_config(write_raw(tmp_path, "origin_grid: [unclosed"))


def test_empty_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="empty"):
        load_config(write_raw(tmp_path, ""))


def test_non_mapping_document_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write_raw(tmp_path, "- just\n- a\n- list\n"))


# --- schema violations -----------------------------------------------------


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["unexpected_key"] = 1

    with pytest.raises(ConfigError, match="extra"):
        load_config(write(tmp_path, mutate))


def test_unknown_point_key_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["points"][0]["altitude"] = 20

    with pytest.raises(ConfigError, match="extra"):
        load_config(write(tmp_path, mutate))


def test_missing_required_key_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        del doc["destinations"]["commute"]["used_for_screening"]

    with pytest.raises(ConfigError, match="missing"):
        load_config(write(tmp_path, mutate))


@pytest.mark.parametrize("bad_time", ["24:00", "9:00", "10:60", "1000", "", "10:00:00", "aa:bb"])
def test_invalid_time_format_is_rejected(tmp_path: Path, bad_time: str) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][0]["time"] = bad_time

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


@pytest.mark.parametrize("good_time", ["00:00", "23:59", "09:05"])
def test_valid_boundary_times_are_accepted(tmp_path: Path, good_time: str) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][0]["time"] = good_time

    assert load_config(write(tmp_path, mutate)).destinations["commute"].schedules[0].time == (
        good_time
    )


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(0.0, 0.0), (35.0, 0.0), (0.0, 139.0), (90.0, 200.0), (-35.6, 139.7), (139.7600, 35.6800)],
)
def test_coordinates_outside_japan_are_rejected(tmp_path: Path, lat: float, lon: float) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["points"][0].update({"lat": lat, "lon": lon})

    with pytest.raises(ConfigError, match="Japan"):
        load_config(write(tmp_path, mutate))


def test_unknown_daytype_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][0]["daytype"] = "someday"

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


def test_unknown_schedule_type_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][0]["type"] = "leave"

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


def test_unknown_aggregate_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["aggregate"] = "median"

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


# --- cross-field invariants ------------------------------------------------


def test_single_aggregate_requires_exactly_one_point(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["points"].append(
            {"name": "extra", "lat": 35.7, "lon": 139.8}
        )

    with pytest.raises(ConfigError, match="single"):
        load_config(write(tmp_path, mutate))


def test_single_aggregate_cannot_be_combined(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["aggregate"] = ["single", "mean"]

    with pytest.raises(ConfigError, match="single"):
        load_config(write(tmp_path, mutate))


def test_min_aggregate_requires_at_least_two_points(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["points"] = doc["destinations"]["group_b"]["points"][:1]

    with pytest.raises(ConfigError, match="min"):
        load_config(write(tmp_path, mutate))


def test_duplicate_schedule_id_within_group_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["schedules"][1]["id"] = "arr_0930"

    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write(tmp_path, mutate))


def test_schedule_id_shared_between_groups_is_rejected(tmp_path: Path) -> None:
    """Measurement tables are keyed by schedule id alone."""

    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["schedules"][0]["id"] = "dep_1730"

    with pytest.raises(ConfigError, match=r"'dep_1730' in \['commute', 'group_b'\]"):
        load_config(write(tmp_path, mutate))


def test_distinct_schedule_ids_across_groups_load(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path))

    ids = [s.id for group in cfg.destinations.values() for s in group.schedules]
    assert len(ids) == len(set(ids))


def test_duplicate_point_name_within_group_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["points"][1]["name"] = "Site D"

    with pytest.raises(ConfigError, match="duplicate"):
        load_config(write(tmp_path, mutate))


def test_target_schedule_must_exist_in_the_screening_group(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["thresholds"]["commute"]["target_schedule"] = "arrive_9999"

    with pytest.raises(ConfigError, match="target_schedule"):
        load_config(write(tmp_path, mutate))


def test_target_schedule_from_a_non_screening_group_is_rejected(tmp_path: Path) -> None:
    """A schedule id that exists, but only in a group not used for screening."""

    def mutate(doc: dict[str, Any]) -> None:
        doc["thresholds"]["commute"]["target_schedule"] = "arrive_0600"

    with pytest.raises(ConfigError, match="target_schedule"):
        load_config(write(tmp_path, mutate))


@pytest.mark.parametrize(
    ("good", "hard", "exclude"),
    [(45, 45, 100), (70, 45, 100), (45, 100, 70), (45, 70, 70)],
)
def test_thresholds_must_be_strictly_increasing(
    tmp_path: Path, good: int, hard: int, exclude: int
) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["thresholds"]["commute"].update(
            {"good_max_min": good, "hard_max_min": hard, "exclude_min": exclude}
        )

    with pytest.raises(ConfigError, match="increasing"):
        load_config(write(tmp_path, mutate))


def test_exactly_one_group_may_be_used_for_screening(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["used_for_screening"] = True

    with pytest.raises(ConfigError, match="exactly one"):
        load_config(write(tmp_path, mutate))


def test_no_screening_group_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["commute"]["used_for_screening"] = False

    with pytest.raises(ConfigError, match="exactly one"):
        load_config(write(tmp_path, mutate))


def test_group_without_points_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["points"] = []

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


def test_group_without_schedules_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"]["group_b"]["schedules"] = []

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


def test_empty_destinations_is_rejected(tmp_path: Path) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["destinations"] = {}

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


@pytest.mark.parametrize("field", ["mesh_size_m", "freeflow_minutes"])
def test_non_positive_grid_values_are_rejected(tmp_path: Path, field: str) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["origin_grid"][field] = 0

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))


# --- the region section ------------------------------------------------------

REGION = {"prefectures": {"13": "Tokyo", "14": "Kanagawa"}, "bbox": [139.0, 35.0, 140.0, 36.0]}


def test_a_region_section_is_optional_and_loads(tmp_path: Path) -> None:
    assert load_config(write(tmp_path)).region is None

    def mutate(doc: dict[str, Any]) -> None:
        doc["region"] = REGION

    region = load_config(write(tmp_path, mutate)).region
    assert region is not None
    assert region.prefectures == {"13": "Tokyo", "14": "Kanagawa"}
    assert region.bbox == (139.0, 35.0, 140.0, 36.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"prefectures": {}, "bbox": [139.0, 35.0, 140.0, 36.0]},
        {"prefectures": {"1": "x"}, "bbox": [139.0, 35.0, 140.0, 36.0]},
        {"prefectures": {"13": "x"}, "bbox": [140.0, 35.0, 139.0, 36.0]},
        {"prefectures": {"13": "x"}, "bbox": [35.0, 139.0, 36.0, 140.0]},
        {"prefectures": {"13": "x"}, "bbox": [139.0, 35.0, 140.0]},
        {"prefectures": {"13": "x"}},
    ],
)
def test_an_invalid_region_is_rejected(tmp_path: Path, bad: dict[str, Any]) -> None:
    def mutate(doc: dict[str, Any]) -> None:
        doc["region"] = bad

    with pytest.raises(ConfigError):
        load_config(write(tmp_path, mutate))
