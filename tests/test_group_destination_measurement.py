"""Measuring several destinations per point.

A group with one destination would let the measurement loop name it once and
carry it in a constant. A group can have several destinations, and here
`group_a` has three points sharing one `schedule_id` -- which is why the
destination is part of the key, and why the loop has to walk destinations rather
than assume one.

Every case here runs against a mock transport. The point of testing this at all
is that a real run costs money: many billable calls, and a loop that measures
the wrong thing costs the same as one that measures the right thing.
"""

from __future__ import annotations

import json
import traceback
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from landeval.config import Point, Schedule
from landeval.measure import MeasurementPlan, measure_points, merge_group
from landeval.routes import RoutesClient

# A neutral limit for these tests; the value a real run uses is the caller's choice.
MAX_UNROUTED = 5

TOKYO = (35.68, 139.76)

GROUP_A = [
    Point(name="Site A", lat=43.100000, lon=141.300000),
    Point(name="Site C", lat=43.400000, lon=142.500000),
    Point(name="Site B", lat=44.200000, lon=142.300000),
]

OUTBOUND = Schedule(
    id="group_a_depart_0630",
    type="depart",
    time="06:30",
    daytype=["weekend"],
    direction="outbound",
)

INBOUND = Schedule(
    id="group_a_depart_1730",
    type="depart",
    time="17:30",
    daytype=["weekend"],
    direction="inbound",
)


def points(count: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "muni_code": [f"1310{index}" for index in range(count)],
            "muni_name": [f"市{index}" for index in range(count)],
            "mesh_id": [f"5339358{index}" for index in range(count)],
            "lat": [35.68 + index * 0.01 for index in range(count)],
            "lon": [139.76 + index * 0.01 for index in range(count)],
            "freeflow_min": [60.0] * count,
        }
    )


def responder(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "routes": [{"duration": "3600s", "distanceMeters": 50_000, "staticDuration": "3000s"}]
        },
    )


def client(tmp_path: object) -> RoutesClient:
    return RoutesClient(
        "test-key",
        cache_dir=tmp_path,  # type: ignore[arg-type]
        transport=httpx.MockTransport(responder),
    )


def plan(destinations: list[Point], schedules: list[Schedule]) -> MeasurementPlan:
    return MeasurementPlan(
        group="group_a",
        destinations=destinations,
        schedules=schedules,
        today=date(2026, 7, 12),
        stage="B",
    )


def test_one_row_per_point_per_destination_per_schedule(tmp_path: object) -> None:
    """The shape a full run buys: municipalities x destinations x legs. Here
    2 x 3 x 2 = 12, and the arithmetic is the same one that scales up."""
    with client(tmp_path) as routes:
        rows = measure_points(
            routes, points(2), plan(GROUP_A, [OUTBOUND, INBOUND]), max_unrouted=MAX_UNROUTED
        )

    assert len(rows) == 12


def test_every_row_names_the_destination_it_measured(tmp_path: object) -> None:
    """The four destination columns are what make the rows tellable apart: three
    destinations share `group_a_depart_0630`, so without them a third of the
    table is indistinguishable from another third."""
    with client(tmp_path) as routes:
        rows = measure_points(
            routes, points(1), plan(GROUP_A, [OUTBOUND]), max_unrouted=MAX_UNROUTED
        )

    frame = pd.DataFrame.from_records(rows)
    assert sorted(frame["destination_name"]) == sorted(point.name for point in GROUP_A)
    assert set(frame["destination_group"]) == {"group_a"}
    assert frame["destination_lat"].tolist() == [point.lat for point in GROUP_A]


def test_the_declared_key_is_unique(tmp_path: object) -> None:
    """`(mesh_id, schedule_id, destination_name)` -- the declared key, which
    `TABLE_KEYS` pins. `(mesh_id, schedule_id)` alone would not be."""
    with client(tmp_path) as routes:
        rows = measure_points(
            routes, points(2), plan(GROUP_A, [OUTBOUND, INBOUND]), max_unrouted=MAX_UNROUTED
        )

    frame = pd.DataFrame.from_records(rows)
    assert not frame.duplicated(["mesh_id", "schedule_id", "destination_name"]).any()
    assert frame.duplicated(["mesh_id", "schedule_id"]).any()  # the reason the key is wider


def test_the_direction_decides_which_end_is_the_origin(tmp_path: object) -> None:
    """The inbound leg leaves the group_a; measuring it as a trip *to*
    the group_a answers a different question at the same price -- and the row
    would look identical either way. So the assertion is on what went over the
    wire, not on the column that merely repeats the intent."""
    origins: list[float] = []

    def record(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        origins.append(body["origin"]["location"]["latLng"]["latitude"])
        return responder(request)

    with RoutesClient(
        "test-key",
        cache_dir=tmp_path,  # type: ignore[arg-type]
        transport=httpx.MockTransport(record),
    ) as routes:
        measure_points(
            routes, points(1), plan(GROUP_A[:1], [OUTBOUND, INBOUND]), max_unrouted=MAX_UNROUTED
        )

    assert origins == [points(1)["lat"].iloc[0], GROUP_A[0].lat]


def test_every_row_records_when_it_was_fetched(tmp_path: object) -> None:
    """A batch is `(destination_group, schedule_id, fetch date)`, and the fetch date has
    to come from the row."""
    with client(tmp_path) as routes:
        rows = measure_points(
            routes, points(1), plan(GROUP_A[:1], [OUTBOUND]), max_unrouted=MAX_UNROUTED
        )

    assert all(row["fetched_at"] for row in rows)


def test_the_worst_case_is_counted_before_anything_is_spent() -> None:
    """The number the dry run prints and `--max-calls` compares against. A
    departure-anchored
    schedule is one call -- no arrival back-solving."""
    assert plan(GROUP_A, [OUTBOUND, INBOUND]).worst_case(50) == 300


def test_an_arrival_anchored_schedule_costs_its_iterations() -> None:
    """Not the group-destination case, but the counter that keeps the estimate honest if a
    schedule is ever switched back to an arrival anchor."""
    arrival = Schedule(
        id="group_a_arrive_0600",
        type="arrive",
        time="06:00",
        daytype=["weekend"],
        direction="outbound",
    )

    assert plan(GROUP_A, [arrival]).worst_case(1) > 3


def test_a_route_failure_is_recorded_not_dropped(tmp_path: object) -> None:
    """A point the router cannot reach is a fact about that point. Dropping it
    leaves a gap that reads as "not measured yet"."""

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"routes": []})

    with RoutesClient(
        "test-key",
        cache_dir=tmp_path,  # type: ignore[arg-type]
        transport=httpx.MockTransport(refuse),
    ) as routes:
        rows = measure_points(
            routes, points(1), plan(GROUP_A[:1], [OUTBOUND]), max_unrouted=MAX_UNROUTED
        )

    assert len(rows) == 1
    assert rows[0]["measured_min"] is None
    assert rows[0]["route_error"]


def test_a_run_of_failures_stops_the_job(tmp_path: object) -> None:
    """A systemic failure -- a revoked key, a changed response shape -- looks
    exactly like a series of unreachable points, and burns the budget while
    producing nothing."""

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"routes": []})

    with (
        RoutesClient(
            "test-key",
            cache_dir=tmp_path,  # type: ignore[arg-type]
            transport=httpx.MockTransport(refuse),
        ) as routes,
        pytest.raises(RuntimeError, match="経路なし"),
    ):
        measure_points(routes, points(3), plan(GROUP_A, [OUTBOUND]), max_unrouted=2)


# Keys holding characters the quoting functions escape, beside a plain one and
# a one-character one. Any of them can be sent in a header.
RELAY_KEYS = [
    "synthetic-relay-key-0123456789abcdef",
    "a",
    "synthetic\\relay-key-0123",
    "synthetic'relay-key-0123",
    'synthetic"relay-key-0123',
    "synthetic\trelay-key-0123",
    "synthetic\x7frelay-key-0123",
]


def escaped_forms(key: str) -> set[str]:
    """The key raw, or as ``repr``, ``ascii`` and ``json.dumps`` write it."""
    both = key + "'\""
    return {
        key,
        repr(key)[1:-1],
        repr(both)[1:-4],
        ascii(key)[1:-1],
        ascii(both)[1:-4],
        json.dumps(key)[1:-1],
        json.dumps(key, ensure_ascii=False)[1:-1],
    }


def carries_the_key(text: str, key: str) -> bool:
    return any(form in text for form in escaped_forms(key))


@pytest.mark.parametrize("key", RELAY_KEYS)
def test_the_recorded_error_and_the_stop_never_carry_the_key(tmp_path: Path, key: str) -> None:
    """Both relays: ``route_error`` on the kept row, and the stop, which chains
    the routes error on purpose and is only as clean as everything chained to it."""
    text = f"{key}\"'{key}!"

    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"routes": [{"duration": text, "distanceMeters": 1}]})

    def make(directory: Path) -> RoutesClient:
        return RoutesClient(key, cache_dir=directory, transport=httpx.MockTransport(echo))

    with make(tmp_path / "kept") as routes:
        rows = measure_points(routes, points(1), plan(GROUP_A[:1], [OUTBOUND]), max_unrouted=5)
    assert rows[0]["route_error"]
    assert not carries_the_key(str(rows[0]["route_error"]), key)

    with make(tmp_path / "stop") as routes, pytest.raises(RuntimeError) as excinfo:
        measure_points(routes, points(1), plan(GROUP_A[:1], [OUTBOUND]), max_unrouted=0)

    chained: list[BaseException] = []
    current: BaseException | None = excinfo.value
    while current is not None and len(chained) < 10:
        chained.append(current)
        current = current.__cause__ or current.__context__
    assert len(chained) == 2, "the stop chains exactly the routes error"
    stop, routes_error = chained
    assert not carries_the_key(str(routes_error), key)
    assert not carries_the_key(repr(routes_error), key)
    # The stop's own text also names the cell, the schedule and the
    # destination (``group_a_depart_0630``, ``Site A``), so a one-character key
    # occurs there whatever the redaction does; it is checked whole only for a
    # realistic key.
    if len(key) >= 8:
        assert not carries_the_key(str(stop), key)
        assert not carries_the_key(repr(stop), key)
        assert not carries_the_key("".join(traceback.format_exception(stop)), key)


# Written by a provenance backfill, not by the measurement.
_BACKFILLED_COLUMNS = {"batch_id"}


def existing(group: str, rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mesh_id": [f"5339358{index}" for index in range(rows)],
            "schedule_id": [f"{group}_depart_0500"] * rows,
            "destination_name": [f"{group}地点"] * rows,
            "destination_group": [group] * rows,
            "measured_min": [60.0] * rows,
        }
    )


def test_a_second_group_keeps_the_first(tmp_path: Path) -> None:
    """A writer that replaced the whole file would make running `group_b`
    after `group_a` discard every row that had already been paid for -- and
    the run would report only "wrote ...".

    The groups are measured in separate runs because each has its own
    `--max-calls` ceiling, so this is the ordinary path, not an edge case.
    """
    path = tmp_path / "group.parquet"
    existing("group_a").to_parquet(path, index=False)

    merged = merge_group(path, existing("group_b"), group="group_b")

    assert sorted(set(merged["destination_group"])) == ["group_a", "group_b"]
    assert len(merged) == 4


def test_rerunning_a_group_replaces_its_own_rows(tmp_path: Path) -> None:
    """A re-run after an interruption is free and must not double the
    group's rows -- which would break the key and inflate every count taken
    from the table."""
    path = tmp_path / "group.parquet"
    existing("group_a").to_parquet(path, index=False)

    merged = merge_group(path, existing("group_a", rows=3), group="group_a")

    assert len(merged) == 3


def test_merging_into_a_table_without_the_group_column_is_refused(tmp_path: Path) -> None:
    """`--out` pointed at a table of some other shape. Concatenating would
    produce a frame that is neither, and the mistake would surface later as
    nulls in columns nobody expected to be nullable."""
    path = tmp_path / "other.parquet"
    pd.DataFrame({"mesh_id": ["1"], "measured_min": [1.0]}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="destination_group"):
        merge_group(path, existing("group_b"), group="group_b")


def test_merging_into_nothing_is_just_the_new_rows(tmp_path: Path) -> None:
    fresh = merge_group(tmp_path / "absent.parquet", existing("group_a"), group="group_a")

    assert len(fresh) == 2


def test_merging_into_a_backfilled_table_is_refused(tmp_path: Path) -> None:
    """After a provenance backfill, the table carries `batch_id`
    and a timestamp `fetched_at`; freshly measured rows carry neither.
    Concatenating them makes parquet raise from inside pyarrow, which is the
    lucky version -- the unlucky one is a table whose fetch times are half
    strings and whose batch ids are half null.
    """
    path = tmp_path / "group.parquet"
    backfilled = existing("group_a")
    backfilled["batch_id"] = "group_a/group_a_depart_0630/2026-07-12"
    backfilled.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="batch_id"):
        merge_group(path, existing("group_b"), group="group_b")


def test_replacing_the_only_group_in_a_backfilled_table_is_allowed(tmp_path: Path) -> None:
    """Nothing is kept, so nothing can disagree. This is the re-measure path the
    refusal above tells the operator to take."""
    path = tmp_path / "group.parquet"
    backfilled = existing("group_a")
    backfilled["batch_id"] = "group_a/group_a_depart_0630/2026-07-12"
    backfilled.to_parquet(path, index=False)

    merged = merge_group(path, existing("group_a", rows=3), group="group_a")

    assert len(merged) == 3
    assert "batch_id" not in merged.columns
