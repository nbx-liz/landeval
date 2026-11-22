"""Running a measurement over points, destinations and schedules.

A loop inside a script cannot be tested without spending money, so it lives
here. A destination group may have one destination or several, and a loop that
assumes one destination cannot measure the others -- it would measure the first
point of a group once per destination and cost the same.

The whole file exists to be exercised against a mock transport before a run that
bills many calls. A loop measuring the wrong thing costs exactly what a correct
one does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from landeval.basedate import next_reference_datetime
from landeval.config import Point, Schedule
from landeval.routes import DEFAULT_MAX_ARRIVAL_ITERATIONS, RoutesClient, RoutesError


@dataclass(frozen=True)
class MeasurementPlan:
    """What one run measures: which destinations, at which time anchors."""

    group: str
    destinations: Sequence[Point]
    schedules: Sequence[Schedule]
    today: date
    stage: str
    # Extra columns copied from the point onto every row. The municipality-level table carries
    # the selection and prediction columns so `municipality_rep_target_times.parquet` keeps the
    # schema the schema contract pins; the group-destination table carries only
    # `sector` and `freeflow_min`, because `predicted_min` there would be the
    # *commute* model's prediction sitting in a column named as if it belonged to the trip measured.
    carry: tuple[str, ...] = ()

    def worst_case(self, points: int) -> int:
        """The most billable calls this plan can make over ``points`` points.

        Arrival anchors are the expensive ones -- the API ignores `arrivalTime`
        for driving, so each is back-solved over several departures. A
        departure-anchored leg costs one call per leg and municipality rather
        than several times that.
        """
        arrivals = sum(1 for schedule in self.schedules if schedule.is_arrival_anchored)
        departures = len(self.schedules) - arrivals
        per_destination = arrivals * DEFAULT_MAX_ARRIVAL_ITERATIONS + departures
        return points * len(self.destinations) * per_destination


def _number(value: object) -> float:
    """Narrow a pandas scalar to float; its stub type is a wide union."""
    return float(value)  # type: ignore[arg-type]


# How each carried column is narrowed. A column absent from this map is copied
# as a string rather than guessed at, because a silently widened dtype is the
# schema drift the schema contract exists to catch.
_CARRY_TYPES: dict[str, Callable[[object], object]] = {
    "sector": lambda value: int(_number(value)),
    "adopted_cells": lambda value: int(_number(value)),
    "clear_cells": lambda value: int(_number(value)),
    "selected_hazard_ratio": _number,
    "freeflow_min": _number,
    "predicted_min": _number,
}


def identity_of(
    point: pd.Series[Any], columns: pd.Index[Any], carry: Sequence[str] = ()
) -> dict[str, object]:
    """The columns describing *which* point this is, shared by every outcome."""
    identity: dict[str, object] = {}
    if "block_id" in columns:
        # Block-level points group by grid block; carrying the key through keeps the
        # measurement joinable back to the block.
        identity["block_id"] = str(point["block_id"])
    identity.update(
        {
            "muni_code": str(point["muni_code"]),
            "muni_name": str(point["muni_name"]),
            "mesh_id": str(point["mesh_id"]),
            "lat": _number(point["lat"]),
            "lon": _number(point["lon"]),
        }
    )
    for column in carry:
        if column not in columns:
            raise KeyError(f"the plan carries {column!r}, which the point table does not have")
        identity[column] = _CARRY_TYPES.get(column, str)(point[column])
    return identity


def _destination_columns(group: str, destination: Point) -> dict[str, object]:
    """The four columns that say which destination a row was measured for."""
    return {
        "destination_group": group,
        "destination_name": destination.name,
        "destination_lat": destination.lat,
        "destination_lon": destination.lon,
    }


def measure_points(
    client: RoutesClient,
    points: pd.DataFrame,
    plan: MeasurementPlan,
    *,
    max_unrouted: int,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[dict[str, object]]:
    """Measure every (point, destination, schedule) and return one row for each.

    A point the router cannot reach is recorded with null values and a
    ``route_error`` rather than dropped: a dropped row is indistinguishable from
    one that was never measured. Past ``max_unrouted`` the run stops
    instead, because a revoked key and a series of unreachable points look the
    same from here and only one of them is worth paying to continue. How many is
    too many is the operator's budget decision, so the caller passes it.
    """
    records: list[dict[str, object]] = []
    unrouted: list[str] = []

    for position, (_, point) in enumerate(points.iterrows(), start=1):
        cell = (_number(point["lat"]), _number(point["lon"]))
        freeflow_min = _number(point["freeflow_min"])
        identity = identity_of(point, points.columns, plan.carry)
        for destination in plan.destinations:
            target_point = (destination.lat, destination.lon)
            columns = _destination_columns(plan.group, destination)
            for schedule in plan.schedules:
                # The schedule owns which end is the origin; no caller decides
                # it locally, because a local guess can measure an inbound
                # schedule as an outbound trip.
                origin, target = schedule.endpoints(cell, target_point)
                moment = next_reference_datetime(plan.today, schedule.daytype[0], schedule.time)
                try:
                    if schedule.is_arrival_anchored:
                        solution = client.route_for_arrival(
                            origin,
                            target,
                            moment,
                            initial_guess=timedelta(minutes=freeflow_min * 1.3),
                        )
                        result = solution.result
                        converged, iterations = solution.converged, solution.iterations
                        error_s: float | None = solution.error_s
                    else:
                        result = client.route(origin, target, moment)
                        converged, iterations, error_s = True, 1, None
                except RoutesError as failure:
                    unrouted.append(
                        f"{point['mesh_id']} {schedule.id} {destination.name}: {failure}"
                    )
                    records.append(
                        {
                            **identity,
                            **columns,
                            "fetched_at": None,
                            "schedule_id": schedule.id,
                            "direction": schedule.direction,
                            "measured_min": None,
                            "measured_distance_m": None,
                            "departure_time": None,
                            "arrival_time": None,
                            "solver_converged": False,
                            "solver_iterations": 0,
                            "solver_error_s": None,
                            "route_error": str(failure),
                            "source_stage": plan.stage,
                        }
                    )
                    if len(unrouted) > max_unrouted:
                        raise RuntimeError(
                            f"経路なしが {len(unrouted)} 件に達しました（上限 {max_unrouted}）。"
                            "個別地点の問題ではなく系統的な失敗の可能性があります: "
                            + "; ".join(unrouted[:3])
                        ) from failure
                    continue

                records.append(
                    {
                        **identity,
                        **columns,
                        # ``None`` when the answer came from a cache entry
                        # written before fetch times were recorded. Those entries have no fetch
                        # time of their own, and the read time is a different
                        # quantity; the provenance table carries theirs.
                        "fetched_at": (
                            result.fetched_at.isoformat() if result.fetched_at else None
                        ),
                        "schedule_id": schedule.id,
                        "direction": schedule.direction,
                        "measured_min": result.duration_min,
                        "measured_distance_m": result.distance_m,
                        "departure_time": result.departure_time.isoformat(),
                        "arrival_time": result.arrival_time.isoformat(),
                        "solver_converged": converged,
                        "solver_iterations": iterations,
                        "solver_error_s": error_s,
                        "route_error": None,
                        "source_stage": plan.stage,
                    }
                )
        if progress is not None:
            progress(position, len(points), client.billable_calls)

    return records


GROUP_COLUMN = "destination_group"


def merge_group(path: Path, rows: pd.DataFrame, *, group: str) -> pd.DataFrame:
    """Fold one group's freshly measured ``rows`` into the table at ``path``.

    The groups are measured in separate runs -- each has its own ceiling of
    billable calls -- so the second run writes into a file the first one made.
    Replacing that file would discard paid rows while reporting only that it
    wrote something.

    Rows of the same group are replaced rather than appended, so a re-run after
    an interruption is idempotent: cached answers are free, and a run
    that doubled its own rows would break the key it is written under.
    """
    if not path.is_file():
        return rows
    existing = pd.read_parquet(path)
    if GROUP_COLUMN not in existing.columns:
        raise ValueError(
            f"{path} has no {GROUP_COLUMN} column, so rows of {group!r} cannot be merged into "
            "it; it is not a measurement table of this shape"
        )
    kept = existing[existing[GROUP_COLUMN] != group]
    if kept.empty:
        return rows
    # The table on disk may have been backfilled with provenance, which adds
    # `batch_id` and turns `fetched_at` into a timestamp. Fresh rows
    # have neither. Concatenating the two produces object columns that parquet
    # refuses -- loudly, which is lucky, because the quiet version would be a
    # table whose fetch times are half strings. Say which columns disagree
    # and what to do about it instead.
    missing = sorted(set(kept.columns) - set(rows.columns))
    if missing:
        raise ValueError(
            f"{path} has columns the fresh rows do not: {missing}. They are derived "
            "by a provenance backfill, so merging would leave them null for "
            f"{group!r}. Delete the table and re-measure every group -- cached answers "
            "are free -- then run the backfill again"
        )
    return pd.concat([kept, rows], ignore_index=True)


def unrouted_notes(records: Sequence[dict[str, object]]) -> list[str]:
    """The rows that failed to route, as readable notes for the run summary."""
    return [
        f"{row['mesh_id']} {row['schedule_id']} {row['destination_name']}: {row['route_error']}"
        for row in records
        if row.get("route_error")
    ]
