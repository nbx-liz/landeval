"""Adopted values for one address, bought on request.

The **adopted value** comes only from the Google Routes API and only when the
user asks for it. It is never filled from a free, locally computed figure:
only a call made here produces it.

This is not the batch measurement in `landeval.measure`. That one fills
parquet tables over many points and its schema is pinned; this
one answers one address, returns values rather than rows, and changes no
artefact. What the two share -- which end of a trip is the origin, and which
instant a schedule names -- lives in `Schedule.endpoints` and
`landeval.basedate`, so the decisions are made once even though the loops are
separate.

Every call carries the toll question. The toll estimate is an extra
computation on the same route request, so asking separately would cost one more
call per (destination, schedule) and buy nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from landeval.basedate import next_reference_datetime
from landeval.config import Config, DayType, Direction, Point, Schedule, ScheduleType
from landeval.routes import Coordinate, RoutesClient, RoutesError
from landeval.tolls import RequestShape, TollError

# Used when nothing better is known about the trip length. The batch path
# passes a free-flow estimate; a single address may have none, for a group
# destination that has no free-flow layer at all.
DEFAULT_INITIAL_GUESS_MIN = 60.0


class AdoptedError(Exception):
    """Raised when a measurement cannot be attempted at all."""


@dataclass(frozen=True)
class AdoptedValue:
    """One adopted measurement: a real answer for a real trip.

    ``daytype`` is carried and printed rather than left implicit. A toll
    differs between a weekday and a holiday on the same route, so a toll
    without the day it was measured on cannot be read.
    """

    group: str
    destination: str
    schedule_id: str
    # The clock the schedule actually names, carried rather than left to be
    # inferred from the id. `{id: arrive_0930, type: depart, time: "10:30"}` is
    # valid configuration, and printing the id alone would invite the reader to
    # believe the id. The id is a label; these two are the question.
    anchor: ScheduleType
    time: str
    direction: Direction
    daytype: DayType
    # What the schedule declared, when it declared more than one. The first is
    # the one measured; naming the rest keeps a two-daytype schedule from
    # looking like a one-daytype schedule in the output.
    declared_daytypes: tuple[DayType, ...]
    departure_time: datetime
    duration_s: int
    distance_m: int
    toll_yen: int | None
    converged: bool
    fetched_at: datetime | None

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60.0

    @property
    def arrival_time(self) -> datetime:
        return self.departure_time + timedelta(seconds=self.duration_s)


@dataclass(frozen=True)
class AdoptedFailure:
    """A leg that could not be measured, kept rather than dropped.

    A dropped leg is indistinguishable from one that was never asked for, and an output
    listing three destinations when four were requested reads as a complete answer.
    """

    group: str
    destination: str
    schedule_id: str
    reason: str


@dataclass(frozen=True)
class AdoptedMeasurement:
    """Everything one measurement run produced, including what it could not."""

    origin: Coordinate
    shape: RequestShape
    values: tuple[AdoptedValue, ...]
    failures: tuple[AdoptedFailure, ...]
    billable_calls: int

    @property
    def unsettled(self) -> tuple[AdoptedValue, ...]:
        """Legs whose arrival back-solve never landed on the target.

        The API answered, and the duration is real for the departure that was
        sent -- but that departure is not the one the schedule asked about. The
        number is a measurement of a different question, so it is separated
        from the ones that answer theirs.
        """
        return tuple(value for value in self.values if not value.converged)

    @property
    def complete(self) -> bool:
        """Every leg asked for produced an answer to the question it asked.

        A non-converged solve counts against this. Otherwise the run would report
        itself complete while carrying a duration for a departure nobody had
        asked about -- with a note beside it, which is not the same as the run
        knowing it is short of an answer.
        """
        return not self.failures and not self.unsettled


def request_shape(config: Config) -> RequestShape:
    """The toll question this configuration says to ask.

    Refuses rather than defaulting. A configuration with no ``vehicle`` block
    has not said which unit the driver has, and the amount differs substantially
    between declaring one and not; printing the cash figure for an ETC
    driver would be an answer to a question nobody asked.
    """
    vehicle = config.vehicle
    if vehicle is None:
        raise AdoptedError(
            "the configuration has no `vehicle:` block, so the toll question cannot be "
            "asked. Add toll_passes (e.g. [JP_ETC]) and emission_type, or the measured "
            "amount would be the cash rate rather than this driver's"
        )
    try:
        return RequestShape(
            tolls=True,
            toll_passes=tuple(vehicle.toll_passes),
            emission_type=vehicle.emission_type,
        )
    except TollError as exc:
        raise AdoptedError(f"vehicle configuration cannot be sent: {exc}") from None


def legs(config: Config) -> list[tuple[str, Point, Schedule]]:
    """Every (group, destination, schedule) one address is measured over.

    Flat and ordered so the caller can count the work before paying for it:
    A measurement run bills once per leg, plus the extra departures each
    arrival-anchored leg needs to back-solve.
    """
    plan: list[tuple[str, Point, Schedule]] = []
    for name, group in config.destinations.items():
        for destination in group.points:
            for schedule in group.schedules:
                plan.append((name, destination, schedule))
    return plan


def measure_address(
    client: RoutesClient,
    config: Config,
    origin: Coordinate,
    *,
    today: datetime | None = None,
    freeflow_min: float | None = None,
    shape: RequestShape | None = None,
) -> AdoptedMeasurement:
    """Measure ``origin`` against every destination and return adopted values.

    ``freeflow_min`` seeds the arrival back-solve when the caller has a free-flow
    estimate for this address; without it the solver starts from a flat guess
    and usually pays one extra call.
    """
    asked = request_shape(config) if shape is None else shape
    reference = (today or datetime.now()).date()
    # The client counts for its whole lifetime, and one client serves every
    # address in a run. Reporting its running total per address would tell the
    # second address it cost what the first two did together.
    billed_before = client.billable_calls

    values: list[AdoptedValue] = []
    failures: list[AdoptedFailure] = []
    for group, destination, schedule in legs(config):
        target = (destination.lat, destination.lon)
        # The schedule owns which end is the origin. A caller that decided it
        # locally could measure an inbound schedule as an outbound trip, so
        # this call is the only place it is decided.
        start, end = schedule.endpoints(origin, target)
        moment = next_reference_datetime(reference, schedule.daytype[0], schedule.time)
        try:
            value = _measure_leg(
                client,
                start,
                end,
                moment,
                shape=asked,
                schedule=schedule,
                group=group,
                destination=destination,
                freeflow_min=freeflow_min,
            )
        except RoutesError as failure:
            failures.append(
                AdoptedFailure(
                    group=group,
                    destination=destination.name,
                    schedule_id=schedule.id,
                    reason=str(failure),
                )
            )
            continue
        values.append(value)

    return AdoptedMeasurement(
        origin=origin,
        shape=asked,
        values=tuple(values),
        failures=tuple(failures),
        billable_calls=client.billable_calls - billed_before,
    )


def _measure_leg(
    client: RoutesClient,
    start: Coordinate,
    end: Coordinate,
    moment: datetime,
    *,
    shape: RequestShape,
    schedule: Schedule,
    group: str,
    destination: Point,
    freeflow_min: float | None,
) -> AdoptedValue:
    """One leg, arrival-anchored or not."""
    if schedule.is_arrival_anchored:
        guess = timedelta(minutes=(freeflow_min or DEFAULT_INITIAL_GUESS_MIN) * 1.3)
        solution = client.route_for_arrival(start, end, moment, initial_guess=guess, shape=shape)
        result, converged = solution.result, solution.converged
    else:
        result = client.route(start, end, moment, shape=shape)
        converged = True

    return AdoptedValue(
        group=group,
        destination=destination.name,
        schedule_id=schedule.id,
        anchor=schedule.type,
        time=schedule.time,
        direction=schedule.direction,
        daytype=schedule.daytype[0],
        declared_daytypes=tuple(schedule.daytype),
        departure_time=result.departure_time,
        duration_s=result.duration_s,
        distance_m=result.distance_m,
        toll_yen=result.toll_yen,
        converged=converged,
        fetched_at=result.fetched_at,
    )
