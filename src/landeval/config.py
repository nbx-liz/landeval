"""Destination and threshold configuration.

The configuration is human-edited YAML. It is therefore an external open
grammar, and every off-contract input must be rejected loudly instead
of being tolerated or silently defaulted. All models forbid extra keys, and
cross-field invariants are checked here rather than at the point of use.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Bounding box of the Japanese territory, used to reject transposed or
# placeholder coordinates (a lat/lon swap lands far outside it).
JAPAN_LAT_RANGE = (20.0, 46.5)
JAPAN_LON_RANGE = (122.0, 154.0)

ScheduleType = Literal["arrive", "depart"]
# Which way the trip runs. `type` anchors the clock and says nothing about
# direction, so leaving it implicit would let an evening schedule meant as the
# trip home be measured as a trip *towards* the destination.
Direction = Literal["outbound", "inbound"]
DayType = Literal["friday", "weekday", "weekend", "holiday"]
Aggregate = Literal["single", "min", "mean", "max", "each"]

TIME_PATTERN = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"


class ConfigError(Exception):
    """Raised for any configuration that cannot be read or does not satisfy the contract."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Point(_Strict):
    """A single destination coordinate."""

    name: str = Field(min_length=1)
    lat: float
    lon: float

    @model_validator(mode="after")
    def _within_japan(self) -> Self:
        lat_ok = JAPAN_LAT_RANGE[0] <= self.lat <= JAPAN_LAT_RANGE[1]
        lon_ok = JAPAN_LON_RANGE[0] <= self.lon <= JAPAN_LON_RANGE[1]
        if not (lat_ok and lon_ok):
            raise ValueError(
                f"coordinate ({self.lat}, {self.lon}) for {self.name!r} is outside Japan "
                f"(lat {JAPAN_LAT_RANGE}, lon {JAPAN_LON_RANGE})"
            )
        return self


class Schedule(_Strict):
    """A time anchor for a routing query.

    ``arrive`` anchors are the expensive ones: the Google Routes API ignores
    ``arrivalTime`` for DRIVE mode, so they require back-solving the departure
    time.

    ``type`` and ``direction`` are separate on purpose. ``type`` says which end
    of the trip the clock is pinned to; ``direction`` says which end is the
    origin. Reading one from the other looks reasonable for the commute -- an
    evening departure is obviously the trip home -- and is wrong for every
    other group, where an evening departure is leaving the destination. Left
    implicit in the config, that inference measures an evening schedule as a
    trip towards the destination whenever it guesses wrong.
    """

    id: str = Field(min_length=1)
    type: ScheduleType
    time: Annotated[str, Field(pattern=TIME_PATTERN)]
    daytype: list[DayType] = Field(min_length=1)
    # Required, with no default, for the same reason ``union_ratio`` requires
    # its ``unknown`` mask: a default is what a new schedule silently inherits
    # when its author never considered the question, and inheriting
    # ``outbound`` would silently measure a return trip in the wrong direction.
    # Every schedule must state which way its trip runs.
    direction: Direction

    @field_validator("daytype", mode="before")
    @classmethod
    def _as_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @property
    def is_arrival_anchored(self) -> bool:
        return self.type == "arrive"

    @property
    def is_outbound(self) -> bool:
        return self.direction == "outbound"

    def endpoints(
        self, cell: tuple[float, float], destination: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (origin, destination) for this schedule.

        Every caller routes between a cell and a fixed destination, and this is
        the single place that decides which is which -- so a new measurement
        script cannot get the direction wrong by forgetting to ask.
        """
        return (cell, destination) if self.is_outbound else (destination, cell)

    @property
    def hour(self) -> int:
        return int(self.time[:2])

    @property
    def minute(self) -> int:
        return int(self.time[3:])


class DestinationGroup(_Strict):
    """A set of destinations that share time anchors and an aggregation rule."""

    aggregate: list[Aggregate] = Field(min_length=1)
    used_for_screening: bool
    points: list[Point] = Field(min_length=1)
    schedules: list[Schedule] = Field(min_length=1)

    @field_validator("aggregate", mode="before")
    @classmethod
    def _as_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [p.name for p in self.points]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate point name in group: {sorted(_repeated(names))}")

        ids = [s.id for s in self.schedules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate schedule id in group: {sorted(_repeated(ids))}")

        if "single" in self.aggregate:
            if len(self.aggregate) != 1:
                raise ValueError("aggregate 'single' cannot be combined with other rules")
            if len(self.points) != 1:
                raise ValueError(
                    f"aggregate 'single' requires exactly one point, got {len(self.points)}"
                )
        if "min" in self.aggregate and len(self.points) < 2:
            raise ValueError("aggregate 'min' requires at least two points to choose between")
        return self

    def schedule_ids(self) -> set[str]:
        return {s.id for s in self.schedules}


class CommuteThresholds(_Strict):
    """Screening thresholds, applied at display time rather than baked in."""

    target_schedule: str = Field(min_length=1)
    good_max_min: int = Field(gt=0)
    hard_max_min: int = Field(gt=0)
    exclude_min: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.good_max_min < self.hard_max_min < self.exclude_min:
            raise ValueError(
                "thresholds must be strictly increasing: "
                f"good_max_min ({self.good_max_min}) < hard_max_min ({self.hard_max_min}) "
                f"< exclude_min ({self.exclude_min})"
            )
        return self


class Thresholds(_Strict):
    commute: CommuteThresholds


class OriginGrid(_Strict):
    mesh_size_m: int = Field(gt=0)
    bbox_source: Literal["freeflow_minutes"]
    freeflow_minutes: int = Field(gt=0)


class Region(_Strict):
    """The part of the country a run covers, for the scripts that select by region.

    ``prefectures`` maps the two-digit prefecture code to its name; ``bbox`` is
    ``(lon_min, lat_min, lon_max, lat_max)``. Both are required with no default,
    so a script that needs a region fails on a configuration that never stated
    one instead of covering a region nobody chose.
    """

    prefectures: dict[str, str] = Field(min_length=1)
    bbox: tuple[float, float, float, float]

    @field_validator("prefectures")
    @classmethod
    def _codes_are_two_digits(cls, value: dict[str, str]) -> dict[str, str]:
        bad = sorted(code for code in value if not re.fullmatch(r"[0-9]{2}", code))
        if bad:
            raise ValueError(f"prefecture codes must be two digits, got {bad}")
        return value

    @model_validator(mode="after")
    def _bbox_is_ordered_and_in_japan(self) -> Self:
        lon_min, lat_min, lon_max, lat_max = self.bbox
        if not (lon_min < lon_max and lat_min < lat_max):
            raise ValueError(f"bbox must be (lon_min, lat_min, lon_max, lat_max), got {self.bbox}")
        inside = (
            JAPAN_LON_RANGE[0] <= lon_min
            and lon_max <= JAPAN_LON_RANGE[1]
            and JAPAN_LAT_RANGE[0] <= lat_min
            and lat_max <= JAPAN_LAT_RANGE[1]
        )
        if not inside:
            raise ValueError(f"bbox {self.bbox} is outside Japan (lon/lat swapped?)")
        return self


class Vehicle(_Strict):
    """What the driver actually has, for the toll question.

    Declaring a toll pass changes the amount the API returns -- the same
    route is priced differently with and without a pass -- so this is not
    cosmetic. It is optional because only a measurement run asks about tolls, and
    the measurement path refuses rather than defaulting: guessing the cash rate
    for a driver who has an ETC unit would answer a question nobody asked.

    ``toll_passes`` is validated against the values `landeval.tolls` has
    actually executed. An unmeasured spelling is refused there rather than sent.

    Both fields are **required with no default**, for the reason ``direction``
    on a schedule is: a default is what a new configuration silently inherits
    when its author never considered the question, and inheriting "no pass"
    means being quoted the cash rate. A driver who genuinely has no unit writes
    ``toll_passes: []`` and has then said so.
    """

    toll_passes: list[str]
    emission_type: str


class Config(_Strict):
    origin_grid: OriginGrid
    thresholds: Thresholds
    destinations: dict[str, DestinationGroup] = Field(min_length=1)
    # Absent in every configuration written before tolls existed, and absent is
    # readable: a measurement run says so and stops. A default here would be a cash
    # price presented as this driver's.
    vehicle: Vehicle | None = None
    # Absent unless a script that selects by region is used; those scripts say
    # so and stop rather than assume an area.
    region: Region | None = None

    @model_validator(mode="after")
    def _schedule_ids_are_unique_across_groups(self) -> Self:
        """A schedule id names one schedule in the whole configuration.

        Measurement tables are keyed by ``schedule_id`` without the destination
        group (``schema.py``), so one id in two groups would make two different
        trips indistinguishable there. Duplicates inside one group are refused
        by the group's own validator; this refuses the ones across groups.
        """
        owners: dict[str, list[str]] = {}
        for name, group in self.destinations.items():
            for schedule_id in group.schedule_ids():
                owners.setdefault(schedule_id, []).append(name)
        shared = {sid: groups for sid, groups in owners.items() if len(groups) > 1}
        if shared:
            detail = "; ".join(
                f"{sid!r} in {sorted(groups)}" for sid, groups in sorted(shared.items())
            )
            raise ValueError(f"schedule id appears in more than one destination group: {detail}")
        return self

    @model_validator(mode="after")
    def _screening_target_resolves(self) -> Self:
        screening = [name for name, g in self.destinations.items() if g.used_for_screening]
        if len(screening) != 1:
            raise ValueError(
                "exactly one destination group must set used_for_screening: true, "
                f"found {len(screening)}: {sorted(screening)}"
            )

        group = self.destinations[screening[0]]
        target = self.thresholds.commute.target_schedule
        if target not in group.schedule_ids():
            raise ValueError(
                f"thresholds.commute.target_schedule {target!r} is not a schedule of the "
                f"screening group {screening[0]!r} (available: {sorted(group.schedule_ids())})"
            )
        return self

    def screening_group(self) -> tuple[str, DestinationGroup]:
        """Return the single group used for exclusion screening.

        The single-group invariant is enforced at validation time, so this
        cannot fail for a loaded configuration.
        """
        for name, group in self.destinations.items():
            if group.used_for_screening:
                return name, group
        raise AssertionError("unreachable: validated configuration has a screening group")


def _repeated(values: list[str]) -> set[str]:
    seen: set[str] = set()
    return {v for v in values if v in seen or bool(seen.add(v))}  # type: ignore[func-returns-value]


def load_config(path: Path) -> Config:
    """Load and validate the destination configuration.

    Raises:
        ConfigError: the file is missing, unparseable, or violates the contract.
    """
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML in {path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"configuration file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be a mapping, got {type(raw).__name__}: {path}")

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}:\n{_format(exc)}") from exc


def _format(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']} [{err['type']}]")
    return "\n".join(lines)
