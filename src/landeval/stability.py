"""Testing the premise used to reject repeat sampling.

Skipping multi-week sampling rests on one premise: *the API's future-time
answer is a prediction-model value and does not vary from day to day*. Two
things downstream lean on it. A band of measured times reads as a spatial
spread only if no part of it is time; and measurements fetched on different
days can be treated as one population only if the day does not matter.

Checking the premise is cheap, because the answer
is already stored: re-ask a query that is in the cache and see whether the same
number comes back. This module decides *which* queries to re-ask, refuses to
run once the stored departure is too close to be asked about again, and
fixes what the answers mean before any of them arrive.

Two things it will not do quietly. A schedule with fewer rows than strata is an
error rather than a shorter answer, and a query missing from the stored cache is
an error rather than a second billable call -- buying the "old" value today and
comparing it with the new one would compare a number with itself and report the
premise as holding.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import httpx
import pandas as pd

from landeval.basedate import DEFAULT_MIN_LEAD_DAYS
from landeval.config import Schedule
from landeval.routes import Coordinate, RoutesClient

IDENTICAL = "identical"
WITHIN_NOISE = "within-noise"
MOVES = "moves"

# Low, middle and high of each schedule's own duration distribution. Three is
# the smallest set that can show a trend across trip length rather than a pair
# of endpoints.
QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "schedule_id",
    "mesh_id",
    "lat",
    "lon",
    "departure_time",
    "measured_min",
    "measured_distance_m",
)


class StabilityError(Exception):
    """Raised when the check cannot be run as specified, rather than run smaller."""


class CacheOnlyTransport(httpx.BaseTransport):
    """A transport that cannot reach the network.

    Both sides of the comparison go through :class:`RoutesClient`, so both
    derive the cache key from the same code -- there is no second spelling of
    the key to drift. What makes the stored side *stored* is this: the
    client falls through to the network on a miss, and here that fall-through
    hits a wall instead of a charge.
    """

    # ARG002: the request is refused, never read.
    def handle_request(self, request: httpx.Request) -> httpx.Response:  # noqa: ARG002
        raise StabilityError(
            "this query is not in the stored cache; refusing to buy today's answer "
            "and compare it against itself"
        )


# --- choosing what to re-ask -------------------------------------------------


def _nearest_rank(quantile: float, count: int) -> int:
    """Index of the nearest-rank quantile in an ascending run of ``count`` rows."""
    return min(max(math.ceil(quantile * count) - 1, 0), count - 1)


def select_triples(frame: pd.DataFrame, *, quantiles: Sequence[float] = QUANTILES) -> pd.DataFrame:
    """Pick one row per schedule per quantile of that schedule's own durations.

    Stratifying inside each schedule is what makes the sample cover the axes
    that could break the premise at once: every fetch date, arrival
    back-solving against a fixed departure, both directions, the whole clock,
    and short trips against long ones.

    Selection uses no randomness, so the same table yields the same rows and
    the run can be repeated or argued with later.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise StabilityError(f"measurement frame is missing {missing}")
    unmeasured = int(frame["measured_min"].isna().sum())
    if unmeasured:
        raise StabilityError(
            f"{unmeasured} rows have no measured_min; filter them at the call site so the "
            "drop is visible, rather than losing a whole schedule here"
        )
    # groupby drops null keys by default, so these rows would leave the
    # population without a word -- the same silent drop the check above exists
    # to prevent, one column over.
    unlabelled = int(frame["schedule_id"].isna().sum())
    if unlabelled:
        raise StabilityError(
            f"{unlabelled} rows have no schedule_id; grouping would drop them silently "
            "and the run would re-ask a population it never reported"
        )

    picked: list[pd.DataFrame] = []
    for schedule_id, group in frame.groupby("schedule_id", sort=True):
        if len(group) < len(quantiles):
            raise StabilityError(
                f"schedule {schedule_id} has {len(group)} measured rows, fewer than the "
                f"{len(quantiles)} strata asked for"
            )
        # mesh_id breaks ties, so equal durations do not let row order decide.
        ordered = group.sort_values(["measured_min", "mesh_id"], kind="stable").reset_index(
            drop=True
        )
        rows = {}
        for quantile in quantiles:
            index = _nearest_rank(quantile, len(ordered))
            if index in rows:
                raise StabilityError(
                    f"schedule {schedule_id}: quantiles {list(quantiles)} collapse onto one row "
                    f"at {len(ordered)} rows"
                )
            rows[index] = quantile
        chosen = ordered.iloc[sorted(rows)].copy()
        chosen["quantile"] = [rows[index] for index in sorted(rows)]
        picked.append(chosen)

    if not picked:
        raise StabilityError(
            "the measurement frame holds no rows; there is nothing to re-ask, and an "
            "empty selection would make every check below pass trivially"
        )
    return (
        pd.concat(picked, ignore_index=True)
        .sort_values(["schedule_id", "quantile"], kind="stable")
        .reset_index(drop=True)
    )


# --- the deadline ------------------------------------------------------------


def _parse(stamp: str) -> datetime:
    try:
        return datetime.fromisoformat(str(stamp))
    except ValueError:
        raise StabilityError(f"departure {stamp!r} is not a timestamp") from None


def check_lead_time(
    departures: Iterable[str], today: date, *, min_lead_days: int = DEFAULT_MIN_LEAD_DAYS
) -> None:
    """Refuse the run unless every stored departure is still far enough ahead.

    Stored measurements name a reference date, so this is not a formality --
    it is the expiry. Once that date is inside the lead time, re-asking the same
    query blends live traffic into the answer and the difference stops
    being about the day the question was asked.

    ``today`` is compared against each departure's date *in that departure's own
    offset*, so the caller must pass the date in the same calendar -- a host
    running on UTC would otherwise read a JST departure as a day further off
    than it is, on exactly the day the check is supposed to start refusing.
    """
    too_close = []
    for stamp in departures:
        moment = _parse(stamp)
        lead = (moment.date() - today).days
        if lead < min_lead_days:
            too_close.append((stamp, lead))

    if too_close:
        soonest = min(lead for _, lead in too_close)
        raise StabilityError(
            f"{len(too_close)} of the selected departures are {soonest} days ahead or less, "
            f"under the {min_lead_days}-day minimum the lead-time rule requires; the stored "
            "answers can no longer be "
            "re-asked without blending live traffic into the comparison"
        )


# --- what the answers mean ---------------------------------------------------


def verdict(
    deltas: Sequence[float], *, noise_tolerance_s: float, systematic_tolerance_s: float
) -> str:
    """Read a set of second-differences against the fixed rule.

    The two tolerances are the caller's: the largest single difference may not
    exceed ``noise_tolerance_s``, and the mean may not exceed
    ``systematic_tolerance_s``.
    """
    if not len(deltas):
        raise StabilityError("no deltas to read a verdict from")
    unreadable = [delta for delta in deltas if not math.isfinite(delta)]
    if unreadable:
        # Every comparison below is False against a NaN, so the rule would fall
        # through to WITHIN_NOISE -- the reassuring answer -- for a difference
        # that could not be computed at all.
        raise StabilityError(f"{len(unreadable)} deltas are not finite; no verdict is readable")
    if all(delta == 0 for delta in deltas):
        return IDENTICAL
    if max(abs(delta) for delta in deltas) > noise_tolerance_s:
        return MOVES
    if abs(sum(deltas) / len(deltas)) > systematic_tolerance_s:
        return MOVES
    return WITHIN_NOISE


@dataclass(frozen=True)
class Summary:
    """The verdict together with the numbers it was read from."""

    verdict: str
    count: int
    identical: int
    max_abs_s: float
    mean_s: float
    median_s: float


def summarise(
    deltas: Sequence[float], *, noise_tolerance_s: float, systematic_tolerance_s: float
) -> Summary:
    return Summary(
        verdict=verdict(
            deltas,
            noise_tolerance_s=noise_tolerance_s,
            systematic_tolerance_s=systematic_tolerance_s,
        ),
        count=len(deltas),
        identical=sum(1 for delta in deltas if delta == 0),
        max_abs_s=max(abs(delta) for delta in deltas),
        mean_s=sum(deltas) / len(deltas),
        median_s=float(pd.Series(list(deltas)).median()),
    )


# --- asking twice ------------------------------------------------------------


def _endpoints(
    row: pd.Series, schedules: Mapping[str, Schedule], destination: Coordinate
) -> tuple[Coordinate, Coordinate]:
    schedule_id = str(row["schedule_id"])
    schedule = schedules.get(schedule_id)
    if schedule is None:
        raise StabilityError(
            f"schedule {schedule_id} is not in the configuration; which end of the trip is "
            "the origin cannot be guessed"
        )
    return schedule.endpoints((float(row["lat"]), float(row["lon"])), destination)


def _departure(row: pd.Series) -> datetime:
    stored = str(row["departure_time"])
    moment = _parse(stored)
    # The cache key is built from `departure.isoformat()`. A stored spelling
    # that re-serialises differently addresses a file that was never written,
    # and a miss is a charge.
    if moment.isoformat() != stored:
        raise StabilityError(
            f"departure {stored!r} does not round-trip through isoformat "
            f"(got {moment.isoformat()!r}); the cache key would not match"
        )
    return moment


def _agrees(row: pd.Series, column: str, table_value: int, stored_value: int, unit: str) -> None:
    """The parquet and the cache are two copies of one measurement.

    Checked for both quantities, not only duration: a stored distance that
    disagrees with the table would otherwise be reported downstream as a route
    the API changed, which is exactly the reading this run puts weight on.
    """
    if table_value != stored_value:
        raise StabilityError(
            f"the stored cache says {stored_value}{unit} where {column} says "
            f"{table_value}{unit} for {row['mesh_id']} / {row['schedule_id']}; the two copies "
            "of one measurement disagree"
        )


def compare_triples(
    selected: pd.DataFrame,
    *,
    schedules: Mapping[str, Schedule],
    destination: Coordinate,
    cached: RoutesClient,
    fresh: RoutesClient,
) -> pd.DataFrame:
    """Ask each selected query of the stored cache and of the API, side by side.

    ``cached`` must be pointed at the stored cache and given a transport that
    cannot answer; ``fresh`` must be pointed at a *different* and **empty**
    directory. Two rules follow from the two clients sharing a cache key.

    Writing the new answer into the stored cache would delete the value being
    compared against, which is why the directories differ. And ``fresh`` must
    actually reach the API for every row: :meth:`RoutesClient.route` answers
    from its own cache before it posts, so a second run against a directory a
    previous run filled would re-read that run's answers, recompute the same
    verdict, and stamp it with today's date -- the whole comparison replayed
    with nothing measured. That is checked per row rather than trusted.
    """
    records: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        origin, target = _endpoints(row, schedules, destination)
        departure = _departure(row)

        stored = cached.route(origin, target, departure)
        _agrees(row, "measured_min", round(float(row["measured_min"]) * 60), stored.duration_s, "s")
        _agrees(
            row,
            "measured_distance_m",
            round(float(row["measured_distance_m"])),
            stored.distance_m,
            "m",
        )

        billed = fresh.billable_calls
        again = fresh.route(origin, target, departure)
        if fresh.billable_calls == billed:
            raise StabilityError(
                f"the re-fetch for {row['mesh_id']} / {row['schedule_id']} was answered from a "
                "cache rather than the API; point the re-fetch directory at an empty one. "
                "Re-using a previous run's answers would replay that run's day and report it "
                "as today's"
            )
        records.append(
            {
                "schedule_id": str(row["schedule_id"]),
                "quantile": float(row["quantile"]),
                "mesh_id": str(row["mesh_id"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "direction": schedules[str(row["schedule_id"])].direction,
                "departure_time": departure.isoformat(),
                "cached_duration_s": stored.duration_s,
                "fresh_duration_s": again.duration_s,
                "delta_s": again.duration_s - stored.duration_s,
                "cached_distance_m": stored.distance_m,
                "fresh_distance_m": again.distance_m,
                "delta_distance_m": again.distance_m - stored.distance_m,
            }
        )

    return pd.DataFrame.from_records(records)
