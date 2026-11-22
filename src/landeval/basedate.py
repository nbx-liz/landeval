"""Choosing the reference date a routing query is asked about.

The Routes API answers a future timestamp from a historical model rather than
from live conditions, so the query does not describe one particular day -- it
describes the population of days like the one asked for. That makes the date
choice a modelling decision, not a formality: a holiday, a day adjacent to one,
or a Friday inside Golden Week or Obon belongs to a different population than an
ordinary Friday.

Two rules follow, and both are enforced here rather than left to the caller:

* the date must be far enough ahead that live traffic is not blended in;
* the date must be *ordinary* -- outside every exclusion window.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Literal

import jpholiday

DayType = Literal["friday", "weekday", "weekend", "holiday"]

JST = timezone(timedelta(hours=9))

# Far enough ahead that the API answers from its historical model. Two days
# clears "today and tomorrow", where live conditions still dominate.
DEFAULT_MIN_LEAD_DAYS = 2

# How far forward to look for an ordinary date before giving up. Long enough to
# step over Golden Week and the New Year period from just outside them.
DEFAULT_SEARCH_DAYS = 90

# (month, day) ranges whose traffic does not represent an ordinary week.
# Inclusive on both ends.
_SEASONAL_EXCLUSIONS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((12, 25), (1, 5)),  # year end and new year (wraps the year boundary)
    ((4, 27), (5, 6)),  # Golden Week
    ((8, 10), (8, 17)),  # Obon
)


class BaseDateError(Exception):
    """Raised for an unknown daytype, a bad lead time, or an exhausted search."""


def _in_seasonal_exclusion(day: date) -> bool:
    for (start_month, start_day), (end_month, end_day) in _SEASONAL_EXCLUSIONS:
        start = (start_month, start_day)
        end = (end_month, end_day)
        current = (day.month, day.day)
        if start <= end:
            if start <= current <= end:
                return True
        elif current >= start or current <= end:  # window wraps past December
            return True
    return False


def is_ordinary(day: date) -> bool:
    """Whether ``day`` belongs to an ordinary week.

    A public holiday is excluded, and so is each day touching one: traffic on
    the eve and the morning after does not look like an ordinary day either.
    """
    if _in_seasonal_exclusion(day):
        return False
    one_day = timedelta(days=1)
    return not any(
        jpholiday.is_holiday(day + offset) for offset in (-one_day, timedelta(0), one_day)
    )


def _matches(day: date, daytype: DayType) -> bool:
    weekday = day.weekday()
    if daytype == "friday":
        return weekday == 4
    if daytype == "weekday":
        return weekday <= 4
    if daytype == "weekend":
        return weekday >= 5
    if daytype == "holiday":
        # "Holiday" means a day people are off work: a weekend or a public
        # holiday. is_ordinary() rejects public holidays, so only weekends can
        # satisfy both -- which is the intended reading for leisure trips.
        return weekday >= 5
    raise BaseDateError(f"unknown daytype {daytype!r}")


def reference_date(
    today: date,
    daytype: DayType,
    *,
    min_lead_days: int = DEFAULT_MIN_LEAD_DAYS,
    search_days: int = DEFAULT_SEARCH_DAYS,
) -> date:
    """Return the first ordinary date of the requested kind at least ``min_lead_days`` away."""
    if min_lead_days < 0:
        raise BaseDateError(f"min_lead_days must not be negative, got {min_lead_days}")
    _matches(today, daytype)  # validates the daytype before searching

    for offset in range(min_lead_days, min_lead_days + search_days + 1):
        candidate = today + timedelta(days=offset)
        if _matches(candidate, daytype) and is_ordinary(candidate):
            return candidate

    raise BaseDateError(
        f"no ordinary {daytype!r} date found within {search_days} days of {today.isoformat()}"
    )


def next_reference_datetime(
    today: date,
    daytype: DayType,
    wall_clock: str,
    *,
    min_lead_days: int = DEFAULT_MIN_LEAD_DAYS,
    search_days: int = DEFAULT_SEARCH_DAYS,
) -> datetime:
    """Combine :func:`reference_date` with a ``HH:MM`` local time, in JST.

    The result is timezone-aware on purpose. A naive timestamp serialises
    without an offset and the API reads it as UTC, shifting every query nine
    hours -- a 10:00 arrival would be modelled as a 19:00 one.
    """
    day = reference_date(today, daytype, min_lead_days=min_lead_days, search_days=search_days)
    hour, minute = int(wall_clock[:2]), int(wall_clock[3:])
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=JST)
