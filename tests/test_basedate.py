"""Tests for reference-date selection.

The API's answer for a future timestamp is a prediction built from historical
patterns, so the date chosen decides which population that prediction describes.
A holiday, the day beside one, or a long-weekend Friday is a different traffic
population from an ordinary Friday, and averaging it in silently would make the
whole batch describe something nobody asked about.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from landeval.basedate import (
    BaseDateError,
    is_ordinary,
    next_reference_datetime,
    reference_date,
)

# 2031-08-13 is a Wednesday. 2031-08-15, the Friday of that week, is inside the
# Obon window.
TODAY = date(2031, 8, 13)


# --- ordinary-day classification -------------------------------------------


def test_a_plain_friday_is_ordinary() -> None:
    # Not 2031-08-15: that Friday sits inside the Obon window.
    assert is_ordinary(date(2031, 6, 13)) is True


def test_a_public_holiday_is_not_ordinary() -> None:
    # 2031-01-01 元日
    assert is_ordinary(date(2031, 1, 1)) is False


def test_the_day_before_a_holiday_is_not_ordinary() -> None:
    assert is_ordinary(date(2030, 12, 31)) is False


def test_the_day_after_a_holiday_is_not_ordinary() -> None:
    assert is_ordinary(date(2031, 1, 2)) is False


@pytest.mark.parametrize(
    "day",
    [
        date(2031, 4, 29),  # 昭和の日 -- Golden Week
        date(2031, 5, 4),  # GW中
        date(2031, 8, 13),  # お盆
        date(2031, 8, 15),  # お盆
        date(2031, 12, 30),  # 年末
        date(2031, 1, 3),  # 年始
    ],
)
def test_seasonal_exclusion_periods_are_not_ordinary(day: date) -> None:
    assert is_ordinary(day) is False


@pytest.mark.parametrize("day", [date(2031, 6, 13), date(2031, 9, 12), date(2031, 11, 14)])
def test_plain_fridays_outside_the_exclusions_are_ordinary(day: date) -> None:
    assert is_ordinary(day) is True


# --- weekday selection -----------------------------------------------------


def test_selects_the_next_ordinary_friday() -> None:
    chosen = reference_date(TODAY, "friday", min_lead_days=2)

    assert chosen.weekday() == 4
    assert is_ordinary(chosen)


def test_respects_the_minimum_lead_time() -> None:
    """A date too close to now would be answered with live traffic."""
    chosen = reference_date(TODAY, "friday", min_lead_days=2)

    assert (chosen - TODAY).days >= 2


def test_skips_a_friday_inside_an_exclusion_window() -> None:
    """The first Friday the lead time allows, 2031-08-15, falls in the Obon
    window, so the search has to pass over it to the next ordinary Friday."""
    chosen = reference_date(date(2031, 8, 13), "friday", min_lead_days=2)

    assert chosen == date(2031, 8, 22)
    assert is_ordinary(chosen)


def test_weekend_selects_a_saturday_or_sunday() -> None:
    chosen = reference_date(TODAY, "weekend", min_lead_days=2)

    assert chosen.weekday() in (5, 6)
    assert is_ordinary(chosen)


def test_weekday_selects_a_monday_to_friday() -> None:
    chosen = reference_date(TODAY, "weekday", min_lead_days=2)

    assert chosen.weekday() <= 4
    assert is_ordinary(chosen)


def test_holiday_selects_an_actual_public_holiday() -> None:
    """The 'holiday' daytype means a day people are off, unlike the others."""
    chosen = reference_date(TODAY, "holiday", min_lead_days=2)

    assert chosen.weekday() in (5, 6) or not is_ordinary(chosen)


def test_selection_is_deterministic() -> None:
    a = reference_date(TODAY, "friday", min_lead_days=2)
    b = reference_date(TODAY, "friday", min_lead_days=2)

    assert a == b


def test_unknown_daytype_is_rejected() -> None:
    with pytest.raises(BaseDateError, match="daytype"):
        reference_date(TODAY, "someday", min_lead_days=2)  # type: ignore[arg-type]


def test_negative_lead_is_rejected() -> None:
    with pytest.raises(BaseDateError, match="lead"):
        reference_date(TODAY, "friday", min_lead_days=-1)


def test_no_candidate_within_the_search_window_raises() -> None:
    """Failing to find a date must be loud, not silently answered with today."""
    with pytest.raises(BaseDateError, match="no ordinary"):
        reference_date(TODAY, "friday", min_lead_days=2, search_days=3)


# --- datetime assembly -----------------------------------------------------


def test_combines_the_date_with_the_wall_clock_time() -> None:
    moment = next_reference_datetime(TODAY, "friday", "10:00", min_lead_days=2)

    assert isinstance(moment, datetime)
    assert (moment.hour, moment.minute) == (10, 0)
    assert moment.weekday() == 4


def test_result_carries_the_japan_offset() -> None:
    """A naive timestamp would be read as UTC by the API and shift the answer 9 h."""
    moment = next_reference_datetime(TODAY, "friday", "10:00", min_lead_days=2)

    assert moment.tzinfo is not None
    assert moment.utcoffset() is not None
    assert moment.utcoffset().total_seconds() == 9 * 3600  # type: ignore[union-attr]


def test_rfc3339_form_is_what_the_api_expects() -> None:
    moment = next_reference_datetime(TODAY, "friday", "10:00", min_lead_days=2)

    assert moment.isoformat().endswith("+09:00")
