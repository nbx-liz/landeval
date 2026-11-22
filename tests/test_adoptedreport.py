"""The adopted-value block, which must not read as an estimate.

This project's answers come in two kinds: a free estimate with an error
band, and an adopted value bought from the API. The failure mode is not
arithmetic -- it is a layout where the two sit close enough that the reader
carries the first heading over the second. So the block says what it is, says
which toll question was asked, and never prints a missing amount as zero.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from landeval.adopted import AdoptedFailure, AdoptedMeasurement, AdoptedValue
from landeval.adoptedreport import format_adopted
from landeval.basedate import JST
from landeval.tolls import DEFAULT_SHAPE, RequestShape

ORIGIN = (35.21, 140.05)
FRIDAY = datetime(2030, 6, 14, 8, 0, tzinfo=JST)
ASKED = RequestShape(tolls=True, toll_passes=("JP_ETC",), emission_type="HYBRID")


def value(
    *,
    group: str = "commute",
    destination: str = "Central Station",
    schedule_id: str = "arr_0930",
    anchor: str = "arrive",
    time: str = "09:30",
    direction: str = "outbound",
    daytype: str = "friday",
    declared: tuple[str, ...] = ("friday",),
    duration_s: int = 2730,
    toll_yen: int | None = 1740,
    converged: bool = True,
) -> AdoptedValue:
    return AdoptedValue(
        group=group,
        destination=destination,
        schedule_id=schedule_id,
        anchor=anchor,  # type: ignore[arg-type]
        time=time,
        direction=direction,  # type: ignore[arg-type]
        daytype=daytype,  # type: ignore[arg-type]
        declared_daytypes=declared,  # type: ignore[arg-type]
        departure_time=FRIDAY,
        duration_s=duration_s,
        distance_m=38250,
        toll_yen=toll_yen,
        converged=converged,
        fetched_at=None,
    )


def run_of(
    *values: AdoptedValue,
    shape: RequestShape = ASKED,
    failures: tuple[AdoptedFailure, ...] = (),
    calls: int = 9,
) -> AdoptedMeasurement:
    return AdoptedMeasurement(
        origin=ORIGIN,
        shape=shape,
        values=values,
        failures=failures,
        billable_calls=calls,
    )


def text_of(run: AdoptedMeasurement) -> str:
    return "\n".join(format_adopted(run))


# --- keeping the two tiers apart -------------------------------------------


def test_the_block_names_itself_as_measured_not_estimated() -> None:
    out = text_of(run_of(value()))

    assert "採用値" in out
    assert "実測" in out


def test_the_block_says_the_estimate_is_a_different_quantity() -> None:
    """The reader has just seen an estimate with
    a percentile and an error band; nothing here may inherit those."""
    out = text_of(run_of(value()))

    assert "推定値とは別" in out
    assert "誤差幅" in out


def test_the_block_reports_what_it_billed() -> None:
    assert "14 呼び出し" in text_of(run_of(value(), calls=14))


# --- the toll, and the conditions it was measured under ---------------------


def test_a_toll_is_printed_with_the_pass_that_was_declared() -> None:
    """A cash amount and a pass amount differ on the same route. The amount
    alone is not an answer."""
    out = text_of(run_of(value()))

    assert "JP_ETC" in out
    assert "1,740 円" in out


def test_a_toll_is_printed_with_the_daytype_it_was_measured_on() -> None:
    """A weekday amount and a weekend amount differ, same pass, same route."""
    out = text_of(run_of(value(daytype="weekend")))

    assert "週末" in out


def test_a_cash_measurement_says_it_declared_no_pass() -> None:
    out = text_of(run_of(value(), shape=RequestShape(tolls=True)))

    assert "現金相当" in out


def test_a_run_that_did_not_ask_says_the_toll_is_missing_not_zero() -> None:
    out = text_of(run_of(value(toll_yen=None), shape=DEFAULT_SHAPE))

    assert "欠損であって 0 円ではありません" in out


def test_a_missing_amount_prints_as_unknown() -> None:
    out = text_of(run_of(value(toll_yen=None)))

    assert "通行料金 不明" in out
    assert "0 円" not in out


def test_a_free_route_prints_as_zero() -> None:
    """Zero is a real answer when the question was asked, and it is not the
    same as the line above."""
    out = text_of(run_of(value(toll_yen=0)))

    assert "通行料金 0 円" in out
    assert "不明" not in out


# --- what did not work ------------------------------------------------------


def test_a_leg_that_could_not_be_measured_is_named() -> None:
    """A block listing one destination when two were asked for reads as a
    complete answer, and nothing in it says otherwise."""
    out = text_of(
        run_of(
            value(),
            failures=(
                AdoptedFailure(
                    group="group_a",
                    destination="Site B",
                    schedule_id="group_a_depart_0630",
                    reason="Routes returned no route for the requested pair",
                ),
            ),
        )
    )

    assert "測れなかった経路 1 件" in out
    assert "Site B" in out
    assert "欠測であって 0 ではありません" in out


def test_a_complete_run_carries_no_failure_section() -> None:
    assert "測れなかった" not in text_of(run_of(value()))


def test_a_solve_that_did_not_converge_says_the_number_is_not_the_answer() -> None:
    """The solver returns its best attempt at a *different* departure, so the
    duration answers a question nobody asked. Noting the solve was imperfect is
    not the same as saying the number does not answer the schedule."""
    out = text_of(run_of(value(converged=False)))

    assert "指定時刻の答えではありません" in out


def test_an_unconverged_run_is_not_complete() -> None:
    """A run that carries a duration for a departure nobody had asked about is
    not complete."""
    run = run_of(value(converged=False))

    assert not run.complete
    assert len(run.unsettled) == 1
    assert "指定時刻に収束しなかった経路 1 件" in text_of(run)


def test_an_unconverged_leg_is_not_counted_as_a_failure() -> None:
    """Two different states. The failure has no number at all; this one has a
    real number that answers a different question."""
    out = text_of(run_of(value(converged=False)))

    assert "測れなかった経路" not in out


def test_the_clock_comes_from_the_schedule_not_from_its_id() -> None:
    """`{id: arr_0930, type: depart, time: "10:30"}` is valid configuration.
    Printing the id alone invited the reader to believe the id."""
    out = text_of(run_of(value(schedule_id="arr_0930", anchor="depart", time="10:30")))

    assert "10:30 発" in out


def test_a_schedule_declaring_two_daytypes_names_the_one_not_measured() -> None:
    """Only the first is measured. A schedule with two daytypes must not look
    like a schedule with one."""
    out = text_of(run_of(value(daytype="weekend", declared=("weekend", "holiday"))))

    assert "祝日 は測っていません" in out


# --- ordering ---------------------------------------------------------------


def test_groups_and_destinations_keep_the_order_they_were_measured_in() -> None:
    out = text_of(
        run_of(
            value(group="commute", destination="Central Station"),
            value(group="group_a", destination="Site A"),
            value(group="group_a", destination="Site B"),
        )
    )

    assert out.index("commute") < out.index("group_a")
    assert out.index("Site A") < out.index("Site B")


@pytest.mark.parametrize("duration_s", [0, 60, 7200])
def test_every_leg_prints_a_duration(duration_s: int) -> None:
    out = text_of(run_of(value(duration_s=duration_s)))

    assert f"{duration_s / 60.0:5.1f} 分" in out
