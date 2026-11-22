"""Printing adopted values without letting them blur into estimates.

An adopted value is bought from the Google Routes API and is a different kind
of number from a free, locally computed one. Presenting a free figure as an
adopted one is forbidden. The risk is not in the arithmetic -- it is in a
layout where the two appear as neighbouring rows of one table and the reader
carries the heading of the first across to the second.

So the adopted block is separate, labelled, and says on its own face where its
numbers came from. Lines are returned rather than printed, because a format is
worth a test and `print` is not.
"""

from __future__ import annotations

from collections.abc import Sequence

from landeval.adopted import AdoptedMeasurement, AdoptedValue
from landeval.config import DayType, Direction, ScheduleType

DAYTYPE_LABELS: dict[DayType, str] = {
    "friday": "金曜",
    "weekday": "平日",
    "weekend": "週末",
    "holiday": "祝日",
}

DIRECTION_LABELS: dict[Direction, str] = {
    "outbound": "往路",
    "inbound": "復路",
}

# Which end of the trip the clock is pinned to. Printed beside the time,
# because `11:00 発` and `11:00 着` are different questions and the schedule
# id is free text that can say neither.
ANCHOR_LABELS: dict[ScheduleType, str] = {
    "arrive": "着",
    "depart": "発",
}


def format_adopted(run: AdoptedMeasurement) -> list[str]:
    """The adopted-value block, as printable lines."""
    lines = [
        "",
        f"  実測（採用値）  {run.billable_calls} 呼び出し",
        "    ※ ここから下は Google Routes API の実測値です。上の推定値とは別の量で、",
        "      誤差幅もパーセンタイルも付きません",
    ]
    lines.extend(_pass_note(run))

    for group in _groups(run.values):
        lines.append("")
        lines.append(f"    【{group}】")
        for destination in _destinations(run.values, group):
            lines.append(f"      {destination}")
            for value in _legs(run.values, group, destination):
                lines.append(f"        {_leg_line(value)}")

    lines.extend(_failures(run))
    return lines


def _pass_note(run: AdoptedMeasurement) -> list[str]:
    """Which toll question was asked, said out loud.

    The same route is priced differently without a pass, with one, and on a
    holiday. An amount printed without the pass and
    the day it was measured under cannot be read.
    """
    if not run.shape.tolls:
        return ["    ※ 通行料金は聞いていません（欠損であって 0 円ではありません）"]
    if run.shape.toll_passes:
        declared = "・".join(run.shape.toll_passes)
        return [f"    ※ 通行料金は {declared} を宣言した額です。曜日区分で変わります"]
    return ["    ※ 通行料金はパスを宣言しない額（現金相当）です"]


def _groups(values: Sequence[AdoptedValue]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value.group not in seen:
            seen.append(value.group)
    return seen


def _destinations(values: Sequence[AdoptedValue], group: str) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value.group == group and value.destination not in seen:
            seen.append(value.destination)
    return seen


def _legs(values: Sequence[AdoptedValue], group: str, destination: str) -> list[AdoptedValue]:
    return [v for v in values if v.group == group and v.destination == destination]


def _leg_line(value: AdoptedValue) -> str:
    direction = DIRECTION_LABELS[value.direction]
    daytype = DAYTYPE_LABELS[value.daytype]
    # The clock comes from the schedule, never from its id. An id reading
    # `arrive_0930` on a schedule declaring `{type: depart, time: "10:30"}`
    # is valid configuration, and printing the id alone invited the reader
    # to believe the id.
    anchor = ANCHOR_LABELS[value.anchor]
    line = (
        f"{value.schedule_id:<22} {direction} {daytype} {value.time} {anchor}"
        f"   {value.duration_min:5.1f} 分   {_toll(value)}"
    )
    if not value.converged:
        # The solver returned its best attempt at a *different* departure, so
        # this duration answers a question nobody asked. It is still printed --
        # dropping it would be worse -- but the run does not call itself
        # complete while one is present, and the line says the number is not
        # an answer rather than merely noting the solve was imperfect.
        line += "   ※ 到着時刻に収束せず。この分数は指定時刻の答えではありません"
    if len(value.declared_daytypes) > 1:
        others = "・".join(DAYTYPE_LABELS[d] for d in value.declared_daytypes[1:])
        line += f"   （{others} は測っていません）"
    return line


def _toll(value: AdoptedValue) -> str:
    if value.toll_yen is None:
        # Never 0. The response for "free" and the response for "not asked" are
        # the same bytes, so a missing amount is printed as missing.
        return "通行料金 不明"
    return f"通行料金 {value.toll_yen:,} 円"


# There is deliberately no per-destination total here. Summing every leg for a
# destination and printing the result as 「往復の通行料金」 would be wrong: a
# destination may declare several outbound and several inbound schedules,
# which are alternative times for one journey and not that many journeys, so
# the sum of all the alternatives is not a "there and back" amount. With
# exactly two legs the sum and the round trip coincide, which is why a
# two-leg case cannot tell them apart.
#
# A round trip is a pairing of one outbound schedule with one inbound one, and
# nothing in the configuration says which pairs with which; the aggregation
# rules are about destinations, not about times. Until something declares the
# pairing, the honest output is the per-leg amounts, which the reader can add
# for the pair they actually intend.


def _failures(run: AdoptedMeasurement) -> list[str]:
    """Legs that could not be measured, printed rather than omitted.

    A block listing two destinations when three were asked for reads as a
    complete answer, and nothing in it says otherwise.
    """
    lines: list[str] = []
    if run.failures:
        lines.append("")
        lines.append(
            f"    測れなかった経路 {len(run.failures)} 件（欠測であって 0 ではありません）"
        )
        lines.extend(
            f"      {failure.group} {failure.destination} {failure.schedule_id}: {failure.reason}"
            for failure in run.failures
        )
    if run.unsettled:
        # Counted separately from the failures above. These legs have a number
        # and it is a real one; what they do not have is an answer to the
        # schedule that was asked, and folding them into either the answers or
        # the failures would lose exactly that distinction.
        lines.append("")
        lines.append(
            f"    指定時刻に収束しなかった経路 {len(run.unsettled)} 件"
            "（分数は出ていますが、その時刻の答えではありません）"
        )
        lines.extend(
            f"      {value.group} {value.destination} {value.schedule_id}"
            for value in run.unsettled
        )
    return lines
