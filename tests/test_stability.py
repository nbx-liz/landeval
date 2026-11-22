"""Tests for the repeat-sampling premise check.

The acceptance criteria, one test each. What they are guarding is not
arithmetic but the two ways this check could report a false clean: a stratum
that silently returns fewer rows than asked for, and a cache lookup that turns a
miss into a second billable call and then compares an answer with itself.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from landeval.config import Schedule
from landeval.routes import RoutesClient
from landeval.stability import (
    IDENTICAL,
    MOVES,
    WITHIN_NOISE,
    CacheOnlyTransport,
    StabilityError,
    Summary,
    check_lead_time,
    compare_triples,
    select_triples,
)
from landeval.stability import summarise as _summarise
from landeval.stability import verdict as _verdict

# Neutral tolerances for these tests; the tolerances a real check uses are the
# caller's choice, derived from its own estimate.
NOISE_TOLERANCE_S = 75
SYSTEMATIC_TOLERANCE_S = 20

DESTINATION = (35.680000, 139.760000)

BASE_DEPARTURE = "2026-07-17T07:30:00+09:00"

Handler = Callable[[httpx.Request], httpx.Response]


def verdict(deltas: list[float]) -> str:
    """The verdict under this file's tolerances; the library holds none of its own."""
    return _verdict(
        deltas,
        noise_tolerance_s=NOISE_TOLERANCE_S,
        systematic_tolerance_s=SYSTEMATIC_TOLERANCE_S,
    )


def summarise(deltas: list[float]) -> Summary:
    return _summarise(
        deltas,
        noise_tolerance_s=NOISE_TOLERANCE_S,
        systematic_tolerance_s=SYSTEMATIC_TOLERANCE_S,
    )


def population(
    schedule_ids: tuple[str, ...] = ("dep_0730", "ret_2030"), rows: int = 10
) -> pd.DataFrame:
    """A measurement table shaped like the real ones, with distinct durations."""
    records = []
    for schedule_id in schedule_ids:
        for index in range(rows):
            records.append(
                {
                    "schedule_id": schedule_id,
                    "mesh_id": f"{schedule_id}-{index:03d}",
                    "lat": 35.5 + index / 1000,
                    "lon": 139.5 + index / 1000,
                    "departure_time": BASE_DEPARTURE,
                    "measured_min": 30.0 + index,
                    "measured_distance_m": 10_000 + index,
                }
            )
    return pd.DataFrame.from_records(records)


def test_selection_returns_three_rows_per_schedule_without_randomness() -> None:
    frame = population()

    first = select_triples(frame)
    second = select_triples(frame.sample(frac=1.0, random_state=7))

    assert len(first) == 6
    assert sorted(first["quantile"].unique()) == [0.10, 0.50, 0.90]
    # Row order in must not decide row choice out: the parquet files are not
    # sorted, and a check that moves with input order cannot be re-run.
    assert first["mesh_id"].tolist() == second["mesh_id"].tolist()


def test_selection_spreads_across_the_duration_range() -> None:
    """A stratum that collapses to three neighbouring rows would test one trip
    length three times and call it coverage."""
    selected = select_triples(population(schedule_ids=("dep_0730",), rows=10))

    assert selected["measured_min"].tolist() == [30.0, 34.0, 38.0]


def test_a_schedule_with_fewer_rows_than_strata_raises() -> None:
    frame = population(schedule_ids=("dep_0730",), rows=2)

    with pytest.raises(StabilityError, match="fewer than the"):
        select_triples(frame)


def test_a_missing_column_raises_rather_than_selecting_on_what_is_left() -> None:
    frame = population().drop(columns=["measured_distance_m"])

    with pytest.raises(StabilityError, match="measured_distance_m"):
        select_triples(frame)


def test_an_unmeasured_row_raises_rather_than_being_dropped() -> None:
    """The caller filters unmeasured rows visibly. Filtering here would let a
    schedule that is entirely unmeasured disappear from the run silently."""
    frame = population()
    frame.loc[0, "measured_min"] = None

    with pytest.raises(StabilityError, match="measured_min"):
        select_triples(frame)


def test_lead_time_passes_when_every_departure_is_far_enough_ahead() -> None:
    check_lead_time([BASE_DEPARTURE] * 3, date(2026, 7, 11), min_lead_days=2)


def test_one_departure_inside_the_lead_time_stops_the_run() -> None:
    """One row is enough. Measuring the rest and reporting a verdict would mean
    part of the evidence was gathered against a rule the project states."""
    with pytest.raises(StabilityError, match="lead-time rule"):
        check_lead_time([BASE_DEPARTURE], date(2026, 7, 16), min_lead_days=2)


def test_an_unparseable_departure_raises() -> None:
    with pytest.raises(StabilityError, match="not a timestamp"):
        check_lead_time(["yesterday"], date(2026, 7, 11), min_lead_days=2)


def test_all_zero_deltas_read_as_the_premise_holding() -> None:
    assert verdict([0, 0, 0]) == IDENTICAL


def test_a_delta_at_the_noise_tolerance_is_still_noise() -> None:
    assert verdict([NOISE_TOLERANCE_S, -NOISE_TOLERANCE_S]) == WITHIN_NOISE


def test_a_delta_one_second_past_the_tolerance_moves() -> None:
    assert verdict([NOISE_TOLERANCE_S + 1, -NOISE_TOLERANCE_S]) == MOVES


def test_a_systematic_shift_moves_even_when_every_delta_is_small() -> None:
    """Every value inside the noise band but all of one sign biases every
    printed number in one direction, which the maximum alone cannot see."""
    assert verdict([SYSTEMATIC_TOLERANCE_S + 1] * 4) == MOVES


def test_a_shift_exactly_at_the_systematic_tolerance_is_still_noise() -> None:
    assert verdict([SYSTEMATIC_TOLERANCE_S] * 4) == WITHIN_NOISE


def test_the_library_holds_no_tolerance_of_its_own() -> None:
    """The tolerances come from the caller's own estimate, so the caller must pass them."""
    with pytest.raises(TypeError):
        _verdict([0, 1])  # type: ignore[call-arg]


def test_the_tolerances_passed_in_decide_the_verdict() -> None:
    assert _verdict([5, -5], noise_tolerance_s=5, systematic_tolerance_s=1) == WITHIN_NOISE
    assert _verdict([6, -6], noise_tolerance_s=5, systematic_tolerance_s=1) == MOVES
    assert _verdict([2, 2], noise_tolerance_s=5, systematic_tolerance_s=1) == MOVES


def test_an_empty_comparison_has_no_verdict() -> None:
    with pytest.raises(StabilityError, match="no deltas"):
        verdict([])


def test_the_summary_reports_the_numbers_the_verdict_was_read_from() -> None:
    summary = summarise([0, 30, -30, 90])

    assert summary.verdict == MOVES
    assert summary.count == 4
    assert summary.identical == 1
    assert summary.max_abs_s == 90
    assert summary.mean_s == pytest.approx(22.5)


def answering(duration_s: int) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"routes": [{"duration": f"{duration_s}s", "distanceMeters": 12_345}]},
        )

    return handler


def one_row(duration_s: int) -> pd.DataFrame:
    frame = population(schedule_ids=("dep_0730",), rows=3)
    frame["measured_min"] = duration_s / 60.0
    frame["measured_distance_m"] = 12_345
    return select_triples(frame).head(1)


SCHEDULES = {
    "dep_0730": Schedule(
        id="dep_0730", type="depart", time="07:30", daytype=["friday"], direction="outbound"
    )
}


def seeded_cache(tmp_path: Path, row: pd.Series, duration_s: int) -> Path:
    """Write the stored answer through the same client that will read it back."""
    cache_dir = tmp_path / "cache"
    with RoutesClient(
        "k", transport=httpx.MockTransport(answering(duration_s)), cache_dir=cache_dir
    ) as seeder:
        origin, target = SCHEDULES[row["schedule_id"]].endpoints(
            (row["lat"], row["lon"]), DESTINATION
        )
        seeder.route(origin, target, datetime.fromisoformat(str(row["departure_time"])))
    return cache_dir


def test_the_refetch_writes_nothing_into_the_stored_cache(tmp_path: Path) -> None:
    """Same key, different answer: writing it to the stored cache would delete
    the first-day value this check exists to compare against."""
    selected = one_row(600)
    row = selected.iloc[0]
    cache_dir = seeded_cache(tmp_path, row, 600)
    before = {path: path.read_bytes() for path in sorted(cache_dir.rglob("*.json"))}

    with (
        RoutesClient("k", transport=CacheOnlyTransport(), cache_dir=cache_dir) as cached,
        RoutesClient(
            "k", transport=httpx.MockTransport(answering(660)), cache_dir=tmp_path / "refetch"
        ) as fresh,
    ):
        compared = compare_triples(
            selected, schedules=SCHEDULES, destination=DESTINATION, cached=cached, fresh=fresh
        )

    assert {path: path.read_bytes() for path in sorted(cache_dir.rglob("*.json"))} == before
    assert compared["delta_s"].tolist() == [60]
    assert compared["cached_duration_s"].tolist() == [600]
    assert compared["fresh_duration_s"].tolist() == [660]


def test_a_cache_miss_raises_instead_of_becoming_a_second_billable_call(tmp_path: Path) -> None:
    """`RoutesClient.route` falls through to the network on a miss. Reading the
    stored side through a transport that cannot answer is what keeps a missing
    file from being paid for and then compared against itself."""
    selected = one_row(600)

    with (
        RoutesClient("k", transport=CacheOnlyTransport(), cache_dir=tmp_path / "empty") as cached,
        RoutesClient(
            "k", transport=httpx.MockTransport(answering(660)), cache_dir=tmp_path / "refetch"
        ) as fresh,
    ):
        with pytest.raises(StabilityError, match="not in the stored cache"):
            compare_triples(
                selected, schedules=SCHEDULES, destination=DESTINATION, cached=cached, fresh=fresh
            )

        assert fresh.billable_calls == 0


def test_a_stored_answer_that_disagrees_with_the_table_raises(tmp_path: Path) -> None:
    """The parquet and the cache are two copies of one measurement. A silent
    disagreement means the comparison is against a number nothing downstream
    uses."""
    selected = one_row(600)
    cache_dir = seeded_cache(tmp_path, selected.iloc[0], 900)

    with (
        RoutesClient("k", transport=CacheOnlyTransport(), cache_dir=cache_dir) as cached,
        RoutesClient(
            "k", transport=httpx.MockTransport(answering(660)), cache_dir=tmp_path / "refetch"
        ) as fresh,
        pytest.raises(StabilityError, match="disagree"),
    ):
        compare_triples(
            selected, schedules=SCHEDULES, destination=DESTINATION, cached=cached, fresh=fresh
        )


def test_a_departure_that_does_not_round_trip_raises(tmp_path: Path) -> None:
    """The cache key is built from `departure.isoformat()`. A stored spelling
    that re-serialises differently would miss the file it was written to, and a
    miss is a charge."""
    selected = one_row(600)
    selected = selected.assign(departure_time="2026-07-17T07:30:00+0900")

    with (
        RoutesClient("k", transport=CacheOnlyTransport(), cache_dir=tmp_path / "cache") as cached,
        RoutesClient(
            "k", transport=httpx.MockTransport(answering(660)), cache_dir=tmp_path / "refetch"
        ) as fresh,
        pytest.raises(StabilityError, match="does not round-trip"),
    ):
        compare_triples(
            selected, schedules=SCHEDULES, destination=DESTINATION, cached=cached, fresh=fresh
        )


def test_an_unknown_schedule_raises(tmp_path: Path) -> None:
    selected = one_row(600).assign(schedule_id="depart_2300")

    with (
        RoutesClient("k", transport=CacheOnlyTransport(), cache_dir=tmp_path / "cache") as cached,
        RoutesClient(
            "k", transport=httpx.MockTransport(answering(660)), cache_dir=tmp_path / "refetch"
        ) as fresh,
        pytest.raises(StabilityError, match="depart_2300"),
    ):
        compare_triples(
            selected, schedules=SCHEDULES, destination=DESTINATION, cached=cached, fresh=fresh
        )


def test_custom_quantiles_that_land_on_one_row_raise() -> None:
    """The default three never collide, but the argument is public: two nearby
    quantiles over a short schedule would otherwise return one row twice and
    call it two strata."""
    with pytest.raises(StabilityError, match="collapse"):
        select_triples(population(schedule_ids=("dep_0730",), rows=3), quantiles=(0.1, 0.2))


def test_a_refetch_answered_from_a_cache_raises(tmp_path: Path) -> None:
    """`RoutesClient.route` reads its own cache before it posts, so a second run
    against a directory the first run filled would replay that run's answers,
    recompute the same verdict, and stamp it with today's date -- 0 calls, and
    nothing measured. It is the same false-clean the stored side is guarded
    against."""
    selected = one_row(600)
    cache_dir = seeded_cache(tmp_path, selected.iloc[0], 600)
    refetch = tmp_path / "refetch"

    def run() -> pd.DataFrame:
        with (
            RoutesClient("k", transport=CacheOnlyTransport(), cache_dir=cache_dir) as cached,
            RoutesClient(
                "k", transport=httpx.MockTransport(answering(660)), cache_dir=refetch
            ) as fresh,
        ):
            return compare_triples(
                selected, schedules=SCHEDULES, destination=DESTINATION, cached=cached, fresh=fresh
            )

    assert run()["delta_s"].tolist() == [60]

    with pytest.raises(StabilityError, match="answered from a cache"):
        run()


def test_a_stored_distance_that_disagrees_with_the_table_raises(tmp_path: Path) -> None:
    """Distance is cross-checked like duration: a disagreeing stored distance
    would otherwise be reported as a route the API changed."""
    selected = one_row(600).assign(measured_distance_m=999)
    cache_dir = seeded_cache(tmp_path, selected.iloc[0], 600)

    with (
        RoutesClient("k", transport=CacheOnlyTransport(), cache_dir=cache_dir) as cached,
        RoutesClient(
            "k", transport=httpx.MockTransport(answering(660)), cache_dir=tmp_path / "refetch"
        ) as fresh,
        pytest.raises(StabilityError, match="measured_distance_m"),
    ):
        compare_triples(
            selected, schedules=SCHEDULES, destination=DESTINATION, cached=cached, fresh=fresh
        )


def test_a_non_finite_delta_has_no_verdict() -> None:
    """Every comparison in the rule is False against a NaN, so it would fall
    through to the reassuring branch for a difference that was never computed."""
    with pytest.raises(StabilityError, match="not finite"):
        verdict([float("nan"), 0])


def test_an_empty_population_raises_rather_than_selecting_nothing() -> None:
    with pytest.raises(StabilityError, match="nothing to re-ask"):
        select_triples(population().iloc[0:0])


def test_a_row_without_a_schedule_id_raises_rather_than_leaving_quietly() -> None:
    """`groupby` drops null keys by default, so such rows would go the way the
    unmeasured-row check exists to prevent, one column over."""
    frame = population(schedule_ids=("dep_0730",), rows=6)
    frame.loc[0, "schedule_id"] = None

    with pytest.raises(StabilityError, match="no schedule_id"):
        select_triples(frame)


def test_a_tie_on_duration_does_not_let_input_order_decide() -> None:
    """The `mesh_id` tie-break is load-bearing whenever selected rows sit inside
    a `measured_min` tie group, and the population above cannot detect its
    removal because its durations are all distinct. Without this case, dropping
    the tie-break would leave the suite green while changing which rows are
    selected."""
    frame = population(schedule_ids=("dep_0730",), rows=6)
    frame["measured_min"] = [30.0, 30.0, 30.0, 40.0, 40.0, 40.0]

    forwards = select_triples(frame)
    backwards = select_triples(frame.iloc[::-1])

    assert forwards["mesh_id"].tolist() == backwards["mesh_id"].tolist()


def _carries_a_timezone(node: ast.Call) -> bool:
    """`now(tz=None)` supplies an argument and is still naive, so counting
    arguments is not enough."""
    supplied = [*node.args, *(keyword.value for keyword in node.keywords)]
    if not supplied:
        return False
    return not any(isinstance(value, ast.Constant) and value.value is None for value in supplied)
