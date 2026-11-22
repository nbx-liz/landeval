"""Adopted values for one address.

The estimate side is a separate, free path. This is the
other side, and the two must not blur: an adopted value is a real API answer
for a real trip, and nothing here may present an estimate as one.

Every test runs against a mock transport. A loop that measures the wrong leg
costs exactly what a correct one does, and this loop bills several calls per
address.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from landeval.adopted import AdoptedError, legs, measure_address, request_shape
from landeval.basedate import JST
from landeval.config import Config
from landeval.routes import RoutesClient
from landeval.tolls import RequestShape

ORIGIN = (35.21, 140.05)
TODAY = datetime(2030, 6, 7, 12, 0, tzinfo=JST)

RAW: dict[str, object] = {
    "origin_grid": {
        "mesh_size_m": 500,
        "bbox_source": "freeflow_minutes",
        "freeflow_minutes": 100,
    },
    "thresholds": {
        "commute": {
            "target_schedule": "arr_0930",
            "good_max_min": 45,
            "hard_max_min": 70,
            "exclude_min": 100,
        }
    },
    "vehicle": {"toll_passes": ["JP_ETC"], "emission_type": "GASOLINE"},
    "destinations": {
        "commute": {
            "aggregate": "single",
            "used_for_screening": True,
            "points": [{"name": "Central Station", "lat": 35.680000, "lon": 139.760000}],
            "schedules": [
                {
                    "id": "arr_0930",
                    "type": "arrive",
                    "time": "09:30",
                    "daytype": "friday",
                    "direction": "outbound",
                },
                {
                    "id": "ret_2030",
                    "type": "depart",
                    "time": "20:30",
                    "daytype": "friday",
                    "direction": "inbound",
                },
            ],
        },
        "group_a": {
            "aggregate": ["each", "mean"],
            "used_for_screening": False,
            "points": [
                {"name": "Site A", "lat": 43.100000, "lon": 141.300000},
                {"name": "Site B", "lat": 44.200000, "lon": 142.300000},
            ],
            "schedules": [
                {
                    "id": "group_a_depart_0630",
                    "type": "depart",
                    "time": "06:30",
                    "daytype": "weekend",
                    "direction": "outbound",
                }
            ],
        },
    },
}


def config_of(**overrides: object) -> Config:
    raw = {**RAW, **overrides}
    return Config.model_validate(raw)


def toll_handler(yen: int | None, duration_s: int = 1800) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        route: dict[str, object] = {"duration": f"{duration_s}s", "distanceMeters": 27000}
        if yen is None:
            route["travelAdvisory"] = {}
        else:
            route["travelAdvisory"] = {
                "tollInfo": {"estimatedPrice": [{"currencyCode": "JPY", "units": str(yen)}]}
            }
        return httpx.Response(200, json={"routes": [route]})

    return httpx.MockTransport(handler)


def client_for(transport: httpx.MockTransport, tmp_path: Path) -> RoutesClient:
    return RoutesClient(api_key="unused", transport=transport, cache_dir=tmp_path)


# --- the shape the configuration asks for ----------------------------------


def test_the_shape_comes_from_the_configured_vehicle() -> None:
    shape = request_shape(config_of())

    assert shape == RequestShape(tolls=True, toll_passes=("JP_ETC",), emission_type="GASOLINE")


def test_a_configuration_with_no_vehicle_refuses_rather_than_defaulting() -> None:
    """A pass changes the amount the API returns for the same route. A default
    would print the cash rate for a driver who has an ETC unit -- an answer to a
    question nobody asked."""
    raw = {key: value for key, value in RAW.items() if key != "vehicle"}

    with pytest.raises(AdoptedError, match="vehicle"):
        request_shape(Config.model_validate(raw))


def test_an_unmeasured_pass_in_the_configuration_is_refused() -> None:
    with pytest.raises(AdoptedError, match="cannot be sent"):
        request_shape(config_of(vehicle={"toll_passes": ["JP_ETC9"], "emission_type": "GASOLINE"}))


def test_a_duplicate_pass_in_the_configuration_is_refused() -> None:
    with pytest.raises(AdoptedError, match="cannot be sent"):
        request_shape(
            config_of(vehicle={"toll_passes": ["JP_ETC2", "JP_ETC2"], "emission_type": "GASOLINE"})
        )


@pytest.mark.parametrize(
    "vehicle",
    [
        {},
        {"emission_type": "GASOLINE"},
        {"toll_passes": ["JP_ETC2"]},
    ],
)
def test_a_partial_vehicle_block_is_refused_rather_than_completed(
    vehicle: dict[str, object],
) -> None:
    """Both fields are required with no default, for the reason `direction` on a
    schedule is: a default is what a configuration silently inherits when its
    author never considered the question, and inheriting "no pass" means being
    quoted the cash rate. A driver with no unit writes `toll_passes: []`."""
    with pytest.raises(ValidationError):
        config_of(vehicle=vehicle)


def test_a_driver_with_no_pass_says_so_explicitly() -> None:
    """The empty list is accepted -- it is a declaration, not an omission."""
    shape = request_shape(config_of(vehicle={"toll_passes": [], "emission_type": "GASOLINE"}))

    assert shape.toll_passes == ()
    assert shape.tolls


# --- what one address costs -------------------------------------------------


def test_every_destination_and_schedule_is_measured() -> None:
    """A measurement run measures all destinations, not a chosen few."""
    plan = legs(config_of())

    assert len(plan) == 4  # commute 1x2 + group_a 2x1
    assert {(group, point.name, schedule.id) for group, point, schedule in plan} == {
        ("commute", "Central Station", "arr_0930"),
        ("commute", "Central Station", "ret_2030"),
        ("group_a", "Site A", "group_a_depart_0630"),
        ("group_a", "Site B", "group_a_depart_0630"),
    }


def test_nothing_is_billed_until_measure_is_called(tmp_path: Path) -> None:
    with client_for(toll_handler(1740), tmp_path) as client:
        legs(config_of())

        assert client.billable_calls == 0


# --- the measurement --------------------------------------------------------


def test_each_leg_reports_its_toll_and_the_day_it_was_measured_on(tmp_path: Path) -> None:
    with client_for(toll_handler(1740), tmp_path) as client:
        run = measure_address(client, config_of(), ORIGIN, today=TODAY)

    assert len(run.values) == 4
    assert {value.toll_yen for value in run.values} == {1740}

    by_id = {value.schedule_id: value for value in run.values}
    assert by_id["arr_0930"].daytype == "friday"
    assert by_id["group_a_depart_0630"].daytype == "weekend"


def test_a_value_reports_its_trip_in_minutes_and_when_it_lands(tmp_path: Path) -> None:
    with client_for(toll_handler(1740, duration_s=3600), tmp_path) as client:
        run = measure_address(client, config_of(), ORIGIN, today=TODAY)

    value = next(v for v in run.values if v.schedule_id == "group_a_depart_0630")

    assert value.duration_min == pytest.approx(60.0)
    assert (value.arrival_time - value.departure_time).total_seconds() == 3600


def test_a_free_leg_reports_zero_rather_than_nothing(tmp_path: Path) -> None:
    with client_for(toll_handler(None), tmp_path) as client:
        run = measure_address(client, config_of(), ORIGIN, today=TODAY)

    assert {value.toll_yen for value in run.values} == {0}


def test_the_direction_of_each_leg_comes_from_the_schedule(tmp_path: Path) -> None:
    """The inbound schedule routes from the destination to the address,
    and no caller here decides that locally."""
    seen: list[tuple[float, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        latlng = body["origin"]["location"]["latLng"]
        seen.append((latlng["latitude"], latlng["longitude"]))
        return httpx.Response(
            200,
            json={"routes": [{"duration": "1800s", "distanceMeters": 27000, "travelAdvisory": {}}]},
        )

    with client_for(httpx.MockTransport(handler), tmp_path) as client:
        measure_address(client, config_of(), ORIGIN, today=TODAY)

    # The inbound commute leg starts at Marunouchi, not at the address.
    assert (35.680000, 139.760000) in seen
    assert ORIGIN in seen


def test_the_shape_reaches_every_request(tmp_path: Path) -> None:
    masks: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        masks.append(request.headers["X-Goog-FieldMask"])
        return httpx.Response(
            200,
            json={"routes": [{"duration": "1800s", "distanceMeters": 27000, "travelAdvisory": {}}]},
        )

    with client_for(httpx.MockTransport(handler), tmp_path) as client:
        measure_address(client, config_of(), ORIGIN, today=TODAY)

    assert masks, "no request was made"
    assert all("tollInfo" in mask for mask in masks)


def test_a_leg_that_cannot_be_routed_is_kept_not_dropped(tmp_path: Path) -> None:
    """A dropped leg is indistinguishable from one that was never asked for, and
    three destinations printed where four were requested reads as complete."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"routes": []})
        return httpx.Response(
            200,
            json={"routes": [{"duration": "1800s", "distanceMeters": 27000, "travelAdvisory": {}}]},
        )

    with client_for(httpx.MockTransport(handler), tmp_path) as client:
        run = measure_address(client, config_of(), ORIGIN, today=TODAY)

    assert run.failures, "the unroutable leg vanished"
    assert not run.complete
    assert len(run.values) + len(run.failures) == 4


def test_a_complete_run_says_so(tmp_path: Path) -> None:
    with client_for(toll_handler(1740), tmp_path) as client:
        run = measure_address(client, config_of(), ORIGIN, today=TODAY)

    assert run.complete


def test_the_run_reports_what_it_billed(tmp_path: Path) -> None:
    with client_for(toll_handler(1740), tmp_path) as client:
        run = measure_address(client, config_of(), ORIGIN, today=TODAY)

    assert run.billable_calls >= 4  # one per leg, plus the arrival back-solve


def test_the_second_address_is_not_billed_for_the_first(tmp_path: Path) -> None:
    """One client serves every address in a run and counts for its whole life.
    Reporting its running total would tell the second address it cost what the
    first two did together."""
    other = (35.4657, 139.6222)

    with client_for(toll_handler(1740), tmp_path) as client:
        first = measure_address(client, config_of(), ORIGIN, today=TODAY)
        second = measure_address(client, config_of(), other, today=TODAY)

    assert second.billable_calls == first.billable_calls
    assert client.billable_calls == first.billable_calls + second.billable_calls


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


@pytest.mark.parametrize("echo", ["duration", "toll"])
@pytest.mark.parametrize("key", RELAY_KEYS)
def test_a_failure_reason_never_carries_the_key(tmp_path: Path, key: str, echo: str) -> None:
    """The reason is the routes error's text, relayed. It can only be as clean
    as that text, so this pins the relay rather than trusting it -- in every
    escaped form too, since ``repr`` and ``json.dumps`` quote the echo."""
    text = f"{key}\"'{key}!"

    def handler(request: httpx.Request) -> httpx.Response:
        route: dict[str, object] = {"duration": "1800s", "distanceMeters": 27000}
        if echo == "duration":
            route["duration"] = text
            route["travelAdvisory"] = {}
        else:
            route["travelAdvisory"] = {"tollInfo": {"note": text}}
        return httpx.Response(200, json={"routes": [route]})

    client = RoutesClient(api_key=key, transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with client:
        run = measure_address(client, config_of(), ORIGIN, today=TODAY)

    assert run.failures, "every leg was meant to fail"
    assert not run.values
    assert not any(carries_the_key(failure.reason, key) for failure in run.failures)
