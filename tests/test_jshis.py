"""Reading the earthquake layer at lookup time.

Two J-SHIS endpoints answer one point: the surface-ground one gives how much
the ground amplifies a shake, and the hazard one gives the probability of a
strong one. Both are 250 m mesh, so this layer says something about the mesh a
point sits in -- not about the point, and not about a parcel.

The failure modes are the ones that matter here. A layer that cannot be read
must not read as calm ground, and "the service was down" must not print as
"outside the zone". Every case here therefore checks what happens when the
answer does not arrive, not only when it does.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from landeval.jshis import (
    DEFAULT_GROUND_EDITION,
    DEFAULT_HAZARD_EDITION,
    JshisClient,
    JshisError,
)

# A municipal office: a point on land, so it lies in some 250 m mesh.
FUKUOKA = (33.59018, 130.40170)

Handler = Callable[[httpx.Request], httpx.Response]

GROUND = {
    "features": [
        {
            "properties": {
                "meshcode": "5030330233",
                "AVS": "312.4",
                "ARV": "1.27",
                "JNAME": "砂礫質台地",
                "JCODE": "10",
            }
        }
    ]
}
HAZARD = {"features": [{"properties": {"meshcode": "5030330233", "T30_I55_PS": "0.427915"}}]}


def answering(ground: object = GROUND, hazard: object = HAZARD) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = hazard if "/pshm/" in request.url.path else ground
        return httpx.Response(200, json=body)

    return handler


def make_client(handler: Handler, tmp_path: Path) -> JshisClient:
    return JshisClient(transport=httpx.MockTransport(handler), cache_dir=tmp_path, delay_s=0.0)


# --- what it reads ----------------------------------------------------------


def test_it_reads_the_four_values_the_response_carries(tmp_path: Path) -> None:
    """An invented answer in the shape the API returns. If the
    parsing drifts, these are the numbers it drifts away from."""
    with make_client(answering(), tmp_path) as client:
        reading = client.reading_at(*FUKUOKA)

    assert reading.meshcode == "5030330233"
    assert reading.avs30_ms == pytest.approx(312.4)
    assert reading.amplification == pytest.approx(1.27)
    assert reading.landform == "砂礫質台地"
    assert reading.probability_6lower_30y == pytest.approx(0.427915)


def test_the_editions_it_asked_for_are_the_ones_recorded(tmp_path: Path) -> None:
    """The edition is part of the answer: a 2020 ground model and a 2024 hazard
    model are different measurements, and a reading that does not say which it
    came from cannot be compared with a later one."""
    paths: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return answering()(request)

    with make_client(record, tmp_path) as client:
        reading = client.reading_at(*FUKUOKA)

    assert f"/sstrct/{DEFAULT_GROUND_EDITION}/" in paths[0]
    assert f"/pshm/{DEFAULT_HAZARD_EDITION}/" in paths[1]
    assert reading.ground_edition == DEFAULT_GROUND_EDITION
    assert reading.hazard_edition == DEFAULT_HAZARD_EDITION


def test_the_coordinate_is_sent_as_longitude_then_latitude(tmp_path: Path) -> None:
    """`position=lon,lat`, which is the opposite order from every other call in
    this repository. Transposed, the query asks about some other point, and
    the API may answer about that mesh rather than failing."""
    sent: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.params["position"])
        return answering()(request)

    with make_client(record, tmp_path) as client:
        client.reading_at(*FUKUOKA)

    assert sent[0] == f"{FUKUOKA[1]},{FUKUOKA[0]}"


# --- what it refuses to do --------------------------------------------------


@pytest.mark.parametrize("status", [403, 404, 500, 503])
def test_a_failed_request_is_an_error_not_a_reading(tmp_path: Path, status: int) -> None:
    """Not a null reading, and certainly not a calm one. A layer that reports
    "no risk found" because it could not look is the false-clean shape."""

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    with make_client(refuse, tmp_path) as client, pytest.raises(JshisError, match=str(status)):
        client.reading_at(*FUKUOKA)


def test_an_empty_feature_list_is_an_error(tmp_path: Path) -> None:
    """Every point on land in Japan is in some 250 m mesh. An empty answer is
    the API declining to say, and treating it as "no data for this ground"
    would put the two apart forever."""
    with (
        make_client(answering(ground={"features": []}), tmp_path) as client,
        pytest.raises(JshisError, match="no mesh"),
    ):
        client.reading_at(*FUKUOKA)


def test_a_non_numeric_value_is_an_error(tmp_path: Path) -> None:
    """The API returns its numbers as strings, so a change of format arrives as
    a string that does not parse rather than as a type error."""
    broken = {"features": [{"properties": {"meshcode": "1", "AVS": "-", "ARV": "1.6"}}]}

    with (
        make_client(answering(ground=broken), tmp_path) as client,
        pytest.raises(JshisError, match="AVS"),
    ):
        client.reading_at(*FUKUOKA)


def test_a_body_that_is_not_json_is_an_error(tmp_path: Path) -> None:
    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>maintenance</html>")

    with make_client(html, tmp_path) as client, pytest.raises(JshisError):
        client.reading_at(*FUKUOKA)


# --- the cache --------------------------------------------------------------


def test_the_same_point_is_fetched_once(tmp_path: Path) -> None:
    calls = 0

    def count(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return answering()(request)

    with make_client(count, tmp_path) as client:
        client.reading_at(*FUKUOKA)
        client.reading_at(*FUKUOKA)

    assert calls == 2  # two endpoints, once each


def test_the_cache_records_when_it_was_fetched(tmp_path: Path) -> None:
    """The editions are versioned and republished, so when a reading was
    taken is part of it."""
    with make_client(answering(), tmp_path) as client:
        client.reading_at(*FUKUOKA)

    bodies = list(tmp_path.rglob("*.json"))
    assert bodies
    stored = json.loads(bodies[0].read_text(encoding="utf-8"))
    assert stored["fetched_at"]


def test_a_cached_reading_keeps_the_time_it_was_fetched(tmp_path: Path) -> None:
    with make_client(answering(), tmp_path) as client:
        first = client.reading_at(*FUKUOKA).fetched_at

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with make_client(refuse, tmp_path) as second:
        assert second.reading_at(*FUKUOKA).fetched_at == first


def test_a_different_edition_is_a_different_cache_entry(tmp_path: Path) -> None:
    """Raising the edition must re-fetch. A cache keyed only by position would
    answer the new question with the old model's number."""
    with make_client(answering(), tmp_path) as client:
        client.reading_at(*FUKUOKA)

    calls = 0

    def count(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return answering()(request)

    with JshisClient(
        transport=httpx.MockTransport(count),
        cache_dir=tmp_path,
        delay_s=0.0,
        ground_edition="V3",
        hazard_edition="Y2020",
    ) as other:
        other.reading_at(*FUKUOKA)

    assert calls == 2
