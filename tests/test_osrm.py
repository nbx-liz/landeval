"""Tests for the OSRM client.

All cases run against an in-process transport, so the suite needs no server.
The emphasis is the rejection matrix: a malformed or short response must raise
rather than yield plausible-looking rows.

The table service leaves every coordinate as a source and marks only the last
one as the destination, so a well-formed response carries one row *more* than
the batch. Dropping that trailing row is part of the contract under test.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from landeval.osrm import MAX_URL_LENGTH, OsrmClient, OsrmError, TableRow

TOKYO = (35.6812, 139.7671)
DEST = (35.6800, 139.7600)

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler) -> OsrmClient:
    return OsrmClient("http://osrm.test", transport=httpx.MockTransport(handler))


def coordinate_count(request: httpx.Request) -> int:
    return len(request.url.path.rsplit("/", 1)[-1].split(";"))


def ok_table(durations: list[float | None], snap: list[float] | None = None) -> Handler:
    """Handler returning a well-formed table for a batch with these durations."""
    snap_distances = snap if snap is not None else [0.0] * len(durations)
    rows: list[float | None] = [*durations, 0.0]  # trailing destination-to-itself row

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "durations": [[d] for d in rows],
                "distances": [[None if d is None else d * 10] for d in rows],
                "sources": [{"distance": s} for s in [*snap_distances, 0.0]],
                "destinations": [{"distance": 0.0}],
            },
        )

    return handler


# --- happy path ------------------------------------------------------------


def test_returns_one_row_per_source() -> None:
    with make_client(ok_table([600.0, 1200.0, 1800.0])) as client:
        rows = list(client.table_to_destination([TOKYO, TOKYO, TOKYO], DEST))

    assert [r.duration_s for r in rows] == [600.0, 1200.0, 1800.0]
    assert [r.duration_min for r in rows] == [10.0, 20.0, 30.0]
    assert all(r.is_reachable for r in rows)


def test_the_destinations_own_row_is_dropped() -> None:
    with make_client(ok_table([600.0, 1200.0])) as client:
        rows = list(client.table_to_destination([TOKYO, TOKYO], DEST))

    assert len(rows) == 2


def test_unreachable_source_stays_none() -> None:
    """'No route' must never be coerced into a number."""
    with make_client(ok_table([600.0, None])) as client:
        rows = list(client.table_to_destination([TOKYO, TOKYO], DEST))

    assert rows[1].duration_s is None
    assert rows[1].duration_min is None
    assert rows[1].is_reachable is False
    assert len(rows) == 2  # not dropped


def test_snap_distance_is_reported() -> None:
    with make_client(ok_table([600.0, 600.0], snap=[3.5, 4200.0])) as client:
        rows = list(client.table_to_destination([TOKYO, TOKYO], DEST))

    assert rows[0].snap_distance_m == 3.5
    assert rows[1].snap_distance_m == 4200.0


def test_empty_source_list_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with make_client(handler) as client:
        assert list(client.table_to_destination([], DEST)) == []


# --- request shape ---------------------------------------------------------


def test_destination_is_last_and_sources_are_left_implicit() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return ok_table([1.0, 2.0])(request)

    with make_client(handler) as client:
        list(client.table_to_destination([TOKYO, TOKYO], DEST))

    params = seen["params"]
    assert isinstance(params, dict)
    assert params["destinations"] == "2"
    # Listing the sources explicitly would roughly double the URL length.
    assert "sources" not in params
    path = seen["path"]
    assert isinstance(path, str)
    # OSRM takes lon,lat -- a swap here would silently route to the wrong place.
    assert path.endswith("139.760000,35.680000")


def test_sources_are_batched_and_order_is_preserved() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        n = coordinate_count(request) - 1
        calls.append(n)
        offset = sum(calls[:-1])
        return ok_table([float(offset + i) for i in range(n)])(request)

    with make_client(handler) as client:
        rows = list(client.table_to_destination([TOKYO] * 7, DEST, chunk_size=3))

    assert calls == [3, 3, 1]
    assert [r.duration_s for r in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size: int) -> None:
    with make_client(ok_table([1.0])) as client, pytest.raises(OsrmError, match="chunk_size"):
        list(client.table_to_destination([TOKYO], DEST, chunk_size=chunk_size))


def test_over_long_url_is_rejected_before_sending() -> None:
    """A batch too large for the URL must fail loudly, not be silently truncated."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    oversized = MAX_URL_LENGTH // 21 + 100
    with make_client(handler) as client, pytest.raises(OsrmError, match="chunk_size"):
        list(client.table_to_destination([TOKYO] * oversized, DEST, chunk_size=oversized))


# --- rejection matrix ------------------------------------------------


def test_http_error_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with make_client(handler) as client, pytest.raises(OsrmError, match="HTTP 500"):
        list(client.table_to_destination([TOKYO], DEST))


def test_transport_failure_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with make_client(handler) as client, pytest.raises(OsrmError, match="request failed"):
        list(client.table_to_destination([TOKYO], DEST))


def test_non_json_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with make_client(handler) as client, pytest.raises(OsrmError, match="non-JSON"):
        list(client.table_to_destination([TOKYO], DEST))


def test_non_object_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=json.dumps([1, 2, 3]), headers={"content-type": "application/json"}
        )

    with make_client(handler) as client, pytest.raises(OsrmError, match="non-object"):
        list(client.table_to_destination([TOKYO], DEST))


@pytest.mark.parametrize("code", ["NoRoute", "InvalidQuery", "TooBig", "NoSegment"])
def test_error_code_raises(code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": code, "message": "nope"})

    with make_client(handler) as client, pytest.raises(OsrmError, match=code):
        list(client.table_to_destination([TOKYO], DEST))


def test_missing_durations_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "Ok", "sources": [{"distance": 0.0}]})

    with make_client(handler) as client, pytest.raises(OsrmError, match="missing"):
        list(client.table_to_destination([TOKYO], DEST))


def test_missing_sources_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "Ok", "durations": [[1.0], [0.0]]})

    with make_client(handler) as client, pytest.raises(OsrmError, match="missing"):
        list(client.table_to_destination([TOKYO], DEST))


def test_short_duration_list_raises() -> None:
    """A response covering fewer sources than requested must not silently truncate."""
    with (
        make_client(ok_table([1.0])) as client,
        pytest.raises(OsrmError, match="requested sources"),
    ):
        list(client.table_to_destination([TOKYO, TOKYO, TOKYO], DEST))


def test_response_without_the_destination_row_raises() -> None:
    """Exactly len(batch) rows means OSRM did not treat the last coordinate as a source."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "durations": [[1.0], [2.0]],
                "distances": [[1.0], [2.0]],
                "sources": [{"distance": 0.0}, {"distance": 0.0}],
                "destinations": [{"distance": 0.0}],
            },
        )

    with make_client(handler) as client, pytest.raises(OsrmError, match="requested sources"):
        list(client.table_to_destination([TOKYO, TOKYO], DEST))


def test_multi_destination_row_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "durations": [[1.0, 2.0], [0.0, 0.0]],
                "distances": [[1.0, 2.0], [0.0, 0.0]],
                "sources": [{"distance": 0.0}, {"distance": 0.0}],
                "destinations": [{"distance": 0.0}, {"distance": 0.0}],
            },
        )

    with make_client(handler) as client, pytest.raises(OsrmError, match="single-destination"):
        list(client.table_to_destination([TOKYO], DEST))


def test_source_without_snap_distance_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "durations": [[1.0], [0.0]],
                "distances": [[1.0], [0.0]],
                "sources": [{"location": [139.0, 35.0]}, {"distance": 0.0}],
                "destinations": [{"distance": 0.0}],
            },
        )

    with make_client(handler) as client, pytest.raises(OsrmError, match="snap distance"):
        list(client.table_to_destination([TOKYO], DEST))


def test_table_row_helpers() -> None:
    assert TableRow(None, None, 0.0).duration_min is None
    assert TableRow(90.0, 100.0, 0.0).duration_min == 1.5


# --- /route, the node annotation the nearest-interchange rule reads ----------


def ok_route(nodes: list[int], coordinates: list[list[float]], durations: list[float]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "routes": [
                    {
                        "legs": [{"annotation": {"nodes": nodes, "duration": durations}}],
                        "geometry": {"coordinates": coordinates},
                    }
                ],
            },
        )

    return handler


def test_route_nodes_returns_a_cumulative_clock_beside_the_nodes() -> None:
    """The clock is what makes the entry time exact: `cumulative_s[i]` is when
    the route reached `nodes[i]`, not when it started the step containing it."""
    handler = ok_route([1, 2, 3], [[139.0, 35.0], [139.1, 35.1], [139.2, 35.2]], [60.0, 30.0])
    with make_client(handler) as client:
        route = client.route_nodes(TOKYO, DEST)
    assert route is not None
    assert route.nodes == (1, 2, 3)
    assert route.cumulative_s == (0.0, 60.0, 90.0)
    # OSRM answers lon,lat; everything else in this codebase is lat,lon.
    assert route.coordinates == ((35.0, 139.0), (35.1, 139.1), (35.2, 139.2))


def test_route_nodes_requests_the_annotation_it_depends_on() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return ok_route([1, 2], [[139.0, 35.0], [139.1, 35.1]], [1.0])(request)

    with make_client(handler) as client:
        client.route_nodes(TOKYO, DEST)
    assert "annotations=nodes%2Cduration" in seen[0]
    assert "overview=full" in seen[0]


@pytest.mark.parametrize("code", ["NoRoute", "NoSegment"])
def test_a_route_that_does_not_exist_is_none_rather_than_an_error(code: str) -> None:
    """No route and a route that uses no expressway are different answers, and
    the caller prints different things for them.

    The shape here is what a running OSRM server answers, not what was
    convenient to construct: an unroutable destination returns **HTTP 400**
    with `{"code": "NoRoute", "message": "Impossible route between points"}`. A
    fixture inventing a 200 carrying an empty `routes` list would let the branch
    pass while being unreachable in production, where the client would reject
    the real 400 as a transport failure.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"code": code, "message": "Impossible route between points"}
        )

    with make_client(handler) as client:
        assert client.route_nodes(TOKYO, DEST) is None


def test_an_ok_answer_carrying_no_route_is_refused_rather_than_read_as_no_route() -> None:
    """ "There is no route" and "I could not read the reply" must not collapse
    into the same silence. `Ok` with no usable `routes` is the second one."""
    bodies: list[dict[str, object]] = [
        {"code": "Ok"},
        {"code": "Ok", "routes": []},
        {"code": "Ok", "routes": "x"},
    ]
    for payload in bodies:

        def handler(request: httpx.Request, payload: dict[str, object] = payload) -> httpx.Response:
            return httpx.Response(200, json=payload)

        with make_client(handler) as client, pytest.raises(OsrmError, match="no usable route"):
            client.route_nodes(TOKYO, DEST)


@pytest.mark.parametrize("status", [500, 502, 503, 404])
def test_a_no_route_code_on_the_wrong_status_is_a_failure_not_an_answer(status: int) -> None:
    """A 500 carrying `{"code": "NoRoute"}` is a server that fell over, and
    reading the code without the status reports the crash as a routing answer.

    The service pairs `NoRoute` with 400 and nothing else. Accepting
    the code on any status is the shape this test exists to refuse.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"code": "NoRoute", "message": "Impossible route"})

    with make_client(handler) as client, pytest.raises(OsrmError, match=f"HTTP {status}"):
        client.route_nodes(TOKYO, DEST)


def test_an_http_error_that_is_not_a_no_route_code_still_raises() -> None:
    """The `answers` set is two codes wide, not "any 400 with a body"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": "InvalidOptions", "message": "bad"})

    with make_client(handler) as client, pytest.raises(OsrmError, match="HTTP 400"):
        client.route_nodes(TOKYO, DEST)


def test_a_route_without_the_annotation_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": "Ok",
                "routes": [{"legs": [{}], "geometry": {"coordinates": [[139.0, 35.0]]}}],
            },
        )

    with make_client(handler) as client, pytest.raises(OsrmError, match="no annotation"):
        client.route_nodes(TOKYO, DEST)


def test_a_route_whose_arrays_do_not_line_up_raises() -> None:
    """One duration short of the node count means the clock and the nodes are
    not the same route, and every minute figure read off it is wrong."""
    handler = ok_route([1, 2, 3], [[139.0, 35.0], [139.1, 35.1], [139.2, 35.2]], [60.0])
    with make_client(handler) as client, pytest.raises(OsrmError, match="must line up"):
        client.route_nodes(TOKYO, DEST)
