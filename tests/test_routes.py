"""Tests for the Google Routes API client.

Two concerns dominate:

* **The key must not leak.** It travels in a header and must never reach a URL,
  a cache key, a log line or an exception message.
* **Arrival anchoring is solved, not assumed.** The API ignores ``arrivalTime``
  for DRIVE, so the departure time is back-solved. A solve that does not
  converge must say so rather than return its last guess as if it were an
  answer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import traceback
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pytest

from landeval.basedate import JST
from landeval.routes import (
    RoutesAuthError,
    RoutesClient,
    RoutesError,
    RoutesRateLimitError,
    api_key_from_env,
    cache_digest,
)
from landeval.tolls import DEFAULT_SHAPE, RequestShape

ORIGIN = (35.4657, 139.6222)
DEST = (35.680000, 139.760000)
FRIDAY_1000 = datetime(2027, 9, 17, 10, 0, tzinfo=JST)

Handler = Callable[[httpx.Request], httpx.Response]
SECRET = "test-key-do-not-leak"  # noqa: S105 - a fixture, not a credential


def route_response(duration_s: int, distance_m: int = 27000) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"routes": [{"duration": f"{duration_s}s", "distanceMeters": distance_m}]},
        )

    return handler


def make_client(handler: Handler, tmp_path: Path, **kwargs: object) -> RoutesClient:
    return RoutesClient(
        api_key=SECRET,
        transport=httpx.MockTransport(handler),
        cache_dir=tmp_path,
        **kwargs,  # type: ignore[arg-type]
    )


# --- key handling ----------------------------------------------------------


def test_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "  abc  ")

    assert api_key_from_env() == "abc"


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    with pytest.raises(RoutesError, match="GOOGLE_MAPS_API_KEY"):
        api_key_from_env()


def test_blank_key_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "   ")

    with pytest.raises(RoutesError, match="GOOGLE_MAPS_API_KEY"):
        api_key_from_env()


def test_key_travels_in_a_header_and_never_in_the_url(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key_header"] = request.headers.get("X-Goog-Api-Key", "")
        seen["field_mask"] = request.headers.get("X-Goog-FieldMask", "")
        return route_response(1800)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)

    assert seen["key_header"] == SECRET
    assert SECRET not in seen["url"]
    assert "routes.duration" in seen["field_mask"]


def test_cache_files_do_not_contain_the_key(tmp_path: Path) -> None:
    with make_client(route_response(1800), tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)

    written = list(tmp_path.rglob("*.json"))
    assert written
    for path in written:
        assert SECRET not in path.read_text(encoding="utf-8")
        assert SECRET not in str(path)


def test_error_messages_do_not_contain_the_key(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad request for key {SECRET}")

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError) as excinfo:
        client.route(ORIGIN, DEST, FRIDAY_1000)

    assert SECRET not in str(excinfo.value)


def test_a_response_that_echoes_the_key_does_not_put_it_in_a_duration_error(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        route = {"duration": f"echo {SECRET}", "distanceMeters": 27000}
        return httpx.Response(200, json={"routes": [route]})

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError) as excinfo:
        client.route(ORIGIN, DEST, FRIDAY_1000)

    assert SECRET not in str(excinfo.value)
    assert "duration" in str(excinfo.value)
    assert f"(str, {len(f'echo {SECRET}')} chars)" in str(excinfo.value)


def test_a_response_that_echoes_the_key_does_not_put_it_in_a_toll_error(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        route = {
            "duration": "1800s",
            "distanceMeters": 27000,
            "travelAdvisory": {"tollInfo": {"note": SECRET}},
        }
        return httpx.Response(200, json={"routes": [route]})

    shape = RequestShape(tolls=True, toll_passes=("JP_ETC",), emission_type="HYBRID")
    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError) as excinfo:
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=shape)

    assert SECRET not in str(excinfo.value)
    assert "tollInfo: dict, 1 items" in str(excinfo.value)


# --- redaction, for every key and every exit ------------------------------
#
# The key is the caller's, so nothing here may assume it is long or random. A
# key that is a substring of the replacement marker, or one whose edge
# characters meet the marker's, must still be absent from every surface an
# exception offers: its text, its args, its chain and its rendered traceback.

LONG_KEY = "synthetic-routes-key-7f3c9a1b2d4e6f80-91a2b3c4d5e6f708"
REDACTION_KEYS = [
    pytest.param(LONG_KEY, id="long"),
    pytest.param("red", id="red"),
    pytest.param("a", id="one-character"),
    pytest.param("acted", id="marker-substring"),
    pytest.param(">a", id="marker-boundary"),
]
# Keys holding a character that a quoting function escapes. ``repr`` and
# ``json.dumps`` write such a key in an escaped, reversible form that a
# verbatim replacement does not match. Every one of them can be sent: h11
# admits a tab and the printable ASCII inside a header value.
ESCAPE_KEYS = [
    pytest.param("synthetic\\routes-key-0123", id="backslash"),
    pytest.param("synthetic'routes-key-0123", id="single-quote"),
    pytest.param('synthetic"routes-key-0123', id="double-quote"),
    pytest.param("synthetic\troutes-key-0123", id="tab"),
    pytest.param("synthetic\x7froutes-key-0123", id="delete"),
]
ALL_KEYS = REDACTION_KEYS + ESCAPE_KEYS
ENTRY_POINTS = ["route", "route_for_arrival"]


def escaped_forms(key: str) -> set[str]:
    """The key as it can appear inside a message: raw, or as ``repr``,
    ``ascii`` and ``json.dumps`` write it inside a longer string.

    ``repr`` escapes a single quote only when the string also holds a double
    one, so that style is forced by quoting the key with both appended.
    """
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


def assert_no_form(text: str, key: str) -> None:
    for form in escaped_forms(key):
        assert form not in text, (form, text)


def _echo(key: str) -> str:
    """Text that repeats the key and puts its edge characters against it.

    ``key[0] + key`` and ``key + key[-1]`` are the shapes in which a marker
    that starts or ends with one of the key's own characters re-forms the key
    across the boundary (``>a`` in ``>aa`` becomes ``<redacted>a``). Both
    quote characters are in it, so ``repr`` has to escape one of them.
    """
    return f"{key[0]}{key}\"'{key}{key[-1]}!"


def _call(client: RoutesClient, entry: str, shape: RequestShape) -> object:
    if entry == "route":
        return client.route(ORIGIN, DEST, FRIDAY_1000, shape=shape)
    return client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000, shape=shape)


def _chain(error: BaseException) -> list[BaseException]:
    """Every exception reachable from ``error`` through its cause and context."""
    seen: list[BaseException] = []
    pending: list[BaseException | None] = [error]
    while pending:
        current = pending.pop()
        if current is None or any(current is known for known in seen):
            continue
        seen.append(current)
        pending.extend([current.__cause__, current.__context__])
    return seen


def assert_key_absent(error: BaseException, key: str) -> None:
    """The key is on no surface of ``error``.

    The rendered traceback is checked structurally for every key -- nothing is
    chained, and the exception line is the type and the checked message --
    and literally only for keys of eight characters or more. A one-character
    key such as ``a``
    occurs in ``Traceback (most recent call last)`` itself, so a literal check
    there would measure the traceback header rather than the redaction.
    """
    assert error.__cause__ is None, "a chained cause keeps the unredacted original"
    assert error.__context__ is None, "a chained context keeps the unredacted original"
    for reachable in _chain(error):
        assert not isinstance(reachable, httpx.HTTPError), "an httpx error carries the request"
        assert_no_form(str(reachable), key)
        assert_no_form(repr(reachable), key)
        for arg in reachable.args:
            assert_no_form(str(arg), key)
            assert_no_form(repr(arg), key)

    rendered = "".join(traceback.format_exception(error))
    assert "During handling of the above exception" not in rendered
    assert "The above exception was the direct cause" not in rendered
    module = type(error).__module__
    qualified = (
        type(error).__qualname__ if module == "builtins" else f"{module}.{type(error).__qualname__}"
    )
    assert traceback.format_exception_only(error) == [f"{qualified}: {error}\n"]
    if len(key) >= 8:
        assert_no_form(rendered, key)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_200_response_echoing_the_key_leaves_no_trace_of_it(
    tmp_path: Path, key: str, entry: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        route = {"duration": _echo(key), "distanceMeters": 27000}
        return httpx.Response(200, json={"routes": [route]})

    client = RoutesClient(key, transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with client, pytest.raises(RoutesError) as excinfo:
        _call(client, entry, DEFAULT_SHAPE)

    assert_key_absent(excinfo.value, key)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_toll_answer_echoing_the_key_leaves_no_trace_of_it(
    tmp_path: Path, key: str, entry: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        route = {
            "duration": "1800s",
            "distanceMeters": 27000,
            "travelAdvisory": {"tollInfo": {"note": _echo(key)}},
        }
        return httpx.Response(200, json={"routes": [route]})

    shape = RequestShape(tolls=True, toll_passes=("JP_ETC",), emission_type="HYBRID")
    client = RoutesClient(key, transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with client, pytest.raises(RoutesError) as excinfo:
        _call(client, entry, shape)

    # The toll path is the one that failed. Only the prefix is matched: a short
    # key rewrites the rest of the phrase ("answer" holds an `a`).
    assert str(excinfo.value).startswith("Routes toll ")
    assert_key_absent(excinfo.value, key)


# --- no response or cache value is ever quoted --------------------------
#
# A response can echo the key already escaped, any number of times. Quoting it
# escapes it once more, and no finite list of spellings then finds every copy,
# so no message quotes such a value at all: it names the field and describes
# the value by type and length. Every field the client validates is driven
# with an echo that holds the key raw and escaped one, two and three times by
# every composition of `repr` and `json.dumps`.

DEEP_KEYS = [
    pytest.param(LONG_KEY, id="plain"),
    pytest.param("synthetic\\routes\"key'-0123", id="backslash-and-quotes"),
    pytest.param("synthetic\troutes-key-0123", id="tab"),
]


def _repr_inner(text: str) -> str:
    return repr(text)[1:-1]


def _json_inner(text: str) -> str:
    return json.dumps(text)[1:-1]


def deep_spellings(key: str, depth: int = 3) -> list[str]:
    """The key, and every spelling of it after one to ``depth`` quotings."""
    quotings: tuple[Callable[[str], str], ...] = (_repr_inner, _json_inner)
    layer, spellings = [key], [key]
    for _ in range(depth):
        layer = [quote(text) for text in layer for quote in quotings]
        spellings.extend(layer)
    return spellings


def deep_echo(key: str) -> str:
    return " ".join(deep_spellings(key)) + "!"


def assert_no_spelling_at_any_depth(error: BaseException, key: str) -> None:
    assert_key_absent(error, key)
    rendered = "".join(traceback.format_exception(error))
    for reachable in _chain(error):
        surfaces = [str(reachable), repr(reachable), *map(repr, reachable.args), rendered]
        for spelling in deep_spellings(key):
            assert all(spelling not in surface for surface in surfaces), spelling


def _with_tolls(advisory: object) -> dict[str, object]:
    return {"duration": "1800s", "distanceMeters": 27000, "travelAdvisory": advisory}


def _priced(price: object) -> dict[str, object]:
    return _with_tolls({"tollInfo": {"estimatedPrice": [price]}})


def _text(echo: str) -> str:
    return f"str, {len(echo)} chars"


def _one_item(echo: str) -> str:
    return "list, 1 items"


# (case id, field as the message names it, what the message says of the value
# given the echo, the route built from the echo)
RESPONSE_FIELDS: list[tuple[str, str, Callable[[str], str], Callable[[str], dict[str, object]]]] = [
    ("duration-text", "duration", _text, lambda e: {"duration": e, "distanceMeters": 27000}),
    (
        "duration-not-numeric",
        "duration is not numeric",
        lambda e: _text(e + "s"),
        lambda e: {"duration": e + "s", "distanceMeters": 27000},
    ),
    (
        "distanceMeters",
        "distanceMeters",
        _text,
        lambda e: {"duration": "1800s", "distanceMeters": e},
    ),
    ("travelAdvisory", "travelAdvisory", lambda _: "object: str", _with_tolls),
    (
        "tollInfo-text",
        "tollInfo",
        lambda _: "object: str",
        lambda e: _with_tolls({"tollInfo": e}),
    ),
    (
        "tollInfo-no-price",
        "tollInfo",
        lambda _: "dict, 1 items",
        lambda e: _with_tolls({"tollInfo": {"note": e}}),
    ),
    (
        "estimatedPrice-text",
        "estimatedPrice",
        _text,
        lambda e: _with_tolls({"tollInfo": {"estimatedPrice": e}}),
    ),
    (
        "estimatedPrice-two",
        "estimatedPrice",
        lambda _: "got 2",
        lambda e: _with_tolls({"tollInfo": {"estimatedPrice": [e, e]}}),
    ),
    ("price-entry", "price entry", lambda _: "object: str", _priced),
    (
        "currencyCode",
        "currencyCode",
        _text,
        lambda e: _priced({"currencyCode": e, "units": "1"}),
    ),
    (
        "nanos",
        "nanos",
        _text,
        lambda e: _priced({"currencyCode": "JPY", "units": "1", "nanos": e}),
    ),
    (
        "units-type",
        "units",
        _one_item,
        lambda e: _priced({"currencyCode": "JPY", "units": [e]}),
    ),
    (
        "units-spelling",
        "units",
        _text,
        lambda e: _priced({"currencyCode": "JPY", "units": e}),
    ),
]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("key", DEEP_KEYS)
@pytest.mark.parametrize(
    ("field", "described", "build"),
    [pytest.param(*case[1:], id=case[0]) for case in RESPONSE_FIELDS],
)
def test_no_response_field_is_quoted_at_any_depth(
    tmp_path: Path,
    key: str,
    entry: str,
    field: str,
    described: Callable[[str], str],
    build: Callable[[str], dict[str, object]],
) -> None:
    echo = deep_echo(key)
    route = build(echo)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"routes": [route]})

    shape = TOLLED if "travelAdvisory" in route else DEFAULT_SHAPE
    client = RoutesClient(key, transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with client, pytest.raises(RoutesError) as excinfo:
        _call(client, entry, shape)

    # The leak first, so a run against older code fails on it rather than on
    # the wording.
    assert_no_spelling_at_any_depth(excinfo.value, key)
    message = str(excinfo.value)
    assert field in message
    assert described(echo) in message


def _cached_departure(entry: str) -> datetime:
    """The departure each entry point asks the cache about first."""
    return FRIDAY_1000 if entry == "route" else FRIDAY_1000 - timedelta(minutes=60)


# (field as the message names it, what it says of the value given the echo,
# the shape asked, body changes)
CACHED_FIELDS: list[
    tuple[str, Callable[[str], str], RequestShape, Callable[[str], dict[str, object]]]
] = [
    ("duration_s", _text, DEFAULT_SHAPE, lambda e: {"duration_s": e}),
    ("distance_m", _text, DEFAULT_SHAPE, lambda e: {"distance_m": e}),
    ("departure_time", _text, DEFAULT_SHAPE, lambda e: {"departure_time": e}),
    ("readable departure_time", _one_item, DEFAULT_SHAPE, lambda e: {"departure_time": [e]}),
    ("fetched_at", _text, DEFAULT_SHAPE, lambda e: {"fetched_at": e}),
    (
        "toll_yen",
        _text,
        RequestShape(tolls=True, toll_passes=("JP_ETC",), emission_type="HYBRID"),
        lambda e: {
            "shape": {"tolls": True, "toll_passes": ["JP_ETC"], "emission_type": "HYBRID"},
            "toll_yen": e,
        },
    ),
    ("shape", lambda _: "object: str", DEFAULT_SHAPE, lambda e: {"shape": e}),
    ("shape.tolls", _text, DEFAULT_SHAPE, lambda e: {"shape": {"tolls": e}}),
    (
        "shape.toll_passes",
        _text,
        DEFAULT_SHAPE,
        lambda e: {"shape": {"tolls": True, "toll_passes": e}},
    ),
    (
        "toll_passes",
        _one_item,
        DEFAULT_SHAPE,
        lambda e: {"shape": {"tolls": True, "toll_passes": [e]}},
    ),
    (
        "shape.emission_type",
        _one_item,
        DEFAULT_SHAPE,
        lambda e: {"shape": {"tolls": True, "toll_passes": [], "emission_type": [e]}},
    ),
    (
        "emission_type",
        _text,
        DEFAULT_SHAPE,
        lambda e: {"shape": {"tolls": True, "toll_passes": [], "emission_type": e}},
    ),
]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("key", DEEP_KEYS)
@pytest.mark.parametrize(
    ("field", "described", "shape", "change"),
    [
        pytest.param(*case, id=f"{case[0]}-{index}".replace(" ", "-"))
        for index, case in enumerate(CACHED_FIELDS)
    ],
)
def test_no_cached_field_is_quoted_at_any_depth(
    tmp_path: Path,
    key: str,
    entry: str,
    field: str,
    described: Callable[[str], str],
    shape: RequestShape,
    change: Callable[[str], dict[str, object]],
) -> None:
    echo = deep_echo(key)
    departure = _cached_departure(entry)
    body: dict[str, object] = {
        "duration_s": 1800,
        "distance_m": 27000,
        "departure_time": departure.isoformat(),
        "fetched_at": "2027-09-10T10:00:00+09:00",
    }
    body.update(change(echo))
    digest = cache_digest(ORIGIN, DEST, departure, shape)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(body), encoding="utf-8")

    def unreachable(request: httpx.Request) -> httpx.Response:
        # A cache miss would answer with this, and the field check below fails.
        return httpx.Response(500)

    client = RoutesClient(key, transport=httpx.MockTransport(unreachable), cache_dir=tmp_path)
    with client, pytest.raises(RoutesError) as excinfo:
        _call(client, entry, shape)

    assert_no_spelling_at_any_depth(excinfo.value, key)
    message = str(excinfo.value)
    assert field in message
    assert described(echo) in message


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_transport_error_leaves_no_trace_of_the_key(tmp_path: Path, key: str, entry: str) -> None:
    """The httpx error holds the request, and the request holds the key header,
    so it must not stay reachable from the error the caller gets."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused while sending {_echo(key)}", request=request)

    client = RoutesClient(key, transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with client, pytest.raises(RoutesError) as excinfo:
        _call(client, entry, DEFAULT_SHAPE)

    assert_key_absent(excinfo.value, key)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("key", ALL_KEYS)
def test_an_http_400_echoing_the_key_leaves_no_trace_of_it(
    tmp_path: Path, key: str, entry: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad request for key {_echo(key)}")

    client = RoutesClient(key, transport=httpx.MockTransport(handler), cache_dir=tmp_path)
    with client, pytest.raises(RoutesError) as excinfo:
        _call(client, entry, DEFAULT_SHAPE)

    assert_key_absent(excinfo.value, key)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_key_that_cannot_be_encoded_into_its_header_leaves_no_trace_of_it(
    tmp_path: Path, entry: str
) -> None:
    """httpx encodes a header value as ASCII and raises UnicodeEncodeError with
    the whole value in its args. The type is kept; the value is not."""
    key = "synthetic-routes-key-é-0123456789"

    client = RoutesClient(
        key, transport=httpx.MockTransport(route_response(1800)), cache_dir=tmp_path
    )
    with client, pytest.raises(UnicodeEncodeError) as excinfo:
        _call(client, entry, DEFAULT_SHAPE)

    assert_key_absent(excinfo.value, key)


def test_the_redaction_never_leaves_the_key_for_any_short_key_and_text() -> None:
    """Exhaustive over a small alphabet that holds the marker's edge characters
    and some of its interior, so the substring and boundary shapes are all in
    it. Examples alone cannot show "for any non-empty key"."""
    from landeval.routes import _redact

    alphabet = "<>rea"
    keys = [
        "".join(chars)
        for length in range(1, 4)
        for chars in itertools.product(alphabet, repeat=length)
    ]
    texts = [
        "".join(chars)
        for length in range(0, 6)
        for chars in itertools.product(alphabet, repeat=length)
    ]
    for key in keys:
        for text in texts:
            assert_no_form(_redact(text, key), key)


def test_the_redaction_leaves_no_escaped_form_for_any_short_key_and_text() -> None:
    """The same, over the characters the quoting functions escape, with each
    text also quoted the ways a message builder quotes a value."""
    from landeval.routes import _redact

    alphabet = "<>\\'\"\ta"
    keys = [
        "".join(chars)
        for length in range(1, 3)
        for chars in itertools.product(alphabet, repeat=length)
    ]
    texts = [
        "".join(chars)
        for length in range(0, 5)
        for chars in itertools.product(alphabet, repeat=length)
    ]
    for key in keys:
        for text in texts:
            for quoted in (text, repr(text), ascii(text), json.dumps(text)):
                assert_no_form(_redact(quoted, key), key)


@pytest.mark.parametrize(
    "key", [*ESCAPE_KEYS, pytest.param("synthetic-\u00e9-key-0123", id="non-ascii")]
)
def test_every_escaped_form_of_a_realistic_key_is_masked(key: str) -> None:
    from landeval.routes import _redact

    value = f"x{key}\"'{key}y"
    quotings = (
        repr(value),
        ascii(value),
        json.dumps(value),
        json.dumps(value, ensure_ascii=False),
    )
    for quoted in quotings:
        assert_no_form(_redact(quoted, key), key)


def test_a_realistic_key_keeps_the_readable_marker() -> None:
    from landeval.routes import _redact

    assert _redact(f"echo {LONG_KEY}", LONG_KEY) == "echo <redacted>"


def test_the_fallback_marker_skips_characters_the_key_holds() -> None:
    """The fallback starts at ``*``; a key holding ``*`` moves it on."""
    from landeval.routes import _redact

    redacted = _redact("x<*+y<*", "<*+")
    assert "<*+" not in redacted
    assert redacted == "x,,,y<*"


def test_an_empty_key_redacts_nothing() -> None:
    from landeval.routes import _redact

    assert _redact("anything at all", "") == "anything at all"


@pytest.mark.parametrize(
    ("value", "described"),
    [
        ("text with a key in it", "str, 21 chars"),
        (["a", "b"], "list, 2 items"),
        ({"a": 1}, "dict, 1 items"),
        (1800, "int 1800"),
        (-0.5, "float -0.5"),
        (True, "bool True"),
        (None, "NoneType None"),
        (b"raw", "bytes"),
    ],
)
def test_a_value_is_described_by_type_and_length_never_by_its_text(
    value: object, described: str
) -> None:
    """Numbers, booleans and None are shown, since once parsed they cannot hold
    the key; text and containers are described by length only."""
    from landeval.tolls import _described

    assert _described(value) == described


# --- request shape ---------------------------------------------------------


def test_request_body_asks_for_traffic_aware_driving(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["method"] = request.method
        return route_response(1800)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)

    body = seen["body"]
    assert isinstance(body, dict)
    assert seen["method"] == "POST"
    assert body["travelMode"] == "DRIVE"
    # Without this the answer is free-flow and the whole measurement is void.
    assert body["routingPreference"] == "TRAFFIC_AWARE"
    assert body["departureTime"] == "2027-09-17T10:00:00+09:00"
    assert body["origin"]["location"]["latLng"] == {"latitude": 35.4657, "longitude": 139.6222}


def test_arrival_time_is_never_sent(tmp_path: Path) -> None:
    """The API ignores arrivalTime for DRIVE; sending it would look like it worked."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return route_response(1800)(request)

    with make_client(handler, tmp_path) as client:
        client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000)

    body = seen["body"]
    assert isinstance(body, dict)
    assert "arrivalTime" not in body


def test_naive_departure_time_is_rejected(tmp_path: Path) -> None:
    """A naive timestamp serialises without an offset and is read as UTC."""
    naive = datetime(2027, 9, 17, 10, 0)

    with (
        make_client(route_response(1800), tmp_path) as client,
        pytest.raises(RoutesError, match="timezone"),
    ):
        client.route(ORIGIN, DEST, naive)


# --- response parsing ------------------------------------------------------


def test_duration_is_parsed_from_the_seconds_suffix(tmp_path: Path) -> None:
    with make_client(route_response(3660, 45000), tmp_path) as client:
        result = client.route(ORIGIN, DEST, FRIDAY_1000)

    assert result.duration_s == 3660
    assert result.duration_min == pytest.approx(61.0)
    assert result.distance_m == 45000
    assert result.arrival_time == FRIDAY_1000 + timedelta(seconds=3660)


@pytest.mark.parametrize("body", [{}, {"routes": []}, {"routes": "nope"}, {"routes": [None]}])
def test_missing_route_raises(tmp_path: Path, body: dict[str, object]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError, match="no route"):
        client.route(ORIGIN, DEST, FRIDAY_1000)


@pytest.mark.parametrize("duration", ["1234", "abc", "", "12.5x", None])
def test_malformed_duration_raises(tmp_path: Path, duration: object) -> None:
    """A route present but missing or mis-typed duration is not a usable answer."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"routes": [{"duration": duration, "distanceMeters": 100}]})

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError, match="duration"):
        client.route(ORIGIN, DEST, FRIDAY_1000)


@pytest.mark.parametrize("distance", [None, "27000", True])
def test_malformed_distance_raises(tmp_path: Path, distance: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"routes": [{"duration": "1800s", "distanceMeters": distance}]}
        )

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError, match="distance"):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_non_json_body_raises(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>")

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError, match="non-JSON"):
        client.route(ORIGIN, DEST, FRIDAY_1000)


@pytest.mark.parametrize(("status", "error"), [(401, RoutesAuthError), (403, RoutesAuthError)])
def test_auth_failures_are_distinguished(
    tmp_path: Path, status: int, error: type[Exception]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="denied")

    with make_client(handler, tmp_path) as client, pytest.raises(error):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_rate_limit_is_distinguished(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesRateLimitError):
        client.route(ORIGIN, DEST, FRIDAY_1000)


# --- caching and call accounting -------------------------------------------


def test_repeated_query_is_served_from_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return route_response(1800)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)
        client.route(ORIGIN, DEST, FRIDAY_1000)

    assert calls == 1


def test_a_different_departure_time_is_a_different_cache_entry(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return route_response(1800)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)
        client.route(ORIGIN, DEST, FRIDAY_1000 + timedelta(hours=3))

    assert calls == 2


def test_the_cache_key_distinguishes_the_two_directions(tmp_path: Path) -> None:
    """The cache key carries the direction.

    The adopted-value pass measures the same pair of coordinates at the same instant in both
    directions. If the key were symmetric the return trip would be served the
    outbound answer from cache -- silently, and with no call to show for it, so
    the only visible symptom would be a suspiciously cheap adopted-value pass.
    """
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return route_response(1800)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)
        client.route(DEST, ORIGIN, FRIDAY_1000)

    assert calls == 2


def test_a_fresh_response_records_when_it_was_fetched(tmp_path: Path) -> None:
    """Every record has to carry a fetch time, and the only place that
    knows it is the call itself. Writing it into the cache body is what makes
    the next backfill unnecessary."""
    with make_client(route_response(1800), tmp_path) as client:
        result = client.route(ORIGIN, DEST, FRIDAY_1000)

    assert result.fetched_at is not None
    assert result.fetched_at.tzinfo is not None
    body = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert body["fetched_at"] == result.fetched_at.isoformat()


def test_a_cache_body_without_a_fetch_time_reads_as_unknown(tmp_path: Path) -> None:
    """Entries written before fetch times were recorded have none. Filling one
    in at read time would record when the file was *read* under a column that
    says when the value was *fetched*. The backfill table owns those."""
    with make_client(route_response(1800), tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)
        path = next(tmp_path.rglob("*.json"))
        body = json.loads(path.read_text(encoding="utf-8"))
        del body["fetched_at"]
        path.write_text(json.dumps(body), encoding="utf-8")

        cached = client.route(ORIGIN, DEST, FRIDAY_1000)

    assert cached.fetched_at is None
    assert cached.duration_s == 1800


def test_a_cached_response_keeps_the_time_it_was_fetched(tmp_path: Path) -> None:
    """Not the time it was served from cache -- that is a different quantity."""
    with make_client(route_response(1800), tmp_path) as client:
        fresh = client.route(ORIGIN, DEST, FRIDAY_1000)
        cached = client.route(ORIGIN, DEST, FRIDAY_1000)

    assert cached.fetched_at == fresh.fetched_at


def test_billable_calls_are_counted(tmp_path: Path) -> None:
    """Cost is the binding constraint, so the client tracks what it spent."""
    with make_client(route_response(1800), tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)
        client.route(ORIGIN, DEST, FRIDAY_1000)  # cached
        client.route(ORIGIN, DEST, FRIDAY_1000 + timedelta(hours=1))

        assert client.billable_calls == 2


# --- arrival back-solving --------------------------------------------------


def test_constant_duration_converges_in_one_correction(tmp_path: Path) -> None:
    """With a duration independent of departure, one correction lands exactly."""
    with make_client(route_response(3600), tmp_path) as client:
        solution = client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000)

    assert solution.converged
    assert abs(solution.error_s) <= client.arrival_tolerance_s
    assert solution.result.arrival_time == pytest.approx(
        FRIDAY_1000.timestamp(), abs=client.arrival_tolerance_s
    ) or abs((solution.result.arrival_time - FRIDAY_1000).total_seconds()) <= (
        client.arrival_tolerance_s
    )


def test_departure_dependent_duration_still_converges(tmp_path: Path) -> None:
    """Duration rises as departure moves into the peak, as real traffic does."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        depart = datetime.fromisoformat(body["departureTime"])
        # 40 min at 06:00 growing to 80 min at 10:00.
        hours_past_six = (depart - depart.replace(hour=6, minute=0)).total_seconds() / 3600
        minutes = 40 + 10 * max(hours_past_six, 0)
        return route_response(int(minutes * 60))(request)

    with make_client(handler, tmp_path) as client:
        solution = client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000)

    assert solution.converged
    assert abs(solution.error_s) <= client.arrival_tolerance_s
    assert solution.iterations >= 2


def test_non_convergence_is_reported_not_hidden(tmp_path: Path) -> None:
    """An oscillating response must not be returned as if it had converged."""
    toggle = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        toggle[0] += 1
        return route_response(1800 if toggle[0] % 2 else 7200)(request)

    with make_client(handler, tmp_path, max_arrival_iterations=3) as client:
        solution = client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000)

    assert solution.converged is False
    assert solution.iterations == 3


def test_solver_stops_once_inside_the_tolerance(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return route_response(3600)(request)

    with make_client(handler, tmp_path) as client:
        client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000)

    assert calls <= 3  # initial guess plus at most two corrections


def test_solution_reports_the_departure_it_settled_on(tmp_path: Path) -> None:
    with make_client(route_response(3600), tmp_path) as client:
        solution = client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000)

    assert solution.result.departure_time.tzinfo is not None
    assert solution.result.departure_time < FRIDAY_1000


# --- the request shape in the key and in the body -----------------

TOLLED = RequestShape(tolls=True, toll_passes=("JP_ETC",), emission_type="HYBRID")

# The digest this pair, time and default shape produced before the shape
# existed. Pinned as a literal *and* re-derived below, because a pin that only
# echoes the implementation moves whenever the implementation does.
LEGACY_DIGEST = "e954703ac0d058fb86985e4dabb74fee"


def toll_response(duration_s: int, yen: int | None, distance_m: int = 27000) -> Handler:
    """A response carrying the toll shape the Routes API answers with."""

    def handler(request: httpx.Request) -> httpx.Response:
        route: dict[str, object] = {
            "duration": f"{duration_s}s",
            "distanceMeters": distance_m,
        }
        if yen is None:
            # Asked properly, nothing owed: present and empty.
            route["travelAdvisory"] = {}
        else:
            route["travelAdvisory"] = {
                "tollInfo": {"estimatedPrice": [{"currencyCode": "JPY", "units": str(yen)}]}
            }
        return httpx.Response(200, json={"routes": [route]})

    return handler


def test_the_default_shape_reproduces_the_digest_that_predates_it() -> None:
    """Entries written before shapes existed came from the only Routes request
    builder in this repository, which sent neither routeModifiers nor
    extraComputations. If this value moved, all of them would silently become
    unreachable and be re-purchased."""
    assert cache_digest(ORIGIN, DEST, FRIDAY_1000) == LEGACY_DIGEST


def test_the_default_digest_is_still_the_hash_of_the_three_part_token() -> None:
    """Re-derived rather than compared to the constant above, so that the pin
    and the formula have to agree with each other and not merely with
    themselves."""
    token = "|".join(
        (
            f"{ORIGIN[0]:.6f},{ORIGIN[1]:.6f}",
            f"{DEST[0]:.6f},{DEST[1]:.6f}",
            FRIDAY_1000.isoformat(),
        )
    )
    expected = hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]

    assert cache_digest(ORIGIN, DEST, FRIDAY_1000) == expected
    assert cache_digest(ORIGIN, DEST, FRIDAY_1000, shape=DEFAULT_SHAPE) == expected


def test_a_toll_shape_is_a_different_cache_entry() -> None:
    """The other direction. A shape that asks a different question must not be
    served the answer to the old one -- which is what would happen if the shape
    rode only in the body."""
    assert cache_digest(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED) != LEGACY_DIGEST


def test_two_toll_shapes_that_differ_are_different_entries() -> None:
    etc = RequestShape(tolls=True, toll_passes=("JP_ETC",), emission_type="GASOLINE")

    assert cache_digest(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED) != cache_digest(
        ORIGIN, DEST, FRIDAY_1000, shape=etc
    )


def test_a_toll_request_carries_both_halves_on_the_wire(tmp_path: Path) -> None:
    """Checked where it matters: on the request that
    actually leaves. A request missing either half still comes back 200
    looking like a free route."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["mask"] = request.headers["X-Goog-FieldMask"]
        return toll_response(1800, 1740)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["extraComputations"] == ["TOLLS"]
    assert body["routeModifiers"]["tollPasses"] == ["JP_ETC"]
    mask = seen["mask"]
    assert isinstance(mask, str)
    assert "routes.travelAdvisory.tollInfo" in mask


def test_the_default_request_still_carries_neither(tmp_path: Path) -> None:
    """The default body stays the one it has always been. Adding an empty
    routeModifiers key would make it a different request from the one the
    cache was filled with."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["mask"] = request.headers["X-Goog-FieldMask"]
        return route_response(1800)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)

    body = seen["body"]
    assert isinstance(body, dict)
    assert "routeModifiers" not in body
    assert "extraComputations" not in body
    assert seen["mask"] == "routes.duration,routes.distanceMeters"


def test_a_toll_measurement_reports_the_amount(tmp_path: Path) -> None:
    with make_client(toll_response(1800, 1740), tmp_path) as client:
        result = client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    assert result.toll_yen == 1740
    assert result.shape == TOLLED


def test_a_free_route_measured_with_tolls_reports_zero(tmp_path: Path) -> None:
    with make_client(toll_response(1800, None), tmp_path) as client:
        result = client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    assert result.toll_yen == 0


def test_a_default_measurement_reports_no_toll_at_all(tmp_path: Path) -> None:
    """Not zero. The question was not asked, and the two are different answers."""
    with make_client(route_response(1800), tmp_path) as client:
        result = client.route(ORIGIN, DEST, FRIDAY_1000)

    assert result.toll_yen is None


def test_the_cache_body_records_the_shape_it_was_fetched_under(tmp_path: Path) -> None:
    """Without this the free-or-not-asked ambiguity comes back through the
    cache: an entry that says nothing about its request is an
    entry whose empty advisory cannot be read."""
    with make_client(toll_response(1800, 1740), tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    written = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))

    assert written["shape"]["tolls"] is True
    assert written["shape"]["toll_passes"] == ["JP_ETC"]
    assert written["toll_yen"] == 1740


def test_a_default_cache_body_gains_no_shape_key(tmp_path: Path) -> None:
    """A default entry written today has to be the same bytes as one written a
    year ago, or entries written before shapes existed are a second format."""
    with make_client(route_response(1800), tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000)

    written = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))

    assert "shape" not in written
    assert "toll_yen" not in written


def test_a_cached_toll_measurement_reads_its_amount_back(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return toll_response(1800, 1740)(request)

    with make_client(handler, tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)
        again = client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    assert calls == 1
    assert again.toll_yen == 1740
    assert again.shape == TOLLED


def test_a_cached_free_toll_measurement_reads_back_as_zero(tmp_path: Path) -> None:
    with make_client(toll_response(1800, None), tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)
        again = client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    assert again.toll_yen == 0


def test_a_legacy_cache_body_reports_no_toll(tmp_path: Path) -> None:
    """Entries written without a toll shape were fetched without asking, so
    they answer "unknown" and never "free"."""
    digest = cache_digest(ORIGIN, DEST, FRIDAY_1000)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "duration_s": 1800,
                "distance_m": 27000,
                "departure_time": FRIDAY_1000.isoformat(),
            }
        ),
        encoding="utf-8",
    )

    with make_client(route_response(9999), tmp_path) as client:
        result = client.route(ORIGIN, DEST, FRIDAY_1000)

    assert result.duration_s == 1800  # served from cache, not re-fetched
    assert result.toll_yen is None


def test_a_toll_shaped_body_missing_its_amount_is_an_error(tmp_path: Path) -> None:
    """A body that says it asked but carries no answer was written by something
    that got it wrong. Reading it as zero is the failure this whole change
    exists to prevent."""
    digest = cache_digest(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "duration_s": 1800,
                "distance_m": 27000,
                "departure_time": FRIDAY_1000.isoformat(),
                "shape": {"tolls": True, "toll_passes": ["JP_ETC"], "emission_type": "HYBRID"},
            }
        ),
        encoding="utf-8",
    )

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="toll_yen"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)


def test_a_default_body_carrying_a_price_is_an_error(tmp_path: Path) -> None:
    """The mirror image: a price with no shape to justify it. Neither half may
    be trusted without the other."""
    digest = cache_digest(ORIGIN, DEST, FRIDAY_1000)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "duration_s": 1800,
                "distance_m": 27000,
                "departure_time": FRIDAY_1000.isoformat(),
                "toll_yen": 1740,
            }
        ),
        encoding="utf-8",
    )

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="toll_yen"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_arrival_solving_carries_the_shape_through(tmp_path: Path) -> None:
    """Every arrival-anchored schedule is measured this way, so a shape that
    stopped here would mean those rows silently lost their tolls."""
    with make_client(toll_response(3600, 1740), tmp_path) as client:
        solution = client.route_for_arrival(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    assert solution.result.toll_yen == 1740
    assert solution.result.shape == TOLLED


def test_an_unreadable_price_on_the_wire_is_an_error_not_a_zero(tmp_path: Path) -> None:
    """A currency outside the grammar reaches the caller as a refusal. The
    alternative -- dropping the price and keeping the duration -- would file a
    toll-shaped entry with no toll in it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "routes": [
                    {
                        "duration": "1800s",
                        "distanceMeters": 27000,
                        "travelAdvisory": {
                            "tollInfo": {"estimatedPrice": [{"currencyCode": "USD", "units": "14"}]}
                        },
                    }
                ]
            },
        )

    with (
        make_client(handler, tmp_path) as client,
        pytest.raises(RoutesError, match="toll answer is unreadable"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    assert not list(tmp_path.rglob("*.json")), "a refused answer must not be cached"


def _write_cache_entry(tmp_path: Path, shape: RequestShape, extra: dict[str, object]) -> None:
    digest = cache_digest(ORIGIN, DEST, FRIDAY_1000, shape)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {
        "duration_s": 1800,
        "distance_m": 27000,
        "departure_time": FRIDAY_1000.isoformat(),
    }
    body.update(extra)
    path.write_text(json.dumps(body), encoding="utf-8")


def test_a_cache_entry_whose_shape_cannot_be_read_is_an_error(tmp_path: Path) -> None:
    _write_cache_entry(tmp_path, DEFAULT_SHAPE, {"shape": {"tolls": "yes"}})

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="unreadable request shape"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_an_entry_filed_under_one_shape_but_recording_another_is_an_error(
    tmp_path: Path,
) -> None:
    """The shape is in the key, so these agree for everything this client wrote.
    They can disagree for an entry edited by hand, and then the answer on disk
    is to a different question than the one asked -- the cache-mediated form of
    the free-or-not-asked ambiguity on the wire."""
    _write_cache_entry(
        tmp_path,
        TOLLED,
        {"shape": {"tolls": True, "toll_passes": ["JP_ETC"], "emission_type": "GASOLINE"}},
    )

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="not the one asked for"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)


def test_a_cached_fetch_time_without_a_timezone_is_named_by_its_parsed_value(
    tmp_path: Path,
) -> None:
    """Shown only once parsed: the message carries the timestamp as
    `datetime.isoformat` writes it, never the text the cache body held."""
    _write_cache_entry(tmp_path, DEFAULT_SHAPE, {"fetched_at": "2027-09-10 10:00"})

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match=r"fetched_at 2027-09-10T10:00:00 has no timezone"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_a_cached_price_that_is_not_an_integer_is_an_error(tmp_path: Path) -> None:
    _write_cache_entry(
        tmp_path,
        TOLLED,
        {
            "shape": {"tolls": True, "toll_passes": ["JP_ETC"], "emission_type": "HYBRID"},
            "toll_yen": "1740",
        },
    )

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="not an integer"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)


# --- the same bounds, on the way in -----------------------------------------
#
# The non-negative and integer checks apply to the cache reader and the wire
# alike: a negative duration that came off the wire would be written to disk
# and then refused on the next read of the entry just created. A bound worth
# having on the way out is worth having on the way in.


@pytest.mark.parametrize(
    ("duration", "match"),
    [("-1s", "negative"), ("-0.5s", "negative")],
)
def test_a_negative_duration_on_the_wire_is_refused(
    tmp_path: Path, duration: str, match: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"routes": [{"duration": duration, "distanceMeters": 27000}]}
        )

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError, match=match):
        client.route(ORIGIN, DEST, FRIDAY_1000)

    assert not list(tmp_path.rglob("*.json")), "a refused answer must not be cached"


@pytest.mark.parametrize("distance", [-1, 27000.5])
def test_a_distance_that_is_not_a_whole_non_negative_metre_is_refused(
    tmp_path: Path, distance: object
) -> None:
    """`int(raw)` would truncate a fractional metre silently and admit a negative
    one. The API documents the field as a non-negative integer."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"routes": [{"duration": "1800s", "distanceMeters": distance}]}
        )

    with (
        make_client(handler, tmp_path) as client,
        pytest.raises(RoutesError, match="distanceMeters"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


@pytest.mark.parametrize("duration", ["nans", "infs", "-infs", "NaNs"])
def test_a_non_finite_duration_is_refused_as_a_routes_error(tmp_path: Path, duration: str) -> None:
    """`float()` accepts "nan" and "inf" and the sign test does not catch them:
    `nan < 0` is False. `int()` would then raise a bare ValueError or
    OverflowError that escapes this module's error type and aborts the run.
    Finiteness is part of "numeric"."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"routes": [{"duration": duration, "distanceMeters": 27000}]}
        )

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError, match="finite"):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_a_non_numeric_duration_is_still_refused(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"routes": [{"duration": "abcs", "distanceMeters": 27000}]})

    with make_client(handler, tmp_path) as client, pytest.raises(RoutesError, match="numeric"):
        client.route(ORIGIN, DEST, FRIDAY_1000)


@pytest.mark.parametrize("field", ["fetched_at", "toll_yen"])
def test_an_explicit_null_in_a_cache_body_is_not_an_absent_field(
    tmp_path: Path, field: str
) -> None:
    """Like the shape and the advisory, these two fields treat an explicit
    `null` as a writer that put something there and got it wrong, not as an
    entry written without the field."""
    _write_cache_entry(tmp_path, DEFAULT_SHAPE, {field: None})

    with make_client(route_response(9999), tmp_path) as client, pytest.raises(RoutesError):
        client.route(ORIGIN, DEST, FRIDAY_1000)


# --- the cache axes the reader must not leave open --------------------------


@pytest.mark.parametrize("value", [True, "1800", 1800.5, None])
def test_a_cached_duration_that_is_not_an_integer_is_an_error(
    tmp_path: Path, value: object
) -> None:
    """Narrowing with `int(...)` accepts more than it should: `True` would
    become one second. This module writes the field from an int, so anything
    else is an edited file rather than an older format."""
    _write_cache_entry(tmp_path, DEFAULT_SHAPE, {"duration_s": value})

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="duration_s"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_an_entry_whose_departure_is_not_the_key_s_departure_is_an_error(
    tmp_path: Path,
) -> None:
    """The reader must compare the recorded departure with the requested one:
    a Friday body moved under a Sunday key would otherwise be served as a
    Sunday measurement, tolls and all."""
    digest = cache_digest(ORIGIN, DEST, FRIDAY_1000)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "duration_s": 1800,
                "distance_m": 27000,
                "departure_time": (FRIDAY_1000 + timedelta(days=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="records departure"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_a_negative_cached_duration_is_an_error(tmp_path: Path) -> None:
    _write_cache_entry(tmp_path, DEFAULT_SHAPE, {"duration_s": -1})

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="negative"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_an_entry_whose_departure_is_not_a_timestamp_is_an_error(tmp_path: Path) -> None:
    digest = cache_digest(ORIGIN, DEST, FRIDAY_1000)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"duration_s": 1800, "distance_m": 27000, "departure_time": "friday"}),
        encoding="utf-8",
    )

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="not an ISO timestamp"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_an_entry_with_no_departure_at_all_is_an_error(tmp_path: Path) -> None:
    digest = cache_digest(ORIGIN, DEST, FRIDAY_1000)
    path = tmp_path / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"duration_s": 1800, "distance_m": 27000}), encoding="utf-8")

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="departure_time"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000)


def test_a_negative_cached_toll_is_an_error(tmp_path: Path) -> None:
    _write_cache_entry(
        tmp_path,
        TOLLED,
        {
            "shape": {"tolls": True, "toll_passes": ["JP_ETC"], "emission_type": "HYBRID"},
            "toll_yen": -1,
        },
    )

    with (
        make_client(route_response(9999), tmp_path) as client,
        pytest.raises(RoutesError, match="negative"),
    ):
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)


def test_a_zero_cached_toll_is_served_not_refused(tmp_path: Path) -> None:
    """The other direction. Zero is a real measured answer for a free route."""
    _write_cache_entry(
        tmp_path,
        TOLLED,
        {
            "shape": {"tolls": True, "toll_passes": ["JP_ETC"], "emission_type": "HYBRID"},
            "toll_yen": 0,
        },
    )

    with make_client(route_response(9999), tmp_path) as client:
        result = client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    assert result.toll_yen == 0


def test_the_toll_shape_does_not_leak_the_key_into_the_cache(tmp_path: Path) -> None:
    """The key stays out of the cache, also for the toll body shape."""
    with make_client(toll_response(1800, 1740), tmp_path) as client:
        client.route(ORIGIN, DEST, FRIDAY_1000, shape=TOLLED)

    for path in tmp_path.rglob("*.json"):
        assert SECRET not in path.read_text(encoding="utf-8")
