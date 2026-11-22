"""Google Routes API client -- the only source of adopted travel times.

Two things this module is careful about.

**The key.** ``computeRoutes`` authenticates with the ``X-Goog-Api-Key``
header. The key is read from the environment, never from a file, and is kept
out of every durable surface: it is not in the URL, not in a cache path or
cache body, and not in an exception message or in anything chained to one.
A response can echo the key, so no error message quotes a value taken from a
response or from a cache body: it names the field and describes the value by
type and length, and shows a number only once it has been parsed as one.

**Arrival anchoring.** A schedule is often arrival-based
("be there by 10:00"), but the API ignores ``arrivalTime`` unless the travel
mode is TRANSIT. The departure time is therefore back-solved: guess, measure,
correct. Traffic makes the trip duration a function of departure time, so this
is a fixed-point iteration and it can fail to settle. When it does not settle,
:class:`ArrivalSolution` says so; the last guess is never dressed up as an
answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from landeval.basedate import JST
from landeval.tolls import DEFAULT_SHAPE, RequestShape, TollError, _described, toll_yen

Coordinate = tuple[float, float]  # (lat, lon)

ENDPOINT = "https://routes.googleapis.com/directions/v2:computeRoutes"
API_KEY_ENV = "GOOGLE_MAPS_API_KEY"

# Only the fields actually used. A wider mask can move the request into a more
# expensive SKU, and every extra field is one more thing to keep in sync. The
# toll variant lives in `landeval.tolls` so the two halves of a toll request --
# the computation and the mask -- are assembled in one place and cannot drift
# apart (a request missing either half still returns HTTP 200).
FIELD_MASK = "routes.duration,routes.distanceMeters"

# How close the solved arrival has to land on the target before the answer is
# accepted. Two minutes is well inside the day-to-day spread of real traffic.
DEFAULT_ARRIVAL_TOLERANCE_S = 120

# Corrections allowed before giving up. Three is enough for the smooth,
# monotone way congestion varies with departure time; more usually means the
# response is not behaving like a function of departure at all.
DEFAULT_MAX_ARRIVAL_ITERATIONS = 4

# First guess for the trip length when nothing better is known.
DEFAULT_INITIAL_GUESS = timedelta(minutes=60)

_COORD_PRECISION = 6


class RoutesError(Exception):
    """Raised for a missing key, a transport failure, or an unusable response."""


class RoutesAuthError(RoutesError):
    """The key was rejected."""


class RoutesRateLimitError(RoutesError):
    """The request was throttled."""


def api_key_from_env() -> str:
    """Return the API key from the environment, failing fast when unset.

    Reading only the environment keeps the key out of the repository. Bridge a
    key file into the variable at the call site instead of reading it here.
    """
    value = os.environ.get(API_KEY_ENV, "").strip()
    if not value:
        raise RoutesError(
            f"environment variable {API_KEY_ENV} is not set. "
            f"export {API_KEY_ENV}=... before running (the key is read from that variable)"
        )
    return value


def cache_digest(
    origin: Coordinate,
    destination: Coordinate,
    departure: datetime,
    shape: RequestShape = DEFAULT_SHAPE,
) -> str:
    """The cache key for one query, derived from the query alone.

    Public because the provenance table has to look a measurement up by the
    same key the cache filed it under. Two implementations of one hash
    drift with nothing failing, so there is one -- and ``shape`` is a
    default argument for the same reason: every existing caller keeps working
    without growing a second way to compute this.

    ``shape`` is part of the key because a request that asks about tolls asks a
    different question, and the answers are not interchangeable. It contributes
    nothing in the default shape, so **entries written before tolls existed keep
    resolving** -- they were written by this same builder, in a body carrying
    neither ``routeModifiers`` nor ``extraComputations``.

    The API key is deliberately absent, so a cache directory can be inspected,
    copied or committed safely. The batch is absent too: including it would
    make every new batch re-fetch everything, which is the opposite of what
    is wanted.
    """
    parts = [
        f"{origin[0]:.{_COORD_PRECISION}f},{origin[1]:.{_COORD_PRECISION}f}",
        f"{destination[0]:.{_COORD_PRECISION}f},{destination[1]:.{_COORD_PRECISION}f}",
        departure.isoformat(),
    ]
    token_for_shape = shape.token()
    if token_for_shape:
        parts.append(token_for_shape)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class RouteResult:
    """One traffic-aware route measurement."""

    duration_s: int
    distance_m: int
    departure_time: datetime
    # When the answer came back from the API. ``None`` for a cache
    # entry written before fetch times were recorded: those have no fetch time in the body, and
    # substituting the read time would answer a different question. The
    # backfill table carries the fetch time for those.
    fetched_at: datetime | None = None
    # The toll in yen, or ``None`` when the request did not ask. Zero and None
    # are different answers and the response body cannot tell them apart -- see
    # `landeval.tolls`. ``shape`` is what makes the difference readable, so it
    # travels with the result rather than being inferred from it.
    toll_yen: int | None = None
    shape: RequestShape = DEFAULT_SHAPE

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60.0

    @property
    def arrival_time(self) -> datetime:
        return self.departure_time + timedelta(seconds=self.duration_s)


@dataclass(frozen=True)
class ArrivalSolution:
    """The outcome of back-solving a departure time for a target arrival."""

    result: RouteResult
    target_arrival: datetime
    iterations: int
    converged: bool

    @property
    def error_s(self) -> float:
        """Signed seconds by which the solved arrival misses the target."""
        return (self.result.arrival_time - self.target_arrival).total_seconds()


class RoutesClient:
    """Minimal ``computeRoutes`` client with caching and arrival back-solving."""

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: Path | str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
        arrival_tolerance_s: int = DEFAULT_ARRIVAL_TOLERANCE_S,
        max_arrival_iterations: int = DEFAULT_MAX_ARRIVAL_ITERATIONS,
    ) -> None:
        self._key = api_key
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._cache_dir = Path(cache_dir)
        self.arrival_tolerance_s = arrival_tolerance_s
        self.max_arrival_iterations = max_arrival_iterations
        self.billable_calls = 0

    def __enter__(self) -> RoutesClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- caching ----------------------------------------------------------

    def _cache_path(
        self,
        origin: Coordinate,
        destination: Coordinate,
        departure: datetime,
        shape: RequestShape = DEFAULT_SHAPE,
    ) -> Path:
        digest = cache_digest(origin, destination, departure, shape)
        return self._cache_dir / digest[:2] / f"{digest}.json"

    # -- single measurement ------------------------------------------------

    def route(
        self,
        origin: Coordinate,
        destination: Coordinate,
        departure: datetime,
        *,
        shape: RequestShape = DEFAULT_SHAPE,
    ) -> RouteResult:
        """Traffic-aware drive time from ``origin`` to ``destination``.

        ``departure`` must be timezone-aware and should be several days ahead so
        that the API answers from its historical model rather than from live
        conditions.

        ``shape`` decides whether tolls are asked for. It is part of the cache
        key and is written into the cache body, because an answer read back
        later cannot say on its own which question it answered.

        No error raised from here quotes a value from the response or the cache
        (each names the field and describes the value), and every one also has
        the configured key masked out of its message as defence in depth. The
        error type is unchanged.

        The error is rebuilt and raised **after** the ``except`` block, never
        inside it. Raised inside, the original would ride along as
        ``__context__`` -- ``from None`` only hides it from the default
        traceback -- and so would whatever it was itself raised from: an httpx
        error holding the request and its key header, or the ``float()`` and
        ``fromisoformat`` errors, which quote the raw text they refused. The
        rebuilt error therefore carries no chain at all, and the frames below
        this one are not in its traceback.
        """
        failure: RoutesError | None = None
        try:
            return self._route(origin, destination, departure, shape)
        except RoutesError as exc:
            failure = type(exc)(_redact(str(exc), self._key))
        raise failure

    def _route(
        self,
        origin: Coordinate,
        destination: Coordinate,
        departure: datetime,
        shape: RequestShape,
    ) -> RouteResult:
        if departure.tzinfo is None or departure.utcoffset() is None:
            raise RoutesError(
                f"departure {departure!r} has no timezone; a naive timestamp is read as UTC"
            )

        cache_path = self._cache_path(origin, destination, departure, shape)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            recorded = _cached_shape(cached)
            if recorded != shape:
                # The shape is in the key, so the two agree for every entry this
                # client wrote. They can disagree for one edited by hand, or one
                # written by a build whose token function differed -- and then
                # the answer on disk is to a different question than the one
                # asked. Serving it would be the cache-mediated form of the
                # ambiguity this whole change exists to remove.
                raise RoutesError(
                    f"cache entry {cache_path.name} records a request shape "
                    f"({recorded}) that is not the one asked for ({shape})"
                )
            _check_cached_departure(cache_path, cached, departure)
            return RouteResult(
                duration_s=_cached_int(cached, "duration_s"),
                distance_m=_cached_int(cached, "distance_m"),
                departure_time=departure,
                fetched_at=_parse_fetched_at(cached),
                toll_yen=_cached_toll_yen(cached),
                shape=recorded,
            )

        payload = self._post(origin, destination, departure, shape)
        try:
            price = toll_yen(_first_route(payload), shape)
        except TollError as exc:
            raise RoutesError(f"Routes toll answer is unreadable: {exc}") from None
        result = RouteResult(
            duration_s=_parse_duration(payload),
            distance_m=_parse_distance(payload),
            departure_time=departure,
            # The moment of the call, in JST because every departure in this
            # project is a JST wall-clock time and two zones in one column is
            # how a batch boundary moves by nine hours.
            fetched_at=datetime.now(JST),
            toll_yen=price,
            shape=shape,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "duration_s": result.duration_s,
            "distance_m": result.distance_m,
            "departure_time": departure.isoformat(),
        }
        if result.fetched_at is not None:
            body["fetched_at"] = result.fetched_at.isoformat()
        # Empty for the default shape, so an entry written today is the same
        # bytes as one written before tolls existed.
        body.update(shape.to_cache())
        if price is not None:
            body["toll_yen"] = price
        cache_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        return result

    def _post(
        self,
        origin: Coordinate,
        destination: Coordinate,
        departure: datetime,
        shape: RequestShape = DEFAULT_SHAPE,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "origin": {"location": {"latLng": {"latitude": origin[0], "longitude": origin[1]}}},
            "destination": {
                "location": {"latLng": {"latitude": destination[0], "longitude": destination[1]}}
            },
            "travelMode": "DRIVE",
            # Without TRAFFIC_AWARE the answer is free-flow, which OSRM already
            # provides for free -- the whole point of this layer would be lost.
            "routingPreference": "TRAFFIC_AWARE",
            "departureTime": departure.isoformat(),
        }
        # Nothing at all in the default shape: the default body stays the one
        # it has always been, and an empty `routeModifiers` key would be a
        # different request.
        body.update(shape.body_extras())
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._key,
            "X-Goog-FieldMask": shape.field_mask(),
        }

        self.billable_calls += 1
        # Both failures are raised after the handler, so neither original is
        # chained. An httpx error holds the request, whose headers hold the key,
        # and its text can quote a header value: h11 refuses a key with a line
        # break as "Illegal header value b'...'". httpx encodes header values
        # as ASCII, and the UnicodeEncodeError for a key outside it carries the
        # whole key as its `object`; it keeps its type, with the key masked.
        transport_failure: str | None = None
        unencodable: UnicodeEncodeError | None = None
        try:
            response = self._client.post(ENDPOINT, json=body, headers=headers)
        except httpx.HTTPError as exc:
            transport_failure = type(exc).__name__
        except UnicodeEncodeError as exc:
            masked = _redact(exc.object, self._key)
            unencodable = UnicodeEncodeError(exc.encoding, masked, 0, len(masked), exc.reason)
        if unencodable is not None:
            raise unencodable
        if transport_failure is not None:
            raise RoutesError(f"Routes request failed: {transport_failure}")

        if response.status_code != httpx.codes.OK:
            _raise_for_status(response.status_code)

        try:
            payload = response.json()
        except ValueError:
            raise RoutesError("Routes returned a non-JSON body") from None
        if not isinstance(payload, dict):
            raise RoutesError(f"Routes returned a non-object body of type {type(payload).__name__}")
        return payload

    # -- arrival anchoring -------------------------------------------------

    def route_for_arrival(
        self,
        origin: Coordinate,
        destination: Coordinate,
        target_arrival: datetime,
        *,
        initial_guess: timedelta = DEFAULT_INITIAL_GUESS,
        shape: RequestShape = DEFAULT_SHAPE,
    ) -> ArrivalSolution:
        """Back-solve the departure time that lands on ``target_arrival``.

        Each iteration costs one billable call, so the initial guess matters:
        passing a free-flow estimate from the free-flow layer usually saves one round trip.

        ``shape`` is carried through every iteration: a shape that stopped here
        would mean every arrival-anchored row silently lost its toll.
        """
        departure = target_arrival - initial_guess
        result = self.route(origin, destination, departure, shape=shape)
        best = result
        iterations = 1

        while iterations < self.max_arrival_iterations:
            error_s = (result.arrival_time - target_arrival).total_seconds()
            if abs(error_s) <= self.arrival_tolerance_s:
                return ArrivalSolution(result, target_arrival, iterations, converged=True)

            if abs(error_s) < abs((best.arrival_time - target_arrival).total_seconds()):
                best = result
            departure = departure - timedelta(seconds=error_s)
            result = self.route(origin, destination, departure, shape=shape)
            iterations += 1

        error_s = (result.arrival_time - target_arrival).total_seconds()
        if abs(error_s) <= self.arrival_tolerance_s:
            return ArrivalSolution(result, target_arrival, iterations, converged=True)
        if abs(error_s) < abs((best.arrival_time - target_arrival).total_seconds()):
            best = result
        return ArrivalSolution(best, target_arrival, iterations, converged=False)


def _escaped_char(char: str, quote: str, *, ascii_only: bool) -> str:
    """One character as ``repr`` (or ``ascii``) writes it inside ``quote`` quotes."""
    if char == quote:
        return "\\" + char
    if char in "'\"":
        return char
    return (ascii(char) if ascii_only else repr(char))[1:-1]


def _escaped_forms(key: str) -> list[str]:
    """Every spelling of ``key`` an error message can carry, longest first.

    ``repr`` (``!r``), ``ascii`` and ``json.dumps`` each write a string one
    character at a time, escaping backslashes, quotes, control characters and
    (``ascii``, ``json.dumps``) non-ASCII ones, so the key quoted once inside a
    longer string is spelt as one of these. Which quote ``repr`` escapes
    depends on the string around the key, so both styles are listed. Ties are
    ordered by the text, so the order is fixed.
    """
    forms = {key, json.dumps(key)[1:-1], json.dumps(key, ensure_ascii=False)[1:-1]}
    for quote in ("'", '"'):
        for ascii_only in (False, True):
            forms.add("".join(_escaped_char(c, quote, ascii_only=ascii_only) for c in key))
    return sorted(forms, key=lambda form: (-len(form), form))


def _marker_for(forms: list[str]) -> str:
    """A replacement that cannot contain any of ``forms`` or re-form one at its edges.

    ``<redacted>`` is used when no form contains ``<`` or ``>`` and none is a
    substring of it. Then any occurrence of a form in a redacted text lies
    between two brackets, so it is either inside the marker's interior, which
    the second condition rules out, or inside one stretch of text between
    markers. Such a stretch is a piece of a text that the replacement of that
    form left with no whole occurrence of it -- ``str.replace`` leaves none --
    and the later replacements only cut it into smaller pieces.

    Otherwise the marker is one character that no form contains, three times.
    No occurrence of a form can then touch a marker at all, and the same
    argument applies. A finite set of finite forms always leaves such a
    character free, so the search ends.
    """
    marker = "<redacted>"
    if all(marker[0] not in f and marker[-1] not in f and f not in marker for f in forms):
        return marker
    code = ord("*")
    while any(chr(code) in form for form in forms):
        code += 1
    return chr(code) * 3


def _redact(text: str, key: str) -> str:
    """Return ``text`` with every spelling of ``key`` masked: raw or escaped.

    Defence in depth. No message here quotes a response or cache value, so
    this is for exception text that comes from elsewhere -- a transport error,
    or the encoding error for a key httpx cannot put in a header -- and for a
    short key that occurs in a message's own words. A very short key rewrites
    those words (the key ``a`` turns ``failed`` into ``f***iled``); that is
    accepted, because keeping the key out comes before keeping the message
    readable, and refusing such a key would be a new rule.
    """
    if not key:
        return text
    forms = _escaped_forms(key)
    marker = _marker_for(forms)
    for form in forms:
        text = text.replace(form, marker)
    return text


def _parse_fetched_at(cached: dict[str, Any]) -> datetime | None:
    """The fetch time of a cache entry, or ``None`` when it predates fetch-time recording.

    A malformed value is an error rather than a ``None``: absent means "written
    before the field existed", and answering the same way for "written wrong"
    would hide a writer that is currently getting it wrong.
    """
    # Membership, like every other optional field in this body. An explicit
    # `null` is not "written before the field existed" -- it is a writer that
    # put something there and got it wrong, and answering the same way for
    # both hides the one that is wrong now. The shape and the advisory
    # are read the same way.
    if "fetched_at" not in cached:
        return None
    raw = cached["fetched_at"]
    if not isinstance(raw, str):
        raise RoutesError(f"cached fetched_at is not a string: {type(raw).__name__}")
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        raise RoutesError(
            f"cached fetched_at is not an ISO timestamp ({_described(raw)})"
        ) from None
    if moment.tzinfo is None:
        raise RoutesError(f"cached fetched_at {moment.isoformat()} has no timezone")
    return moment


def _cached_shape(cached: dict[str, Any]) -> RequestShape:
    """The shape a cache entry was written under, or an error."""
    try:
        return RequestShape.from_cache(cached)
    except TollError as exc:
        raise RoutesError(f"cache entry records an unreadable request shape: {exc}") from None


def _cached_toll_yen(cached: dict[str, Any]) -> int | None:
    """The price in a cache entry, checked against the shape that entry claims.

    The two halves are required to agree. A body that says it asked and carries
    no price was written by something that got it wrong, and a body that
    carries a price without a shape to justify it cannot say whether the price
    means "free" or "never asked" -- which is the ambiguity this whole change
    exists to keep out of the cache. Neither is answered with a guess.
    """
    shape = _cached_shape(cached)
    present = "toll_yen" in cached
    raw = cached.get("toll_yen")
    if present and raw is None:
        raise RoutesError(
            "cache entry carries an explicit null toll_yen; that is a writer that "
            "got it wrong, not an entry from before the field existed"
        )
    if not shape.tolls:
        if present:
            raise RoutesError(
                "cache entry carries toll_yen but records no toll-bearing request shape; "
                "the price cannot be told apart from an unasked question"
            )
        return None
    if not present:
        raise RoutesError(
            "cache entry records a toll-bearing request shape but carries no toll_yen"
        )
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RoutesError(f"cached toll_yen is not an integer ({_described(raw)})")
    if raw < 0:
        raise RoutesError(
            f"cached toll_yen is negative ({raw}); no API response carries a negative toll"
        )
    return raw


def _cached_int(cached: dict[str, Any], key: str) -> int:
    """One of the two numeric legacy fields, narrowed without coercing.

    ``int(...)`` accepts more than it should: ``True`` would become one
    second, and a float or a numeric string whatever it rounds to. This module
    writes the field from an int, so anything else is an edited file rather than
    an older format, and reading it
    as a measurement is how a hand-typed value becomes an adopted number.
    """
    raw = cached.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RoutesError(f"cached {key} is not an integer ({_described(raw)})")
    if raw < 0:
        raise RoutesError(f"cached {key} is negative: {raw}")
    return raw


def _check_cached_departure(path: Path, cached: dict[str, Any], departure: datetime) -> None:
    """The entry's own departure has to be the one the key was built from.

    A reader that substituted the requested departure into the result without
    looking at the recorded one would let an entry moved to another key answer
    a question it had never been asked -- a Friday body served as a Sunday
    measurement, tolls and all. The body carries the field; comparing it costs
    nothing and closes the axis.

    An entry with no ``departure_time`` predates nothing: this module has
    written the field since the cache existed, so its absence is an edited
    file, not an older format.
    """
    raw = cached.get("departure_time")
    if not isinstance(raw, str):
        raise RoutesError(
            f"cache entry {path.name} has no readable departure_time ({_described(raw)})"
        )
    try:
        recorded = datetime.fromisoformat(raw)
    except ValueError:
        raise RoutesError(
            f"cache entry {path.name} departure_time is not an ISO timestamp ({_described(raw)})"
        ) from None
    if recorded != departure:
        raise RoutesError(
            f"cache entry {path.name} records departure {recorded.isoformat()}, "
            f"but the key was built from {departure.isoformat()}"
        )


def _raise_for_status(status: int) -> None:
    if status in (401, 403):
        raise RoutesAuthError(f"Routes auth failed (HTTP {status}); check {API_KEY_ENV}")
    if status == 429:
        raise RoutesRateLimitError("Routes rate limit (HTTP 429); space out requests")
    # The body is not included: an error echo can contain the submitted request.
    raise RoutesError(f"Routes request failed (HTTP {status})")


def _first_route(payload: dict[str, Any]) -> dict[str, Any]:
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes or not isinstance(routes[0], dict):
        raise RoutesError("Routes returned no route for the requested pair")
    return routes[0]


def _parse_duration(payload: dict[str, Any]) -> int:
    route = _first_route(payload)
    raw = route.get("duration")
    if not isinstance(raw, str) or not raw.endswith("s"):
        raise RoutesError(f"Routes duration is not a seconds string ({_described(raw)})")
    try:
        seconds = float(raw[:-1])
    except ValueError:
        raise RoutesError(f"Routes duration is not numeric ({_described(raw)})") from None
    # The same bound the cache reader applies, and applied **before** the
    # truncation rather than after. The reader and this side must agree: a
    # `duration: "-1s"` that came off the wire would be written to disk and then
    # refused on the next read of the entry just created. Testing after `int()`
    # would close only half of that -- `int(float("-0.5"))` truncates toward zero, so a negative
    # fractional second became a confident 0.
    # `float()` accepts "nan" and "inf", and neither is caught by the sign test
    # below: `nan < 0` is False, and `int()` then raises a bare ValueError or
    # OverflowError that escapes this module's error type entirely and aborts
    # the run. Finiteness is part of "numeric".
    if not math.isfinite(seconds):
        raise RoutesError(f"Routes duration is not a finite number: {seconds!r} seconds")
    if seconds < 0:
        raise RoutesError(f"Routes duration is negative: {seconds!r} seconds")
    return int(seconds)


def _parse_distance(payload: dict[str, Any]) -> int:
    route = _first_route(payload)
    raw = route.get("distanceMeters")
    if isinstance(raw, bool) or not isinstance(raw, int):
        # Integers only, not floats. `int(raw)` truncated a fractional metre
        # silently, and the API documents the field as an integer.
        raise RoutesError(f"Routes distanceMeters is not an integer ({_described(raw)})")
    if raw < 0:
        raise RoutesError(f"Routes distanceMeters is negative: {raw}")
    return raw
