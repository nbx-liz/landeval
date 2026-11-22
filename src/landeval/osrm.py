"""Client for the local OSRM server (the free-flow layer).

Two properties matter more than convenience here:

* **Unreachable is not slow.** OSRM returns ``null`` for an unreachable pair.
  That stays ``None`` all the way out of this module; it is never coerced to a
  large number or silently dropped, because "no route" and "a long route" lead
  to different decisions.
* **Snap distance is reported.** A mesh centre in the sea or deep in a forest
  snaps to a road far away and would otherwise get a plausible-looking travel
  time. Callers filter on ``snap_distance_m`` to reject those cells.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import httpx

from landeval.interchange import RouteNodes

Coordinate = tuple[float, float]  # (lat, lon)

DEFAULT_BASE_URL = "http://127.0.0.1:5000"

# The table service takes its coordinates in the URL path, so the batch size is
# bounded by URL length rather than by OSRM's --max-table-size. One coordinate
# costs 21 characters, and httpx refuses to build a URL beyond 65536.
MAX_URL_LENGTH = 60000
DEFAULT_CHUNK_SIZE = 2500

# The service codes that mean "there is no such route" rather than "the request
# or the server was wrong". The running server, not its documentation, decides
# this: an unroutable destination is answered with **HTTP 400** and
# `{"code": "NoRoute", "message": "Impossible route between points"}`, so a
# caller checking only for an empty `routes` list on a 200 has written a branch
# nothing can reach. `NoSegment` is the sibling case -- a coordinate that
# snaps to nothing -- and is the same answer to the user.
NO_ROUTE_CODES = frozenset({"NoRoute", "NoSegment"})
# ...and only on the status the service pairs them with. A 500 carrying
# `{"code": "NoRoute"}` is a server that fell over, not an answer, and reading
# the code without the status turns one into the other.
NO_ROUTE_STATUS = frozenset({httpx.codes.OK, httpx.codes.BAD_REQUEST})


class OsrmError(Exception):
    """Raised when OSRM is unreachable or answers with anything but a usable table."""


@dataclass(frozen=True)
class TableRow:
    """One source's result against the requested destination."""

    duration_s: float | None
    distance_m: float | None
    snap_distance_m: float

    @property
    def is_reachable(self) -> bool:
        return self.duration_s is not None

    @property
    def duration_min(self) -> float | None:
        return None if self.duration_s is None else self.duration_s / 60.0


def _format(coord: Coordinate) -> str:
    """OSRM takes lon,lat -- the opposite order from the rest of this codebase."""
    lat, lon = coord
    return f"{lon:.6f},{lat:.6f}"


class OsrmClient:
    """Minimal OSRM HTTP client covering the services this project needs."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def __enter__(self) -> OsrmClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _get(
        self, url: str, params: dict[str, str], *, answers: frozenset[str] = frozenset()
    ) -> dict[str, object]:
        """The payload, or an error naming what OSRM said instead.

        `answers` lists service codes that are answers rather than failures. It
        exists because OSRM reports "there is no route" as **HTTP 400 with
        `code: "NoRoute"`**, not as an empty `routes` list on a 200. A caller
        written from the documentation alone has a branch for a state the
        service does not produce. Without this, an
        unroutable pair reaches the caller as "OSRM is unreachable", which is a
        different thing to tell a user.

        The status has to agree. An answer code on a 500 is a server that fell
        over while a proxy or a partial write left a plausible body behind, and
        accepting it would report a crash as "there is no route".
        """
        try:
            response = self._client.get(url, params=params)
        except httpx.HTTPError as exc:
            raise OsrmError(f"OSRM request failed: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            if response.status_code != httpx.codes.OK:
                raise OsrmError(
                    f"OSRM returned HTTP {response.status_code}: {response.text[:200]}"
                ) from exc
            raise OsrmError(f"OSRM returned a non-JSON body: {response.text[:200]}") from exc

        if not isinstance(payload, dict):
            raise OsrmError(f"OSRM returned a non-object body: {payload!r}")
        code = payload.get("code")
        if isinstance(code, str) and code in answers and response.status_code in NO_ROUTE_STATUS:
            return payload
        if response.status_code != httpx.codes.OK:
            raise OsrmError(f"OSRM returned HTTP {response.status_code}: {response.text[:200]}")
        if code != "Ok":
            raise OsrmError(f"OSRM error {code!r}: {payload.get('message')!r}")
        return payload

    def route_nodes(self, origin: Coordinate, destination: Coordinate) -> RouteNodes | None:
        """One route, as the OSM node ids it traverses and the clock at each.

        `None` means OSRM reported `NoRoute`, which stays distinct from a route
        that exists and uses no expressway -- the caller prints different
        things for the two. It does **not** mean the answer was unreadable: a
        malformed body raises, because "there is no route" and "I could not read
        the reply" are the same silence otherwise.

        The node annotation is what makes the nearest-interchange rule exact:
        every other field OSRM offers describes steps, and a step boundary is a
        lower bound on when the route reached a point inside it, so reading the
        entry off the step start reports a different time from the node clock.
        """
        coords = f"{_format(origin)};{_format(destination)}"
        payload = self._get(
            f"/route/v1/driving/{coords}",
            {"annotations": "nodes,duration", "geometries": "geojson", "overview": "full"},
            answers=NO_ROUTE_CODES,
        )
        if payload.get("code") in NO_ROUTE_CODES:
            return None
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            raise OsrmError(f"OSRM answered Ok with no usable route: {payload.get('routes')!r}")
        route = routes[0]
        if not isinstance(route, dict):
            raise OsrmError(f"OSRM route is not an object: {route!r}")
        legs = route.get("legs")
        geometry = route.get("geometry")
        if not isinstance(legs, list) or len(legs) != 1 or not isinstance(geometry, dict):
            raise OsrmError("OSRM route is missing a single leg or its geometry")
        annotation = legs[0].get("annotation")
        if not isinstance(annotation, dict):
            raise OsrmError("OSRM route leg carries no annotation; was it requested?")
        nodes = annotation.get("nodes")
        durations = annotation.get("duration")
        coordinates = geometry.get("coordinates")
        if not (
            isinstance(nodes, list)
            and isinstance(durations, list)
            and isinstance(coordinates, list)
        ):
            raise OsrmError("OSRM route annotation is missing 'nodes', 'duration' or 'coordinates'")
        if len(coordinates) != len(nodes) or len(durations) != len(nodes) - 1:
            raise OsrmError(
                f"OSRM route has {len(nodes)} nodes, {len(coordinates)} coordinates and "
                f"{len(durations)} durations; the three must line up"
            )
        clock = [0.0]
        for duration in durations:
            clock.append(clock[-1] + float(duration))
        return RouteNodes(
            nodes=tuple(int(node) for node in nodes),
            # OSRM answers lon,lat; the rest of this codebase is lat,lon.
            coordinates=tuple((float(c[1]), float(c[0])) for c in coordinates),
            cumulative_s=tuple(clock),
        )

    def table_to_destination(
        self,
        sources: Sequence[Coordinate],
        destination: Coordinate,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Iterator[TableRow]:
        """Yield one :class:`TableRow` per source, in the order given.

        Requests are batched; every batch carries the destination as its last
        coordinate so that each row is a source-to-destination pair.
        """
        if chunk_size < 1:
            raise OsrmError(f"chunk_size must be positive, got {chunk_size}")

        for start in range(0, len(sources), chunk_size):
            batch = sources[start : start + chunk_size]
            yield from self._table_chunk(batch, destination)

    def _table_chunk(
        self, batch: Sequence[Coordinate], destination: Coordinate
    ) -> Iterator[TableRow]:
        # The destination goes last and every coordinate is left as a source.
        # Listing the source indices explicitly would roughly double the URL
        # length for no benefit: the destination's own row is simply dropped.
        coords = ";".join(_format(c) for c in (*batch, destination))
        url = f"/table/v1/driving/{coords}"
        if len(url) > MAX_URL_LENGTH:
            raise OsrmError(
                f"batch of {len(batch)} coordinates builds a {len(url)}-character URL, "
                f"over the {MAX_URL_LENGTH} limit; lower chunk_size"
            )
        params = {"destinations": str(len(batch)), "annotations": "duration,distance"}
        payload = self._get(url, params)

        durations = payload.get("durations")
        distances = payload.get("distances")
        snapped = payload.get("sources")

        if not isinstance(durations, list) or not isinstance(snapped, list):
            raise OsrmError("OSRM table response is missing 'durations' or 'sources'")
        expected = len(batch) + 1  # sources include the destination itself
        if len(durations) != expected or len(snapped) != expected:
            raise OsrmError(
                f"OSRM table response has {len(durations)} duration rows and "
                f"{len(snapped)} sources for {len(batch)} requested sources"
            )

        for index, row in enumerate(durations[: len(batch)]):
            if not isinstance(row, list) or len(row) != 1:
                raise OsrmError(f"OSRM duration row {index} is not a single-destination row: {row}")
            distance_row = distances[index] if isinstance(distances, list) else [None]
            source = snapped[index]
            if not isinstance(source, dict) or "distance" not in source:
                raise OsrmError(f"OSRM source {index} is missing its snap distance: {source!r}")
            yield TableRow(
                duration_s=row[0],
                distance_m=distance_row[0] if isinstance(distance_row, list) else None,
                snap_distance_m=float(source["distance"]),
            )
