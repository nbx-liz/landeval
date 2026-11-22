"""Fetching and reading the inland-flood raster tiles.

The layer is judged from painted pixels rather than from polygons, so the
failure modes are different from the vector layers':

- **404 is not one answer.** Inside a roster municipality it means the point is
  outside the zone; outside the roster it means nothing was ever published.
  The client reports which HTTP answer it got and lets the caller apply the
  roster, because a client that decided this itself would be deciding the
  question the roster exists to answer.
- **A 500 is not a 404.** A server error read as "outside" paints unlooked-at
  ground clear, so anything that is neither 200 nor 404 raises.
- **The cache remembers absence.** A 404 that is not recorded is refetched
  every run, and a run that refetches is a run whose earlier answer nobody kept.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from PIL import Image

from landeval.inland_flood import (
    ABSENT,
    PRESENT,
    InlandFloodError,
    TileClient,
    pixel_index,
    tile_index,
)

# A municipal office: a fixed, public point on land.
FUKUOKA = (33.59018, 130.40170)
ZOOM = 16

Handler = Callable[[httpx.Request], httpx.Response]


def painted_png(*, alpha: int = 255) -> bytes:
    """A tile whose every pixel is painted, or transparent when alpha is 0."""
    image = Image.new("RGBA", (256, 256), (255, 0, 0, alpha))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def responder(status: int, body: bytes = b"") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return handler


def make_client(handler: Handler, tmp_path: Path, **kwargs: object) -> TileClient:
    return TileClient(
        transport=httpx.MockTransport(handler),
        cache_dir=tmp_path,
        delay_s=0.0,
        **kwargs,  # type: ignore[arg-type]
    )


# --- the lattice arithmetic -------------------------------------------------


def test_the_tile_index_of_a_known_coordinate() -> None:
    """This office at z16 is in tile 56506/26269 of the Web Mercator grid."""
    assert tile_index(*FUKUOKA, ZOOM) == (56506, 26269)


def test_the_pixel_index_stays_inside_the_tile() -> None:
    x, y = pixel_index(*FUKUOKA, ZOOM)

    assert 0 <= x < 256
    assert 0 <= y < 256


def test_two_points_100_m_apart_are_different_pixels() -> None:
    """A z16 tile is about 610 m across in 256 pixels, so the 100 m sample
    spacing is roughly 42 pixels. If it were not, every point in a cell would
    read the same pixel and the sample lattice would be decorative."""
    lat, lon = FUKUOKA

    assert pixel_index(lat + 0.0009, lon, ZOOM) != pixel_index(lat, lon, ZOOM)


# --- what the client answers ------------------------------------------------


def test_a_painted_pixel_is_inside_the_zone(tmp_path: Path) -> None:
    with make_client(responder(200, painted_png()), tmp_path) as client:
        state, inside = client.reading_at(*FUKUOKA)

    assert state == PRESENT
    assert inside


def test_a_transparent_pixel_is_outside_the_zone(tmp_path: Path) -> None:
    """The tile exists, so the municipality publishes; this point simply is not
    in the drawn area."""
    with make_client(responder(200, painted_png(alpha=0)), tmp_path) as client:
        state, inside = client.reading_at(*FUKUOKA)

    assert state == PRESENT
    assert not inside


def test_a_missing_tile_is_reported_as_absent_not_as_clear(tmp_path: Path) -> None:
    """Whether absence means 区域外 depends on the roster, which the client
    does not hold. Deciding it here is deciding the question the roster
    exists to answer."""
    with make_client(responder(404), tmp_path) as client:
        state, inside = client.reading_at(*FUKUOKA)

    assert state == ABSENT
    assert not inside


@pytest.mark.parametrize("status", [403, 500, 503])
def test_a_server_error_is_not_a_missing_tile(tmp_path: Path, status: int) -> None:
    """Reading a 500 as absence, and absence as 区域外 inside the roster,
    would paint unlooked-at ground clear -- through two hops, which a check of
    either hop alone would miss."""
    with (
        make_client(responder(status), tmp_path) as client,
        pytest.raises(InlandFloodError, match=str(status)),
    ):
        client.reading_at(*FUKUOKA)


def test_a_body_that_is_not_an_image_is_an_error(tmp_path: Path) -> None:
    with (
        make_client(responder(200, b"<html>maintenance</html>"), tmp_path) as client,
        pytest.raises(InlandFloodError, match="not an image"),
    ):
        client.reading_at(*FUKUOKA)


# --- the cache --------------------------------------------------------------


def test_a_tile_is_fetched_once(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=painted_png())

    with make_client(handler, tmp_path) as client:
        client.reading_at(*FUKUOKA)
        client.reading_at(*FUKUOKA)

    assert calls == 1


def test_absence_is_cached_too(tmp_path: Path) -> None:
    """An unrecorded 404 is refetched every run, so the run that recorded it
    kept nothing. Many tiles at one request each is the difference between a
    resumable job and one that starts over."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with make_client(handler, tmp_path) as client:
        client.reading_at(*FUKUOKA)
        client.reading_at(*FUKUOKA)

    assert calls == 1


def test_the_fetch_time_is_recorded_for_every_tile(tmp_path: Path) -> None:
    """A tile is republished when a
    municipality revises its map, so when it was read is part of the reading."""
    with make_client(responder(200, painted_png()), tmp_path) as client:
        client.reading_at(*FUKUOKA)
        manifest = client.manifest()

    assert len(manifest) == 1
    row = manifest[0]
    assert row.status == 200
    assert row.fetched_at.tzinfo is not None


def test_a_cached_tile_keeps_the_time_it_was_fetched(tmp_path: Path) -> None:
    """Not the time it was read back. The manifest is written once per tile and
    a second run reads it rather than restamping it."""
    with make_client(responder(200, painted_png()), tmp_path) as client:
        client.reading_at(*FUKUOKA)
        first = client.manifest()[0].fetched_at

    with make_client(responder(500), tmp_path) as second:
        second.reading_at(*FUKUOKA)  # served from cache; a fetch would raise

        assert second.manifest()[0].fetched_at == first


def test_a_tile_is_decoded_once_for_its_25_points(tmp_path: Path) -> None:
    """Decoding per point lets the decoding dominate the run time, because every
    tile carries the points of many cells. The cache is a speed cache --
    it never answers anything the disk does not already hold."""
    with make_client(responder(200, painted_png()), tmp_path) as client:
        for offset in range(25):
            client.reading_at(FUKUOKA[0] + offset * 1e-6, FUKUOKA[1])

    assert client.decodes == 1


def test_a_tile_of_the_wrong_size_is_an_error(tmp_path: Path) -> None:
    """The pixel index assumes 256x256. A tile of another size would read some
    other point of the map and never say so."""
    image = Image.new("RGBA", (128, 128), (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    with (
        make_client(responder(200, buffer.getvalue()), tmp_path) as client,
        pytest.raises(InlandFloodError, match="128"),
    ):
        client.reading_at(*FUKUOKA)
