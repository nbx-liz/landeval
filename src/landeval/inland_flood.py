"""Which municipalities publish an inland-flood zone.

The layer has no nationwide dataset. It exists as raster tiles published by
individual municipalities, and a missing tile is ambiguous: 404 means either
「this municipality does not publish」 or 「it publishes and this point is
outside the zone」. Reading every 404 as 区域外 would paint unlooked-at ground
as clear: a false-clean result.

The rule is to put a roster in front of the ingestion:

1. inside the roster, a 404 means outside the zone;
2. outside the roster, the cell stays ``no_data``; and
3. with no roster, nothing is ingested at all -- 「推測で埋めない」.

So loading the roster is a gate, not a convenience, and everything it cannot
read is an error rather than a skipped row: a row dropped here moves a
publishing municipality into the set whose 404s are read as clear.

An empty roster is not evidence that nobody publishes. Only the roster says
who does, so a claim that no municipality in an area publishes has to be read
off it rather than assumed.
"""

from __future__ import annotations

import csv
import io
import math
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import numpy.typing as npt
from PIL import Image

from landeval.basedate import JST

# The published sheet id is `01_XXXXXY`: a fixed prefix, the five-digit local
# government code, and its check digit. The cells are keyed by the five.
_SHEET_PREFIX_SEPARATOR = "_"
_CODE_LENGTH = 5

REQUIRED_COLUMNS = ("sheet_id", "pref", "municipality")


class InlandFloodError(Exception):
    """Raised when the roster cannot be read, which stops ingestion."""


@dataclass(frozen=True)
class Roster:
    """The municipalities that publish, and how many sheets they publish."""

    codes: frozenset[str]
    names: dict[str, str]
    rows: int


def load_roster(path: Path) -> Roster:
    """Read the roster, failing on anything ambiguous.

    A missing file is distinguished from an empty one on purpose. Missing means
    the precondition is unmet and ingestion must not start; empty would be the
    positive claim that nobody publishes, which is the very claim this guards
    against.
    """
    if not path.is_file():
        raise InlandFloodError(
            f"{path} does not exist. The roster is a precondition: "
            "without it every 404 tile is ambiguous, so nothing is ingested"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise InlandFloodError(f"{path} lacks the columns {missing}; it has {columns}")
        names: dict[str, str] = {}
        rows = 0
        for line, row in enumerate(reader, start=2):
            rows += 1
            code = _code_of(row["sheet_id"], path, line)
            names.setdefault(code, (row["municipality"] or "").strip())

    if not names:
        raise InlandFloodError(
            f"{path} names no municipality. An empty roster is not "
            "「nobody publishes」. Supply a roster that lists the publishing municipalities"
        )
    return Roster(codes=frozenset(names), names=names, rows=rows)


def _code_of(sheet_id: str, path: Path, line: int) -> str:
    _, _, digits = (sheet_id or "").partition(_SHEET_PREFIX_SEPARATOR)
    code = digits[:_CODE_LENGTH]
    if len(code) != _CODE_LENGTH or not code.isdigit():
        raise InlandFloodError(
            f"{path} line {line}: sheet_id {sheet_id!r} carries no five-digit municipality "
            "code. Skipping it would move a publishing municipality into the set whose "
            "404 tiles are read as outside the zone"
        )
    return code


def publishes(roster: Roster, muni_code: str) -> bool:
    """Whether this municipality publishes an inland-flood zone.

    Asked by code rather than by name: names repeat across prefectures and
    collapse the ordinance-designated cities' wards, which is the ambiguity
    this avoids.
    """
    return str(muni_code) in roster.codes


# --- the raster tiles -------------------------------------------------------

# The layer as the portal serves it: one 256x256 RGBA PNG per tile, painted
# where the zone is drawn. Zoom 16 is the default here; a tile is at most about
# 610 m across there, so the 100 m sample lattice lands on distinct pixels.
TILE_URL = "https://disaportaldata.gsi.go.jp/raster/02_naisui_data/{z}/{x}/{y}.png"
DEFAULT_ZOOM = 16
TILE_PIXELS = 256

# A tile that exists, and one the server has no file for. Which of the two means
# 「outside the zone」 depends on the roster, so the client reports the state and
# the caller applies the rule.
PRESENT = "present"
ABSENT = "absent"

# Politeness between requests to a public government server. A regional job is
# many tiles, so this sets the wall clock rather than the cost.
DEFAULT_DELAY_S = 0.3

# Decoded tiles held in memory at once. A cell's 25 points touch at most four
# tiles and consecutive cells are neighbours, so a small window catches nearly
# every repeat; 64 tiles of alpha is 4 MB.
_ALPHA_CACHE_TILES = 64

_MANIFEST_NAME = "manifest.csv"
_ABSENT_SUFFIX = ".404"


@dataclass(frozen=True)
class TileRecord:
    """One tile's fetch: which tile, what the server said, and when."""

    z: int
    x: int
    y: int
    status: int
    fetched_at: datetime


def tile_index(lat: float, lon: float, zoom: int = DEFAULT_ZOOM) -> tuple[int, int]:
    """The Web Mercator tile containing this coordinate."""
    count = 2**zoom
    x = int((lon + 180.0) / 360.0 * count)
    radians = math.radians(lat)
    y = int((1.0 - math.log(math.tan(radians) + 1.0 / math.cos(radians)) / math.pi) / 2.0 * count)
    return x, y


def pixel_index(lat: float, lon: float, zoom: int = DEFAULT_ZOOM) -> tuple[int, int]:
    """Where inside its tile this coordinate falls, in pixels."""
    count = 2**zoom
    fx = (lon + 180.0) / 360.0 * count
    radians = math.radians(lat)
    fy = (1.0 - math.log(math.tan(radians) + 1.0 / math.cos(radians)) / math.pi) / 2.0 * count
    x = int((fx - int(fx)) * TILE_PIXELS)
    y = int((fy - int(fy)) * TILE_PIXELS)
    return min(x, TILE_PIXELS - 1), min(y, TILE_PIXELS - 1)


class TileClient:
    """Reads the inland-flood tiles, caching both the tiles and their absence.

    What it will not do is decide what absence means. Inside a roster
    municipality a 404 is 「outside the zone」 and outside the roster it is
    「never published」, and those are the two readings the roster exists to
    separate -- so the state comes back and the caller applies the rule.

    Anything that is neither 200 nor 404 raises. A 500 read as absence, and
    absence read as outside, paints unlooked-at ground clear through two hops.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | str,
        transport: httpx.BaseTransport | None = None,
        zoom: int = DEFAULT_ZOOM,
        timeout: float = 30.0,
        delay_s: float = DEFAULT_DELAY_S,
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout, transport=transport, headers={"User-Agent": "landeval/0.1"}
        )
        self._cache_dir = Path(cache_dir)
        self.zoom = zoom
        self.delay_s = delay_s
        self.requests = 0
        self.decodes = 0
        self._alphas: OrderedDict[tuple[int, int], bytes | None] = OrderedDict()
        self._records: dict[tuple[int, int], TileRecord] = {}
        self._load_manifest()

    def __enter__(self) -> TileClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- cache -------------------------------------------------------------

    def _tile_path(self, x: int, y: int) -> Path:
        return self._cache_dir / str(self.zoom) / str(x) / f"{y}.png"

    def _absent_path(self, x: int, y: int) -> Path:
        return self._tile_path(x, y).with_suffix(_ABSENT_SUFFIX)

    def _manifest_path(self) -> Path:
        return self._cache_dir / _MANIFEST_NAME

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if not path.is_file():
            return
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["z"]) != self.zoom:
                    continue
                self._records[(int(row["x"]), int(row["y"]))] = TileRecord(
                    z=int(row["z"]),
                    x=int(row["x"]),
                    y=int(row["y"]),
                    status=int(row["status"]),
                    fetched_at=datetime.fromisoformat(row["fetched_at"]),
                )

    def _record(self, x: int, y: int, status: int) -> None:
        """Append one tile's fetch to the manifest, once.

        Appended rather than rewritten so that a run interrupted partway keeps
        the tiles it already paid the wall clock for.
        """
        record = TileRecord(z=self.zoom, x=x, y=y, status=status, fetched_at=datetime.now(JST))
        self._records[(x, y)] = record
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.is_file()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(("z", "x", "y", "status", "fetched_at"))
            writer.writerow(
                (record.z, record.x, record.y, record.status, record.fetched_at.isoformat())
            )

    def manifest(self) -> list[TileRecord]:
        """Every tile read in this cache, with the time it was read."""
        return sorted(self._records.values(), key=lambda record: (record.x, record.y))

    # -- fetching ----------------------------------------------------------

    def _body(self, x: int, y: int) -> bytes | None:
        """The tile's bytes, ``None`` when the server has no file for it."""
        cached = self._tile_path(x, y)
        if cached.is_file():
            return cached.read_bytes()
        if self._absent_path(x, y).is_file():
            return None

        if self.delay_s:
            time.sleep(self.delay_s)
        url = TILE_URL.format(z=self.zoom, x=x, y=y)
        self.requests += 1
        try:
            response = self._client.get(url)
        except httpx.HTTPError as error:
            raise InlandFloodError(
                f"tile {self.zoom}/{x}/{y} could not be fetched: {type(error).__name__}"
            ) from None

        if response.status_code == httpx.codes.NOT_FOUND:
            cached.parent.mkdir(parents=True, exist_ok=True)
            self._absent_path(x, y).write_bytes(b"")
            self._record(x, y, response.status_code)
            return None
        if response.status_code != httpx.codes.OK:
            raise InlandFloodError(
                f"tile {self.zoom}/{x}/{y} returned HTTP {response.status_code}. "
                "Only 200 and 404 have a meaning here; treating anything else as a "
                "missing tile would read unlooked-at ground as outside the zone"
            )

        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(response.content)
        self._record(x, y, response.status_code)
        return response.content

    def _alpha(self, x: int, y: int) -> bytes | None:
        """The tile's alpha channel, decoded once and kept for its neighbours.

        A regional job reads many points off each tile, so a point-by-point
        decode does the same PNG twenty-five times over, and the decoding then
        dominates the run time. Points arrive cell by cell and a cell
        touches at most a handful of tiles, so a small window is enough --
        it is a speed cache and never a source of answers the disk does not
        already hold.
        """
        key = (x, y)
        if key in self._alphas:
            self._alphas.move_to_end(key)
            return self._alphas[key]

        body = self._body(x, y)
        if body is None:
            alpha: bytes | None = None
        else:
            try:
                image = Image.open(io.BytesIO(body)).convert("RGBA")
            except (OSError, ValueError):
                raise InlandFloodError(
                    f"tile {self.zoom}/{x}/{y} is not an image; the server answered 200 with "
                    f"{len(body)} bytes of something else"
                ) from None
            if image.size != (TILE_PIXELS, TILE_PIXELS):
                raise InlandFloodError(
                    f"tile {self.zoom}/{x}/{y} is {image.size}, not "
                    f"{(TILE_PIXELS, TILE_PIXELS)}; the pixel index would read the wrong point"
                )
            alpha = image.split()[-1].tobytes()
            self.decodes += 1

        self._alphas[key] = alpha
        if len(self._alphas) > _ALPHA_CACHE_TILES:
            self._alphas.popitem(last=False)
        return alpha

    def reading_at(self, lat: float, lon: float) -> tuple[str, bool]:
        """``(state, inside)`` for one coordinate.

        ``inside`` is true when the pixel under the coordinate is painted at
        all. Anti-aliased edges count as inside: at a 100 m sample spacing the
        error either way is smaller than the lattice, and the direction that
        keeps a point is the conservative one.
        """
        x, y = tile_index(lat, lon, self.zoom)
        alpha = self._alpha(x, y)
        if alpha is None:
            return ABSENT, False
        column, row = pixel_index(lat, lon, self.zoom)
        return PRESENT, alpha[row * TILE_PIXELS + column] > 0


def save_points(
    path: Path,
    *,
    layer: str,
    mesh_ids: Sequence[str],
    membership: npt.NDArray[np.uint8],
    points_per_cell: int,
) -> None:
    """Write the per-point overlay for one layer.

    The mesh ids are stored as fixed-width unicode rather than as the object
    array pandas hands over. An object array can only be read back with
    ``allow_pickle=True``, which none of its readers pass -- so saving one
    writes a file that its own readers raise on, and nothing says so until
    something tries to load it.
    """
    expected = len(mesh_ids) * points_per_cell
    if membership.shape[0] != expected:
        raise InlandFloodError(
            f"{len(mesh_ids)} cells at {points_per_cell} points each is {expected} "
            f"values, but membership holds {membership.shape[0]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        layer=np.array(layer),
        mesh_id=np.asarray(mesh_ids, dtype=np.str_),
        membership=membership,
        points_per_cell=np.int64(points_per_cell),
    )
