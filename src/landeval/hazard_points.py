"""The 100 m sample point nearest one address.

``mesh_hazard.parquet`` answers per 500 m cell: of its 25 sample points, how
many fall inside a designated zone. That is the right question for area
screening and the wrong one for a plot -- a cell at 4% and a cell at 96% both
read 「区域内」, and the plot asked about may be at either end.

``hazard_points.npz`` keeps the per-point membership the ratio was reduced
from, so the point nearest an address can be read directly. Two things it does
not pretend:

- **a lattice point is not a parcel.** The points sit 100 m apart, so the
  nearest one is up to about 70 m from the address. The distance is part of the
  reading rather than a detail for a footnote; and
- **a layer nobody read has a zero bit too.** Zero means "no polygon covered
  this point" only where the layer was read, so the index carries a coverage
  word per cell, and a layer that was not read for the cell -- or not obtained
  at all -- is named as unknown instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from landeval.mesh import mesh_center, mesh_code, mesh_size_of, mesh_step_deg

# Metres per degree of latitude. Good to a few tenths of a percent over Japan,
# and the quantity it sizes is a 100 m lattice spacing, not a route length.
_M_PER_DEG_LAT = 111_320.0


class HazardPointError(Exception):
    """Raised when the point index cannot answer for this coordinate."""


@dataclass(frozen=True)
class SamplePointReading:
    """What the lattice point nearest an address says, and how far away it is."""

    mesh_id: str
    lat: float
    lon: float
    distance_m: float
    inside: tuple[str, ...]
    unknown: tuple[str, ...]

    @property
    def is_clear(self) -> bool:
        """Whether every obtained layer put this point outside its zones.

        Not "safe", and not a statement about the unobtained ones -- those are
        in :attr:`unknown` and stay there.
        """
        return not self.inside


@dataclass(frozen=True)
class SamplePoints:
    """The per-point membership index, loaded once and queried per address."""

    membership: np.ndarray
    mesh_ids: np.ndarray
    layers: tuple[str, ...]
    available: frozenset[str]
    points_per_cell: int
    # One word per cell, one bit per layer at the bit position of `membership`:
    # set where the layer was read for that cell.
    coverage: np.ndarray

    @classmethod
    def load(cls, path: Path) -> SamplePoints:
        """The one reader of ``hazard_points.npz``.

        An index without ``coverage`` is refused rather than read with every
        layer covered or every layer unknown: it was written before the build
        recorded which cells each layer was read for, and either guess answers
        a question the file cannot. Rebuild it with ``scripts/build_hazard.py``.
        """
        with np.load(path, allow_pickle=False) as data:
            keys = ("membership", "coverage", "mesh_id", "layers", "available", "points_per_cell")
            missing = [key for key in keys if key not in data]
            if missing:
                raise HazardPointError(
                    f"{path} lacks {missing}; rebuild it with scripts/build_hazard.py"
                )
            membership = np.asarray(data["membership"])
            coverage = np.asarray(data["coverage"])
            mesh_ids = np.asarray(data["mesh_id"]).astype(str)
            layers = tuple(str(name) for name in np.asarray(data["layers"]))
            available = frozenset(str(name) for name in np.asarray(data["available"]))
            per_cell = int(data["points_per_cell"])

        unlisted = sorted(available - set(layers))
        if unlisted:
            raise HazardPointError(
                f"{path} declares {unlisted} available but has no membership bit for them; "
                "the two arrays were written by different runs"
            )
        expected = len(mesh_ids) * per_cell
        if membership.shape[0] != expected:
            raise HazardPointError(
                f"{path} holds {membership.shape[0]} points for {len(mesh_ids)} cells "
                f"of {per_cell}; {expected} were expected, so the index is truncated "
                "or was written for a different cell set"
            )
        if coverage.shape != (len(mesh_ids),):
            raise HazardPointError(
                f"{path} holds coverage of shape {coverage.shape} for {len(mesh_ids)} cells; "
                "one word per cell was expected, so the arrays were written by different runs"
            )
        return cls(
            membership=membership,
            mesh_ids=mesh_ids,
            layers=layers,
            available=available,
            points_per_cell=per_cell,
            coverage=coverage,
        )

    @property
    def cell_index(self) -> npt.NDArray[np.int64]:
        """The cell each point belongs to: the points are stored cell by cell."""
        return np.repeat(np.arange(len(self.mesh_ids), dtype=np.int64), self.points_per_cell)

    @property
    def subgrid(self) -> int:
        side = round(math.sqrt(self.points_per_cell))
        if side * side != self.points_per_cell:
            raise HazardPointError(
                f"{self.points_per_cell} points per cell is not a square lattice"
            )
        return side

    @property
    def unavailable(self) -> tuple[str, ...]:
        """Layers whose bit exists but was never filled from a source."""
        return tuple(layer for layer in self.layers if layer not in self.available)

    def _cell_position(self, mesh_id: str) -> int:
        found = np.flatnonzero(self.mesh_ids == str(mesh_id))
        if found.size == 0:
            raise HazardPointError(
                f"mesh {mesh_id} is absent from the point index; the nearest indexed cell "
                "would be a different square of ground"
            )
        if found.size > 1:
            raise HazardPointError(f"mesh {mesh_id} appears {found.size} times in the point index")
        return int(found[0])

    def reading_at(
        self,
        mesh_id: str,
        lat: float,
        lon: float,
        overlays: Sequence[LayerOverlay] = (),
    ) -> SamplePointReading:
        """The lattice point nearest ``(lat, lon)`` within ``mesh_id``.

        The caller names the cell it is asking about, and a coordinate outside
        that cell raises: the two disagreeing means one of them is wrong, and
        answering from either would answer a question nobody asked.

        ``overlays`` carry the layers whose availability is per cell rather
        than per layer -- inland flood is published by individual
        municipalities. A layer an overlay covers here stops being unknown; one
        it does not cover stays unknown, which is the whole point of keeping
        them apart from the main index.
        """
        position = self._cell_position(mesh_id)
        size_m = mesh_size_of(str(mesh_id))
        if mesh_code(lat, lon, size_m) != str(mesh_id):
            raise HazardPointError(
                f"mesh {mesh_id} does not contain ({lat}, {lon}), which falls in "
                f"{mesh_code(lat, lon, size_m)}"
            )

        centre_lat, centre_lon = mesh_center(str(mesh_id))
        lat_step, lon_step = mesh_step_deg(size_m)
        side = self.subgrid
        # The same lattice `hazard.subcell_lonlat` builds: sub-square centres,
        # latitude-major. Reproduced rather than stored because storing a
        # coordinate pair for every lattice point to re-derive a handful of
        # offsets is the expensive way round.
        offsets = (np.arange(side, dtype=np.float64) + 0.5) / side - 0.5
        row = int(np.argmin(np.abs(centre_lat + offsets * lat_step - lat)))
        column = int(np.argmin(np.abs(centre_lon + offsets * lon_step - lon)))

        point_lat = float(centre_lat + offsets[row] * lat_step)
        point_lon = float(centre_lon + offsets[column] * lon_step)
        mask = int(self.membership[position * self.points_per_cell + row * side + column])

        read = int(self.coverage[position])

        # A set bit is a real intersection whether or not the layer was read
        # for the rest of the cell. A clear bit means "outside" only where the
        # layer was read; anywhere else it is unknown.
        inside = [
            layer
            for bit, layer in enumerate(self.layers)
            if layer in self.available and mask & (1 << bit)
        ]
        unknown = [
            layer
            for bit, layer in enumerate(self.layers)
            if layer not in self.available or not (mask & (1 << bit) or read & (1 << bit))
        ]
        point = row * side + column
        for overlay in overlays:
            covered = overlay.reading_for(str(mesh_id), point)
            if covered is None:
                continue
            if overlay.layer in unknown:
                unknown.remove(overlay.layer)
            if covered and overlay.layer not in inside:
                inside.append(overlay.layer)
        return SamplePointReading(
            mesh_id=str(mesh_id),
            lat=point_lat,
            lon=point_lon,
            distance_m=_metres_between(lat, lon, point_lat, point_lon),
            inside=tuple(inside),
            unknown=tuple(unknown),
        )


def _metres_between(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Planar distance, which is what a 100 m lattice spacing needs.

    A great-circle formula would agree to well under a metre at this scale and
    would suggest the answer is more precise than the lattice it measures.
    """
    mean_lat = math.radians((lat_a + lat_b) / 2)
    north = (lat_a - lat_b) * _M_PER_DEG_LAT
    east = (lon_a - lon_b) * _M_PER_DEG_LAT * math.cos(mean_lat)
    return math.hypot(north, east)


@dataclass(frozen=True)
class LayerOverlay:
    """One layer's per-point readings for a subset of cells.

    The main index records availability per *layer*, which is the right shape
    for the national datasets: a layer is obtained everywhere or nowhere. The
    inland-flood layer is published per municipality, so
    availability varies by cell, and folding it into the main index would make
    every uncovered cell read as 区域外 -- the false-clean answer the roster
    exists to prevent.

    So it travels separately: a cell this overlay covers has a real reading,
    and a cell it does not is unchanged, still unknown.
    """

    layer: str
    mesh_ids: np.ndarray
    membership: np.ndarray
    points_per_cell: int

    @classmethod
    def load(cls, path: Path) -> LayerOverlay:
        with np.load(path) as data:
            missing = [
                key
                for key in ("layer", "mesh_id", "membership", "points_per_cell")
                if key not in data
            ]
            if missing:
                raise HazardPointError(f"{path} lacks {missing}")
            layer = str(data["layer"])
            mesh_ids = np.asarray(data["mesh_id"]).astype(str)
            membership = np.asarray(data["membership"])
            per_cell = int(data["points_per_cell"])
        expected = len(mesh_ids) * per_cell
        if membership.shape[0] != expected:
            raise HazardPointError(
                f"{path} holds {membership.shape[0]} points for {len(mesh_ids)} cells "
                f"of {per_cell}; {expected} were expected"
            )
        return cls(layer=layer, mesh_ids=mesh_ids, membership=membership, points_per_cell=per_cell)

    def reading_for(self, mesh_id: str, point: int) -> bool | None:
        """Whether this point is inside the zone, or ``None`` if not covered."""
        found = np.flatnonzero(self.mesh_ids == str(mesh_id))
        if found.size == 0:
            return None
        if found.size > 1:
            raise HazardPointError(f"mesh {mesh_id} appears {found.size} times in the overlay")
        return bool(self.membership[int(found[0]) * self.points_per_cell + point])
