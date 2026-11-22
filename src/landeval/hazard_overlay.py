"""Driving the hazard overlay across the downloaded source archives.

One file of 洪水浸水想定区域 can hold a very large number of polygons, so the
polygons are streamed in chunks against a spatial index built once over the
sample points. Everything the run learns is folded into a per-point accumulator
whose only states are "worst severity seen" and NaN for "no polygon covered
this point".

Which members are read is not decided here. The archive pin
(``landeval.hazard_sources.HazardPin``) lists, for every archive and member
fragment, either the exact members or that there are none, and fixes each
archive by its sha256. The failure policy is uniform and deliberate: a missing
archive, an archive that differs from its pin, a listed member that is missing
or unreadable, a null or empty geometry, or an attribute value outside the
published code list all raise. A listed member with no rows is logged and
leaves its coverage unit (a prefecture, or a primary mesh) out of the coverage
the run reports, so the cells there cannot read as outside. Rows that the
attribute filter removes in full were still read, and the unit stays covered.
"""

from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from landeval.hazard import (
    HIT_WITHOUT_DEPTH,
    LANDSLIDE_SPECIAL_CODES,
    HazardDataError,
    PointIndex,
    depth_from_range_text,
    depth_from_rank_code,
)
from landeval.hazard_sources import HazardPin, HazardSources, archive_key, sha256_of

DepthKind = Literal["rank_code", "range_text", "none"]

# Features read per chunk. Bounded so a large shapefile does not have to be
# resident at once; large enough that the per-chunk index query dominates.
CHUNK_SIZE = 50_000


@dataclass(frozen=True)
class OverlaySpec:
    """One output hazard layer and how to build it from the archives."""

    name: str
    label: str
    source: str
    member_contains: tuple[str, ...]
    depth_column: str | None = None
    depth_kind: DepthKind = "none"
    filter_column: str | None = None
    filter_values: frozenset[int] = field(default_factory=frozenset)
    # The complete code list of `filter_column`. Every value is checked against
    # it before the layer keeps `filter_values`, so a code outside the list
    # stops the run instead of reading as "not in the subset".
    filter_domain: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.filter_column is not None and not (
            self.filter_values and self.filter_values <= self.filter_domain
        ):
            raise ValueError(
                f"layer {self.name!r} filters on {self.filter_column!r}, so it needs a "
                f"code list that contains the kept codes {sorted(self.filter_values)}; "
                f"got {sorted(self.filter_domain)}"
            )


SPECS: tuple[OverlaySpec, ...] = (
    OverlaySpec(
        "flood_planned",
        "洪水浸水想定区域（計画規模）",
        "flood",
        ("10_計画規模",),
        "A31b_101",
        "rank_code",
    ),
    OverlaySpec(
        "flood_max",
        "洪水浸水想定区域（想定最大規模）",
        "flood",
        ("20_想定最大規模",),
        "A31b_201",
        "rank_code",
    ),
    # Kept separate from depth on purpose: these areas mean the building is
    # destroyed rather than wetted, which no inundation depth conveys.
    OverlaySpec(
        "flood_collapse",
        "家屋倒壊等氾濫想定区域（氾濫流・河岸侵食）",
        "flood",
        ("41_家屋倒壊等氾濫想定区域", "42_家屋倒壊等氾濫想定区域"),
        None,
        "none",
    ),
    OverlaySpec("landslide", "土砂災害警戒区域", "landslide", ("Polygon",)),
    OverlaySpec(
        "landslide_special",
        "土砂災害特別警戒区域",
        "landslide",
        ("Polygon",),
        filter_column="A33_002",
        filter_values=LANDSLIDE_SPECIAL_CODES,
        # A33_002 区域区分 (国土数値情報 CodeOfZone_A33): 1 警戒区域,
        # 2 特別警戒区域, 3 警戒区域（指定前）, 4 特別警戒区域（指定前）.
        filter_domain=frozenset({1, 2, 3, 4}),
    ),
    OverlaySpec("tsunami", "津波浸水想定", "tsunami", (".shp",), "A40_003", "range_text"),
    OverlaySpec(
        "storm_surge", "高潮浸水想定区域", "storm_surge", (".shp",), "A49_003", "range_text"
    ),
    OverlaySpec("inland_flood", "内水氾濫", "inland_flood", ()),
)


@dataclass
class LayerRun:
    """What one layer produced, including the counts that make gaps visible."""

    spec: OverlaySpec
    severity: NDArray[np.float64] | None
    features: int = 0
    hits: int = 0
    missing_depth: int = 0
    archives: int = 0
    # The coverage units (prefectures, or primary meshes) in which every member
    # the pin lists was read: it returned rows, whether or not the attribute
    # filter kept any. Empty by default, so a run that never says what it read
    # covers nothing.
    covered: frozenset[str] = frozenset()

    @property
    def available(self) -> bool:
        return self.severity is not None


def pin_fragments(
    specs: Sequence[OverlaySpec], sources: HazardSources
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Every output layer that reads archives, with its source layer and fragments.

    This is what the archive pin is generated for and checked against. A layer
    whose source is unavailable, or absent from the manifest, reads no archive
    and so has no pin entry.
    """
    return {
        spec.name: (spec.source, spec.member_contains)
        for spec in specs
        if spec.source in sources.layers and sources.layers[spec.source].is_available
    }


def shapefile_members(names: Sequence[str], fragment: str) -> list[str]:
    """The `.shp` members whose name contains ``fragment``, in archive order.

    This partial match is used only to generate the archive pin. The build
    reads the exact names the pin lists, so it never matches a name itself.
    """
    return [name for name in names if name.lower().endswith(".shp") and fragment in name]


def _shapefile_members(archive: zipfile.ZipFile, spec: OverlaySpec) -> list[str]:
    names = archive.namelist()
    matched = {
        name for fragment in spec.member_contains for name in shapefile_members(names, fragment)
    }
    return [name for name in names if name in matched]


def _severity_for(frame: gpd.GeoDataFrame, spec: OverlaySpec) -> tuple[NDArray[np.float64], int]:
    """Severity per polygon, plus the count whose depth attribute was absent.

    An absent depth still marks the polygon as hazard -- it is inside a
    designated area, only without a published depth. An unreadable depth is a
    different thing and raises, because it means the vocabulary changed.
    """
    if spec.depth_kind == "none" or spec.depth_column is None:
        return np.full(len(frame), HIT_WITHOUT_DEPTH, dtype=np.float64), 0
    if spec.depth_column not in frame.columns:
        raise HazardDataError(
            f"layer {spec.name!r} expects column {spec.depth_column!r}, "
            f"found {sorted(c for c in frame.columns if c != 'geometry')}"
        )
    convert: Callable[[object], float] = (
        depth_from_rank_code if spec.depth_kind == "rank_code" else depth_from_range_text
    )
    values = frame[spec.depth_column].to_numpy()
    severity = np.empty(len(values), dtype=np.float64)
    missing = 0
    for position, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            severity[position] = HIT_WITHOUT_DEPTH
            missing += 1
            continue
        severity[position] = convert(value)
    return severity, missing


def _apply_filter(frame: gpd.GeoDataFrame, spec: OverlaySpec) -> gpd.GeoDataFrame:
    if spec.filter_column is None:
        return frame
    if spec.filter_column not in frame.columns:
        raise HazardDataError(
            f"layer {spec.name!r} expects column {spec.filter_column!r} to filter on"
        )
    domain = spec.filter_domain
    codes = pd.to_numeric(frame[spec.filter_column], errors="coerce")
    if codes.isna().any():
        raise HazardDataError(
            f"layer {spec.name!r}: column {spec.filter_column!r} holds "
            f"{codes.isna().sum()} values that are not codes"
        )
    # Compared as numbers, so 2.0 is the code 2 but 2.5 is no code at all; no
    # value is truncated before it has been checked.
    unknown = ~codes.isin(domain)
    if unknown.any():
        raise HazardDataError(
            f"layer {spec.name!r}: column {spec.filter_column!r} holds "
            f"{int(unknown.sum())} values outside the code list {sorted(domain)}: "
            f"{sorted(set(codes[unknown].tolist()))[:5]}"
        )
    return frame[codes.isin(spec.filter_values)]


def _chunks(path: Path, chunk_size: int) -> Iterator[gpd.GeoDataFrame]:
    start = 0
    while True:
        frame = gpd.read_file(path, rows=slice(start, start + chunk_size))
        if frame.empty:
            return
        yield frame
        if len(frame) < chunk_size:
            return
        start += len(frame)


def _check_geometries(frame: gpd.GeoDataFrame, spec: OverlaySpec, member: str) -> None:
    """Refuse null or empty geometries.

    Such a row is a designated area that cannot be placed. Counting it would
    let a layer report polygons that can intersect nothing, and dropping it
    would lose the area; either way the ground under it would read as outside.
    """
    geometry = frame.geometry
    bad = geometry.isna().to_numpy(dtype=bool) | geometry.is_empty.to_numpy(dtype=bool)
    if bad.any():
        raise HazardDataError(
            f"layer {spec.name!r}: {member} holds {int(bad.sum())} null or empty "
            f"geometries among {len(frame)} rows"
        )


def verify_archive(root: Path, key: str, pin: HazardPin, spec_name: str) -> Path:
    """The archive's path, once it is known to exist and to match its pinned sha256."""
    archive_path = root / key
    if not archive_path.exists():
        raise HazardDataError(
            f"layer {spec_name!r} needs {archive_path}, which is not downloaded; "
            f"run scripts/fetch_hazard.py"
        )
    actual = sha256_of(archive_path)
    expected = pin.archives[key].sha256
    if actual != expected:
        raise HazardDataError(
            f"layer {spec_name!r}: {archive_path} has sha256 {actual}, the pin says "
            f"{expected}; the archive changed since the pin was made -- rerun "
            "scripts/pin_hazard_archives.py --check and regenerate the pin"
        )
    return archive_path


def _read_member(
    work: Path,
    member: str,
    spec: OverlaySpec,
    index: PointIndex,
    severity: NDArray[np.float64],
    run: LayerRun,
    *,
    chunk_size: int,
) -> int:
    """Fold one pinned member into ``run``; the rows read before the filter."""
    read = 0
    frames = _chunks(work / member, chunk_size)
    while True:
        try:
            frame = next(frames)
        except StopIteration:
            break
        except Exception as error:  # the reader's own failure, re-raised as a data error
            raise HazardDataError(
                f"layer {spec.name!r}: cannot read pinned member {member!r}: {error}"
            ) from error
        _check_geometries(frame, spec, member)
        read += len(frame)
        selected = _apply_filter(frame, spec)
        if selected.empty:
            continue
        values, missing = _severity_for(selected, spec)
        run.features += len(selected)
        run.missing_depth += missing
        run.hits += index.accumulate(selected.geometry.values, values, severity)
    return read


def _ingest_archive(
    archive_path: Path,
    key: str,
    spec: OverlaySpec,
    pin: HazardPin,
    index: PointIndex,
    severity: NDArray[np.float64],
    run: LayerRun,
    *,
    chunk_size: int,
    log: Callable[[str], None],
) -> bool:
    """Fold one archive's pinned members into ``run``; False when a member had no rows.

    An `absent` entry is the pin saying the published archive holds no area for
    that fragment, so it reads nothing and does not stand in the way of
    coverage. A listed member that cannot be found or read stops the run. A
    member that holds no rows leaves its unit uncovered. A member whose rows
    the attribute filter removes in full was read, so it counts as covered.
    """
    listed = [
        member
        for fragment in spec.member_contains
        for member in pin.entry(spec.name, key, fragment).members or ()
    ]
    if not listed:
        return True
    complete = True
    with tempfile.TemporaryDirectory() as work, zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        missing = [member for member in listed if member not in names]
        if missing:
            raise HazardDataError(
                f"layer {spec.name!r}: {archive_path.name} lacks the pinned member(s) {missing}"
            )
        archive.extractall(work)
        run.archives += 1
        for member in listed:
            read = _read_member(
                Path(work), member, spec, index, severity, run, chunk_size=chunk_size
            )
            if not read:
                log(f"  {spec.name}: {archive_path.name} の {member} に地物なし")
                complete = False
    return complete


def run_layer(
    spec: OverlaySpec,
    sources: HazardSources,
    root: Path,
    index: PointIndex,
    *,
    pin: HazardPin,
    chunk_size: int = CHUNK_SIZE,
    log: Callable[[str], None] = print,
) -> LayerRun:
    """Fold every pinned member of one layer into a per-point severity accumulator.

    The members read are the ones ``pin`` lists, by exact name. A missing
    archive, an archive whose sha256 differs from the pin, a listed member that
    is missing or unreadable, a null or empty geometry and an attribute value
    outside the code list all raise ``HazardDataError``.

    ``LayerRun.covered`` is the set of coverage units -- prefectures, or
    primary meshes -- in which every listed member returned rows. A unit whose
    members are all declared absent is covered: the published data holds no
    area there. A layer in which no unit holds any area is not refused; it
    reads outside everywhere it was covered.
    """
    if not sources.layers[spec.source].is_available:
        log(f"{spec.name}: 取得不可のため no_data — {sources.layers[spec.source].unavailable}")
        return LayerRun(spec, None)

    severity = index.empty_severity()
    run = LayerRun(spec, severity)
    units = sources.unit_files(spec.source)
    uncovered: set[str] = set()
    for unit, paths in sorted(units.items()):
        for path in sorted(paths):
            key = archive_key(spec.source, path)
            archive_path = verify_archive(root, key, pin, spec.name)
            complete = _ingest_archive(
                archive_path, key, spec, pin, index, severity, run, chunk_size=chunk_size, log=log
            )
            if not complete:
                uncovered.add(unit)
    run.covered = frozenset(set(units) - uncovered)

    missing_note = f" / 深さ欠損 {run.missing_depth:,}" if run.missing_depth else ""
    uncovered_note = f" / 地物の無い単位 {sorted(uncovered)}" if uncovered else ""
    log(
        f"{spec.name}: {run.archives} アーカイブ / {run.features:,} ポリゴン / "
        f"{run.hits:,} 点ヒット{missing_note}{uncovered_note}"
    )
    return run
