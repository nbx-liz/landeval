"""Overlay the published hazard areas onto every mesh cell.

Runs over the whole working set, not just the cells transport kept, because the
result is free to compute and stays valid when a transport threshold moves
(the same reasoning as for travel times, applied to hazard). Makes no API calls.

Before any layer is overlaid, the archive pin is read and checked against the
manifest and the overlay's member fragments, and every pinned archive must be
on disk with its pinned sha256. Any failure stops the build before anything is
written, so an earlier ``mesh_hazard.parquet`` and ``hazard_points.npz`` stay
as they were.

Usage:
    uv run python scripts/build_hazard.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from _root import add_root_argument, resolve_root, under_root
from numpy.typing import NDArray

from landeval.admin import attach_municipality, load_boundaries
from landeval.hazard import (
    NO_DATA,
    NOT_DESIGNATED,
    OUTSIDE,
    SUBGRID,
    HazardDataError,
    PointIndex,
    covered_column,
    layer_status,
    subcell_lonlat,
    summarise_layer,
)
from landeval.hazard_overlay import SPECS, pin_fragments, run_layer, verify_archive
from landeval.hazard_sources import (
    HazardPin,
    HazardPinError,
    HazardSources,
    Layer,
    load_hazard_pin,
    load_hazard_sources,
)
from landeval.mesh import area_km2, primary_mesh_of

# The state of one cell for one layer: covered, not covered, not designated
# (decided, and so covered), or the layer could not be obtained at all.
COVERED = "C"
UNCOVERED = "U"
UNDESIGNATED = "D"
UNOBTAINED = "X"


def cell_states(
    pref: pd.Series,
    source: Layer,
    covered: set[str],
    *,
    primary_mesh: pd.Series | None = None,
) -> NDArray[np.object_]:
    """The coverage state of every cell for one layer.

    ``covered`` is ``LayerRun.covered``: the prefectures, or the primary
    meshes, in which every pinned member was read. A cell whose unit is not in
    it -- unknown, outside the manifest, or with a member that held no rows --
    is not covered. A cell of a prefecture declared not designated is decided,
    whatever ``covered`` says.
    """
    n = len(pref)
    if not source.is_available:
        return np.full(n, UNOBTAINED, dtype=object)
    state = np.full(n, UNCOVERED, dtype=object)
    if source.split == "primary_mesh":
        if primary_mesh is None:
            raise HazardDataError("a mesh-split layer needs each cell's primary mesh")
        state[primary_mesh.isin(covered).to_numpy(dtype=bool)] = COVERED
        return state
    state[pref.isin(covered).fillna(False).to_numpy(dtype=bool)] = COVERED
    if source.not_designated:
        undeclared = pref.isin(set(source.not_designated)).fillna(False).to_numpy(dtype=bool)
        state[undeclared] = UNDESIGNATED
    return state


def cell_statuses(
    ratio: NDArray[np.float64],
    depth: NDArray[np.float64],
    pref: pd.Series,
    source: Layer,
    covered: set[str],
    *,
    primary_mesh: pd.Series | None = None,
) -> NDArray[np.object_]:
    """The status of every cell for one obtained layer.

    A risk status always comes from a real polygon intersection, so it is kept
    whatever the cell's coverage. Only the absence of a hit depends on it: a
    covered cell reads `outside`, a cell of a prefecture declared not
    designated reads `not_designated`, and a cell that is not covered reads
    `no_data`. Prefecture-split and mesh-split layers follow the same rule;
    see :func:`cell_states`.
    """
    state = cell_states(pref, source, covered, primary_mesh=primary_mesh)
    status = np.array(
        [
            layer_status(ratio[i], depth[i], available=True, designated=True)
            for i in range(len(ratio))
        ],
        dtype=object,
    )
    clear = status == OUTSIDE
    status[clear & (state == UNDESIGNATED)] = NOT_DESIGNATED
    status[clear & (state == UNCOVERED)] = NO_DATA
    status[state == UNOBTAINED] = NO_DATA
    return status


def check_pinned_archives(
    pin: HazardPin,
    sources: HazardSources,
    root: Path,
    fragments: Mapping[str, tuple[str, tuple[str, ...]]],
) -> None:
    """Every archive every layer reads is on disk and matches its pinned sha256."""
    for name, (source, _) in fragments.items():
        for key in sorted(sources.archive_keys(source)):
            verify_archive(root, key, pin, name)


def print_bands(frame: pd.DataFrame, obtained: Sequence[str]) -> None:
    """The hazard-free ratio distribution, without the cells nobody could judge.

    ``hazard_free_ratio`` counts only the layers that were read, so a cell in
    which one of them is not covered has no band: it is counted on its own row.
    """
    unjudged = np.zeros(len(frame), dtype=bool)
    for name in obtained:
        unjudged |= ~frame[covered_column(name)].to_numpy(dtype=bool)
    ratios = frame["hazard_free_ratio"]
    print("\nハザード指定区域外の面積率の分布")
    for low, high in ((1.0, 1.01), (0.9, 1.0), (0.5, 0.9), (0.1, 0.5), (0.0, 0.1), (-0.01, 0.0)):
        band = frame[(ratios >= low) & (ratios < high) & ~unjudged]
        label = "1.00" if low == 1.0 else f"{low:.2f}–{high:.2f}"
        print(f"  {label:>12s}  {len(band):8,} セル  {area_km2(band['mesh_id']):8,.0f} km²")
    judged_nan = int((ratios.isna() & ~unjudged).sum())
    print(f"  {'欠測':>12s}  {judged_nan:8,} セル")
    print(
        f"  {'判定不能':>10s}  {int(unjudged.sum()):8,} セル  "
        f"{area_km2(frame.loc[unjudged, 'mesh_id']):8,.0f} km²（被覆の無いレイヤがある）"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--manifest", type=Path, default=under_root("config/hazard_sources.yaml"))
    parser.add_argument("--pin", type=Path, default=under_root("config/hazard_archive_pin.json"))
    parser.add_argument("--raw", type=Path, default=under_root("data/raw/hazard"))
    parser.add_argument("--admin", type=Path, default=under_root("data/raw/admin"))
    parser.add_argument(
        "--cells", type=Path, default=under_root("data/processed/freeflow_horizon.parquet")
    )
    parser.add_argument(
        "--out", type=Path, default=under_root("data/processed/mesh_hazard.parquet")
    )
    parser.add_argument("--mesh-size-m", type=int, default=500)
    parser.add_argument("--subgrid", type=int, default=SUBGRID)
    parser.add_argument("--layer", action="append", help="restrict to these layers")
    args = parser.parse_args()
    resolve_root(args)
    try:
        return build(args)
    except (HazardPinError, HazardDataError) as error:
        print(f"error: {error}", file=sys.stderr)
        print("nothing was written", file=sys.stderr)
        return 1


def build(args: argparse.Namespace) -> int:
    sources = load_hazard_sources(args.manifest)
    # Every check that can refuse the run happens here, before any layer is
    # overlaid: the pin against the manifest and the fragments of every layer,
    # then every pinned archive against its sha256.
    fragments = pin_fragments(SPECS, sources)
    pin = load_hazard_pin(args.pin, sources=sources, fragments=fragments)
    check_pinned_archives(pin, sources, args.raw, fragments)

    cells = pd.read_parquet(args.cells, columns=["mesh_id", "lat", "lon"])
    print(f"{len(cells):,} セル / サブグリッド {args.subgrid}x{args.subgrid}")

    print("市区町村を突き合わせています…")
    located = attach_municipality(cells, load_boundaries(args.admin))
    # Everything below indexes by position against `cells`, so the join result
    # is realigned rather than assumed to have come back in the same order.
    located = located.reindex(cells.index)
    pref = located["muni_code"].astype("string").str[:2]
    primary_mesh = pd.Series(
        [primary_mesh_of(str(code)) for code in cells["mesh_id"]], index=cells.index
    )
    unknown_pref = int(pref.isna().sum())
    print(f"  都道府県不明 {unknown_pref:,} セル（海上等）")

    lon, lat, cell_index = subcell_lonlat(cells, mesh_size_m=args.mesh_size_m, subgrid=args.subgrid)
    points_per_cell = args.subgrid**2
    print(f"{len(lon):,} 標本点の索引を構築しています…")
    index = PointIndex(lon, lat)

    specs = [s for s in SPECS if not args.layer or s.name in args.layer]
    columns: dict[str, object] = {"mesh_id": cells["mesh_id"].to_numpy()}
    any_hazard = np.zeros(len(lon), dtype=bool)
    available_layers: list[str] = []
    unavailable_layers: list[str] = []

    # One bit per layer, per sample point. Kept because a union over a subset
    # of layers cannot be reconstructed from the per-layer fractions, and which
    # layers to apply is a question the reader answers, not this script.
    # `coverage` holds the same bit per cell: set where the layer was read.
    if len(specs) > 16:
        raise SystemExit(f"{len(specs)} layers exceeds the 16 bits of the membership mask")
    membership = np.zeros(len(lon), dtype=np.uint16)
    coverage = np.zeros(len(cells), dtype=np.uint16)

    for position, spec in enumerate(specs):
        run = run_layer(spec, sources, args.raw, index, pin=pin)
        source = sources.layers[spec.source]

        if run.severity is None:
            columns[f"{spec.name}_hazard_ratio"] = np.full(len(cells), np.nan)
            columns[f"{spec.name}_max_depth_m"] = np.full(len(cells), np.nan)
            columns[f"{spec.name}_status"] = np.full(len(cells), NO_DATA, dtype=object)
            columns[covered_column(spec.name)] = np.zeros(len(cells), dtype=bool)
            unavailable_layers.append(spec.name)
            continue

        available_layers.append(spec.name)
        inside = np.isfinite(run.severity)
        any_hazard |= inside
        membership |= inside.astype(np.uint16) << np.uint16(position)
        summary = summarise_layer(
            cell_index, run.severity, n_cells=len(cells), points_per_cell=points_per_cell
        )

        ratio = summary["hazard_ratio"].to_numpy()
        depth = summary["max_depth_m"].to_numpy()
        columns[f"{spec.name}_hazard_ratio"] = ratio
        columns[f"{spec.name}_max_depth_m"] = depth

        covered = set(run.covered)
        columns[f"{spec.name}_status"] = cell_statuses(
            ratio, depth, pref, source, covered, primary_mesh=primary_mesh
        )
        state = cell_states(pref, source, covered, primary_mesh=primary_mesh)
        read = (state == COVERED) | (state == UNDESIGNATED)
        columns[covered_column(spec.name)] = read
        coverage |= read.astype(np.uint16) << np.uint16(position)

    free = summarise_layer(
        cell_index,
        np.where(any_hazard, 0.0, np.nan),
        n_cells=len(cells),
        points_per_cell=points_per_cell,
    )["hazard_ratio"].to_numpy()
    columns["hazard_free_ratio"] = 1.0 - free
    columns["fetched_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    frame = pd.DataFrame(columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    points_out = args.out.with_name("hazard_points.npz")
    np.savez_compressed(
        points_out,
        membership=membership,
        coverage=coverage,
        cell_index=cell_index,
        mesh_id=cells["mesh_id"].to_numpy().astype(str),
        layers=np.array([s.name for s in specs]),
        available=np.array(available_layers),
        points_per_cell=np.int64(points_per_cell),
    )
    print(f"wrote {points_out} ({points_out.stat().st_size / 1e6:.1f} MB)")

    print("\n=== 集計 ===")
    print(f"重畳したレイヤ: {', '.join(available_layers)}")
    if unavailable_layers:
        print(f"取得不可（全セル no_data）: {', '.join(unavailable_layers)}")
    for spec in specs:
        column = f"{spec.name}_status"
        if column not in frame:
            continue
        counts = frame[column].value_counts()
        parts = [f"{name} {int(n):,}" for name, n in counts.items()]
        print(f"  {spec.label}: {' / '.join(parts)}")

    print_bands(frame, available_layers)

    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
