"""Generate, or check, the archive pin the hazard build reads.

The pin fixes what the downloaded hazard archives hold: for every archive the
manifest expects, its sha256, and for every (archive, member fragment) pair a
layer reads, either the exact `.shp` members that match the fragment or the
declaration that none does. ``scripts/build_hazard.py`` reads only the members
the pin names, and refuses an archive whose sha256 differs from it.

The archives are read the way the build reads them -- ``zipfile``, extracted
to a temporary directory, every `.shp` member read with ``geopandas`` -- so
the pin records what a build would see rather than a second reading of the
formats.

Usage:
    uv run python scripts/pin_hazard_archives.py --out <new pin>
    uv run python scripts/pin_hazard_archives.py --check --pin <pin>
    uv run python scripts/pin_hazard_archives.py --inventory <report.json> --out <new pin>

``--inventory`` takes a ``hazard-inventory/v2`` report written by the same
reader over the same archives, for a machine that holds the report but not the
archives. A pin made that way is confirmed by ``--check`` where the archives
are.

The script never overwrites a file, and writes nothing when any archive or
member could not be read, when the manifest expects a file that is not there,
or when one member matches two fragments of the same layer. The build never
generates a pin.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
from _root import add_root_argument, resolve_root, under_root

from landeval.hazard_overlay import CHUNK_SIZE, SPECS, pin_fragments, shapefile_members
from landeval.hazard_sources import (
    HazardPin,
    HazardPinError,
    HazardSources,
    load_hazard_pin,
    load_hazard_sources,
    sha256_of,
)

INVENTORY_SCHEMA = "hazard-inventory/v2"
PIN_SCHEMA = "hazard-archive-pin/v1"


@dataclass
class ArchiveScan:
    """What one archive holds, as the build's reader sees it."""

    sha256: str
    members: list[str] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _count_rows(path: Path) -> int:
    rows = 0
    while True:
        frame = gpd.read_file(path, rows=slice(rows, rows + CHUNK_SIZE))
        rows += len(frame)
        if len(frame) < CHUNK_SIZE:
            return rows


def scan_archive(path: Path, *, read_rows: bool) -> ArchiveScan:
    """The sha256 and member names of one archive, and each `.shp` member's rows."""
    scan = ArchiveScan(sha256=sha256_of(path))
    try:
        with tempfile.TemporaryDirectory() as work, zipfile.ZipFile(path) as archive:
            scan.members = [info.filename for info in archive.infolist() if not info.is_dir()]
            if not read_rows:
                return scan
            archive.extractall(work)
            for member in scan.members:
                if not member.lower().endswith(".shp"):
                    continue
                try:
                    scan.rows[member] = _count_rows(Path(work) / member)
                except Exception as error:  # recorded on the member, and the run fails
                    scan.errors.append(f"{member}: {type(error).__name__}: {error}")
    except Exception as error:  # recorded on the archive, and the run fails
        scan.errors.append(f"{type(error).__name__}: {error}")
    return scan


def scan_disk(sources: HazardSources, raw: Path, *, read_rows: bool) -> dict[str, ArchiveScan]:
    """Every archive the manifest expects that is on disk, keyed as the pin keys it."""
    scans: dict[str, ArchiveScan] = {}
    for key in sorted(expected_archives(sources)):
        path = raw / key
        if path.is_file():
            print(f"  reading {key}", file=sys.stderr)
            scans[key] = scan_archive(path, read_rows=read_rows)
    return scans


def _count_errors(node: object) -> int:
    if isinstance(node, dict):
        return int("error" in node) + sum(_count_errors(value) for value in node.values())
    if isinstance(node, list):
        return sum(_count_errors(value) for value in node)
    return 0


def scans_from_inventory(report: Mapping[str, Any]) -> tuple[dict[str, ArchiveScan], int]:
    """The archive scans a ``hazard-inventory/v2`` report records, and its error count."""
    if report.get("schema") != INVENTORY_SCHEMA:
        raise HazardPinError(
            f"inventory schema is {report.get('schema')!r}, expected {INVENTORY_SCHEMA!r}"
        )
    scans: dict[str, ArchiveScan] = {}
    for entry in report["files"]:
        if not entry.get("zip"):
            continue
        scan = ArchiveScan(sha256=str(entry["sha256"]))
        scan.members = [str(member["name"]) for member in entry.get("members", [])]
        for shp in entry.get("shp", []):
            if "error" in shp:
                scan.errors.append(f"{shp['member']}: {shp['error']}")
            else:
                scan.rows[str(shp["member"])] = int(shp["rows"])
        if "error" in entry:
            scan.errors.append(str(entry["error"]))
        scans[str(entry["path"])] = scan
    return scans, _count_errors(report["files"])


def expected_archives(sources: HazardSources) -> set[str]:
    return {
        key
        for name, layer in sources.layers.items()
        if layer.is_available
        for key in sources.archive_keys(name)
    }


@dataclass
class LayerSummary:
    members: int = 0
    absent: int = 0
    rows: int = 0


def build_pin(
    sources: HazardSources, scans: Mapping[str, ArchiveScan]
) -> tuple[dict[str, Any], dict[str, LayerSummary]]:
    """The pin for these scans, and the per-layer counts printed for review."""
    expected = expected_archives(sources)
    missing = sorted(expected - set(scans))
    if missing:
        raise HazardPinError(f"the manifest expects archives that are not there: {missing}")
    failed = {key: scans[key].errors for key in sorted(expected) if scans[key].errors}
    if failed:
        raise HazardPinError(f"archives or members could not be read: {failed}")

    layers: dict[str, Any] = {}
    summary: dict[str, LayerSummary] = {}
    for name, (source, fragments) in pin_fragments(SPECS, sources).items():
        counts = summary.setdefault(name, LayerSummary())
        entries: dict[str, dict[str, Any]] = {}
        for key in sorted(sources.archive_keys(source)):
            scan = scans[key]
            by_fragment: dict[str, Any] = {}
            owner: dict[str, str] = {}
            for fragment in fragments:
                members = shapefile_members(scan.members, fragment)
                for member in members:
                    if member in owner:
                        raise HazardPinError(
                            f"layer {name!r}: {key} member {member!r} matches both "
                            f"{owner[member]!r} and {fragment!r}"
                        )
                    owner[member] = fragment
                if members:
                    by_fragment[fragment] = {"members": members}
                    counts.members += 1
                    counts.rows += sum(scan.rows.get(member, 0) for member in members)
                else:
                    by_fragment[fragment] = {"absent": True}
                    counts.absent += 1
            entries[key] = by_fragment
        layers[name] = {"fragments": list(fragments), "entries": entries}

    pin = {
        "schema": PIN_SCHEMA,
        "archives": {key: {"sha256": scans[key].sha256} for key in sorted(expected)},
        "layers": layers,
    }
    HazardPin.model_validate(pin)
    return pin, summary


def differences(pin: HazardPin, fresh: Mapping[str, Any]) -> list[str]:
    """Each way the archives on disk differ from the pin, one line per difference."""
    found: list[str] = []
    old, new = set(pin.archives), set(fresh["archives"])
    found += [f"archive no longer expected: {key}" for key in sorted(old - new)]
    found += [f"archive not in the pin: {key}" for key in sorted(new - old)]
    for key in sorted(old & new):
        if pin.archives[key].sha256 != fresh["archives"][key]["sha256"]:
            found.append(f"sha256 changed: {key}")
    for name, layer in fresh["layers"].items():
        for key, by_fragment in layer["entries"].items():
            for fragment, entry in by_fragment.items():
                try:
                    pinned = pin.entry(name, key, fragment)
                except KeyError:
                    found.append(f"no pin entry: {name} {key} {fragment}")
                    continue
                now = tuple(entry.get("members", ()))
                if tuple(pinned.members or ()) != now:
                    found.append(f"members changed: {name} {key} {fragment}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--manifest", type=Path, default=under_root("config/hazard_sources.yaml"))
    parser.add_argument("--raw", type=Path, default=under_root("data/raw/hazard"))
    parser.add_argument("--inventory", type=Path, help="a hazard-inventory/v2 report to pin from")
    parser.add_argument("--out", type=Path, help="where to write the new pin (never overwritten)")
    parser.add_argument("--check", action="store_true", help="compare the archives with --pin")
    parser.add_argument("--pin", type=Path, help="the pin to check against")
    args = parser.parse_args()
    resolve_root(args)

    if args.check == (args.out is not None):
        parser.error("give exactly one of --out or --check")
    if args.check and (args.pin is None or args.inventory is not None):
        parser.error("--check reads the archives on disk and needs --pin")

    try:
        sources = load_hazard_sources(args.manifest)
        if args.check:
            return check(sources, args.raw, args.pin)
        return generate(sources, args)
    except HazardPinError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def generate(sources: HazardSources, args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() or out.is_symlink():
        raise HazardPinError(f"refusing to overwrite {out}; write the new pin to another file")
    if args.inventory is not None:
        report = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        scans, errors = scans_from_inventory(report)
        if errors:
            raise HazardPinError(f"the inventory records {errors} error(s); fix them first")
    else:
        scans = scan_disk(sources, args.raw, read_rows=True)
    pin, summary = build_pin(sources, scans)

    print(f"{'layer':20s} {'members':>8s} {'absent':>7s} {'rows':>12s}")
    for name, counts in summary.items():
        print(f"{name:20s} {counts.members:8d} {counts.absent:7d} {counts.rows:12,d}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(pin, handle, ensure_ascii=False, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out}: {len(pin['archives'])} archives")
    return 0


def check(sources: HazardSources, raw: Path, pin_path: Path) -> int:
    pin = load_hazard_pin(pin_path, sources=sources, fragments=pin_fragments(SPECS, sources))
    fresh, _ = build_pin(sources, scan_disk(sources, raw, read_rows=False))
    found = differences(pin, fresh)
    for line in found:
        print(f"  {line}")
    print(f"{len(found)} difference(s) between {pin_path} and the archives under {raw}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
