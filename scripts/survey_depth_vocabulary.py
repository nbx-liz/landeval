"""Collect every distinct depth-range spelling across the tsunami and storm-surge files.

The depth ranges are free text and the spelling varies by prefecture and by
publication year. Rather than widening the parser one rejected string at a
time, this enumerates the whole population so the grammar can be closed
against it in one go (close the grammar rather than add
tolerance).

Usage:
    uv run python scripts/survey_depth_vocabulary.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import geopandas as gpd
from _root import add_root_argument, resolve_root, under_root

from landeval.hazard import HazardDataError, depth_from_range_text
from landeval.hazard_sources import load_hazard_sources

COLUMNS = {"tsunami": "A40_003", "storm_surge": "A49_003"}


def survey(archive_path: Path, column: str) -> Counter[str]:
    found: Counter[str] = Counter()
    with tempfile.TemporaryDirectory() as work, zipfile.ZipFile(archive_path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".shp")]
        archive.extractall(work)
        for member in members:
            frame = gpd.read_file(Path(work) / member, columns=[column], ignore_geometry=True)
            if column not in frame.columns:
                raise HazardDataError(f"{archive_path.name}: no column {column}")
            found.update(str(value) for value in frame[column].tolist())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--manifest", type=Path, default=under_root("config/hazard_sources.yaml"))
    parser.add_argument("--raw", type=Path, default=under_root("data/raw/hazard"))
    args = parser.parse_args()
    resolve_root(args)

    sources = load_hazard_sources(args.manifest)
    unreadable: list[str] = []

    for layer, column in COLUMNS.items():
        print(f"\n=== {layer} ({column})")
        for key, path in sorted(sources.expected_files(layer).items()):
            archive_path = args.raw / layer / Path(path).name
            counts = survey(archive_path, column)
            print(f"\n  {key} {archive_path.name}")
            for value, count in sorted(counts.items()):
                try:
                    depth = depth_from_range_text(value)
                    verdict = f"-> {depth:g} m"
                except HazardDataError:
                    verdict = "-> UNREADABLE"
                    unreadable.append(f"{layer}/{key}: {value!r}")
                print(f"    {count:9,}  {value!r:40s} {verdict}")

    if unreadable:
        print(f"\n{len(unreadable)} spelling(s) the parser cannot read:")
        for item in sorted(set(unreadable)):
            print(f"  {item}")
        return 1
    print("\nevery spelling parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
