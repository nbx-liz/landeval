"""Print the attribute schema of the downloaded hazard files.

Used once per data vintage to pin the rank code vocabularies in
``landeval.hazard``. Reading the codes out of the data beats trusting the
documentation, because an unrecognised code must map to a rank, never to safe.

Usage:
    uv run python scripts/inspect_hazard_schema.py --layer flood
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
from _root import add_root_argument, resolve_root, under_root

warnings.filterwarnings("ignore")


def readable_members(archive: zipfile.ZipFile) -> list[str]:
    keep = []
    for name in archive.namelist():
        if "META" in name or name.endswith((".txt", ".dbf", ".prj", ".shx")):
            continue
        if name.endswith((".xml", ".shp")):
            keep.append(name)
    return keep


def describe(path: Path, rows: int) -> None:
    with tempfile.TemporaryDirectory() as work, zipfile.ZipFile(path) as archive:
        members = readable_members(archive)
        archive.extractall(work)
        for member in sorted(members):
            target = Path(work) / member
            label = str(Path(member).parent) if Path(member).parent.name else member
            try:
                frame = gpd.read_file(target, rows=rows)
            except Exception as error:
                print(f"  {label}: READ ERROR {str(error)[:120]}")
                continue
            print(f"  {label}")
            print(f"    crs={frame.crs}  cols={list(frame.columns)}")
            for column in frame.columns:
                if column == "geometry":
                    continue
                values = frame[column].dropna().unique()[:6]
                print(f"    {column}: {list(values)}")


def value_counts(path: Path, member_hint: str, column: str) -> None:
    """Full distinct-value scan of one column, so no code is missed."""
    with tempfile.TemporaryDirectory() as work, zipfile.ZipFile(path) as archive:
        archive.extractall(work)
        for member in readable_members(archive):
            if member_hint not in member:
                continue
            frame = gpd.read_file(Path(work) / member)
            if column not in frame.columns:
                print(f"  {member}: no column {column}")
                continue
            print(f"  {member}: {column}")
            print(frame[column].value_counts(dropna=False).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--raw", type=Path, default=under_root("data/raw/hazard"))
    parser.add_argument("--layer", required=True)
    parser.add_argument("--file", help="substring of the archive name to inspect")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--column", help="scan all distinct values of this column")
    parser.add_argument("--member", default="", help="substring of the member name")
    args = parser.parse_args()
    resolve_root(args)

    archives = sorted((args.raw / args.layer).glob("*.zip"))
    if args.file:
        archives = [a for a in archives if args.file in a.name]
    if not archives:
        print(f"no archives under {args.raw / args.layer}", file=sys.stderr)
        return 2

    for archive in archives:
        print(f"\n=== {archive.name}")
        if args.column:
            value_counts(archive, args.member, args.column)
        else:
            describe(archive, args.rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
