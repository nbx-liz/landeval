"""One-off check: can GDAL read the JPGIS GML the hazard archives ship?

Kept because the answer decides which download format the manifest pins, and a
future data vintage may change it.

The archive is read as given, relative to the current directory. The data root is
still required, so that every script that reads data states where its
data lives.

Usage:
    uv run python scripts/probe_gml_read.py --root . <archive.zip>
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
from _root import add_root_argument, resolve_root

warnings.filterwarnings("ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("archive", type=Path, help="the hazard archive (.zip) to probe")
    args = parser.parse_args()
    resolve_root(args)

    archive_path: Path = args.archive
    with tempfile.TemporaryDirectory() as work, zipfile.ZipFile(archive_path) as archive:
        archive.extractall(work)
        for path in sorted(Path(work).rglob("*")):
            if not path.is_file() or "META" in path.name:
                continue
            print(f"\n{path.relative_to(work)}  ({path.stat().st_size / 1e6:.1f} MB)")
            try:
                frame = gpd.read_file(path, rows=2)
            except Exception as error:
                print(f"  read failed: {type(error).__name__}: {str(error)[:160]}")
                continue
            print(f"  ok: {len(frame)} rows, crs={frame.crs}")
            print(f"  columns: {list(frame.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
