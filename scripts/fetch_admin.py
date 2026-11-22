"""Download MLIT administrative boundaries (N03) for the prefectures in scope.

Only the prefectures the configured region touches are fetched: the nationwide
file is several times the size of a few prefectures' files, and the rest of Japan
is outside every horizon the configured region uses.

Usage:
    uv run python scripts/fetch_admin.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from _root import add_config_argument, add_root_argument, load_region, resolve_root, under_root

BASE_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2024"
EDITION = "N03-20240101"

CHUNK = 4 * 1024 * 1024


def download(url: str, dest: Path) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "landeval/0.1"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https host
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        with tmp.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                written += len(chunk)
    if total and written != total:
        tmp.unlink()
        raise RuntimeError(f"truncated download: got {written} bytes, expected {total}")
    tmp.replace(dest)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    add_config_argument(parser)
    parser.add_argument("--dest", type=Path, default=under_root("data/raw/admin"))
    args = parser.parse_args()
    resolve_root(args)
    prefectures = load_region(args.config).prefectures
    args.dest.mkdir(parents=True, exist_ok=True)

    for code, name in prefectures.items():
        target = args.dest / f"{EDITION}_{code}_GML.zip"
        if target.exists():
            print(f"  {code} {name}: already present")
            continue
        size = download(f"{BASE_URL}/{EDITION}_{code}_GML.zip", target)
        print(f"  {code} {name}: {size / 1e6:.1f} MB")

    print(f"\nwrote {len(prefectures)} files to {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
