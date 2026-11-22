"""Download the OSM extract used to build the OSRM free-flow router.

Geofabrik publishes an MD5 checksum next to every extract; it is verified here
so that a truncated or interrupted download fails loudly rather than producing
a router with silently missing roads.

Usage:
    uv run python scripts/fetch_osm.py [--dest data/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from _root import add_root_argument, resolve_root, under_root

BASE_URL = "https://download.geofabrik.de/asia"
EXTRACT = "japan-latest.osm.pbf"
CHUNK = 8 * 1024 * 1024


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https host
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        next_report = 10
        with tmp.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                written += len(chunk)
                # Report sparsely: a carriage-return progress bar produces
                # thousands of log lines when captured to a file.
                if total and 100 * written / total >= next_report:
                    print(f"  {written / 1e9:.2f} / {total / 1e9:.2f} GB ({next_report}%)")
                    next_report += 10
    if total and written != total:
        tmp.unlink()
        raise RuntimeError(f"truncated download: got {written} bytes, expected {total}")
    tmp.replace(dest)


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https host
        body: bytes = response.read()
    return body.decode("utf-8")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - Geofabrik publishes md5 only
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--dest", type=Path, default=under_root("data/raw"))
    parser.add_argument("--extract", default=EXTRACT)
    args = parser.parse_args()
    resolve_root(args)

    args.dest.mkdir(parents=True, exist_ok=True)
    target = args.dest / args.extract
    url = f"{BASE_URL}/{args.extract}"

    expected = fetch_text(f"{url}.md5").split()[0]

    if target.exists() and md5_of(target) == expected:
        print(f"{target} already present and verified")
        return 0

    download(url, target)

    actual = md5_of(target)
    if actual != expected:
        target.unlink()
        raise RuntimeError(f"checksum mismatch: expected {expected}, got {actual}")

    size_gb = target.stat().st_size / 1e9
    print(f"verified {target} ({size_gb:.2f} GB, md5 {actual})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
