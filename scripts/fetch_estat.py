"""Download e-Stat small-area (町丁・字等) boundaries for the prefectures in scope.

One file per prefecture, the same set that `fetch_admin.py` takes: the
nationwide set is every prefecture in Japan and the rest is outside every horizon
this project uses.

The archives are the 2020 census boundaries in JGD2000, published 2022-06-24.
They carry no `.cpg`, so whoever reads them has to state CP932 --
`landeval.smallarea` does, and a test pins it.

Usage:
    uv run python scripts/fetch_estat.py --dry-run
    uv run python scripts/fetch_estat.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from _root import add_config_argument, add_root_argument, load_region, resolve_root, under_root

BASE_URL = "https://www.e-stat.go.jp/gis/statmap-search/data"

# 2020 census, small area (町丁・字等), JGD2000.
SURVEY_ID = "A002005212020"

CHUNK = 4 * 1024 * 1024

# Smaller than any real prefecture's archive, which runs to megabytes. A
# "download" under this is an error page with a 200 status,
# which would otherwise be saved and then fail to open as a zip much later.
MIN_PLAUSIBLE_BYTES = 200_000


def url_for(prefecture: str) -> str:
    query = urllib.parse.urlencode(
        {
            "dlserveyId": SURVEY_ID,
            "code": prefecture,
            "coordSys": "1",
            "format": "shape",
            "downloadType": "5",
        }
    )
    return f"{BASE_URL}?{query}"


def download(url: str, dest: Path) -> int:
    """Fetch to a temporary name and rename, so an interrupted run leaves no
    half file that the next run would treat as complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "landeval/0.1"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 - fixed https host
        written = 0
        with tmp.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                written += len(chunk)
    if written < MIN_PLAUSIBLE_BYTES:
        tmp.unlink(missing_ok=True)
        raise OSError(
            f"{url} returned {written:,} bytes, which is too small to be a prefecture's "
            "boundaries; the server answered something other than the archive"
        )
    tmp.replace(dest)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    add_config_argument(parser)
    parser.add_argument("--out", type=Path, default=under_root("data/raw/estat"))
    parser.add_argument("--dry-run", action="store_true", help="list the URLs, fetch nothing")
    args = parser.parse_args()
    resolve_root(args)
    prefectures = load_region(args.config).prefectures

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    fetched = 0
    for code, name in prefectures.items():
        dest = args.out / f"r2ka{code}.zip"
        if args.dry_run:
            print(f"  {code} {name:5s} {url_for(code)}")
            continue
        if dest.is_file():
            size = dest.stat().st_size
            total += size
            print(f"  {code} {name:5s} 取得済み {size / 1e6:6.2f} MB")
            continue
        try:
            size = download(url_for(code), dest)
        except OSError as error:
            print(f"error: {code} {name}: {error}", file=sys.stderr)
            return 1
        total += size
        fetched += 1
        print(f"  {code} {name:5s} {size / 1e6:6.2f} MB", flush=True)

    if args.dry_run:
        print("\n--dry-run のため取得しない")
        return 0
    print()
    print(f"新規取得      : {fetched} / {len(prefectures)} 都県")
    print(f"合計          : {total / 1e6:.1f} MB（{args.out}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
