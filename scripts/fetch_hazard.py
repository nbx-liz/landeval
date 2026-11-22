"""Download every hazard file the manifest expects.

Nothing here decides what is safe -- it only fetches. A file that fails to
arrive stops the run, because the alternative is a hazard layer that is empty
for reasons nobody recorded, and an empty layer paints as safe.

Usage:
    uv run python scripts/fetch_hazard.py
    uv run python scripts/fetch_hazard.py --layer flood
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from _root import add_root_argument, resolve_root, under_root

from landeval.hazard_sources import HazardSources, load_hazard_sources

CHUNK = 4 * 1024 * 1024


def download(url: str, dest: Path, timeout: float) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(  # noqa: S310 - https host fixed by the manifest
        url, headers={"User-Agent": "landeval/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        expected = int(response.headers.get("Content-Length", 0))
        written = 0
        with tmp.open("wb") as handle:
            while chunk := response.read(CHUNK):
                handle.write(chunk)
                written += len(chunk)
    if expected and written != expected:
        tmp.unlink()
        raise RuntimeError(f"truncated download: got {written} bytes, expected {expected}")
    tmp.replace(dest)
    return written


def fetch_layer(
    sources: HazardSources, name: str, dest_root: Path, timeout: float, force: bool
) -> tuple[int, int, list[str]]:
    """Returns (files fetched, bytes fetched, failures)."""
    layer = sources.layers[name]
    if not layer.is_available:
        print(f"{name}: skipped -- {layer.unavailable}")
        return 0, 0, []

    expected = sources.expected_files(name)
    dest_dir = dest_root / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    fetched = 0
    total = 0
    failures: list[str] = []
    for key, path in sorted(expected.items()):
        dest = dest_dir / Path(path).name
        if dest.exists() and not force:
            continue
        try:
            size = download(sources.url_for(path), dest, timeout)
        except (OSError, RuntimeError) as error:
            failures.append(f"{name}/{key}: {error}")
            print(f"  FAIL {key}: {error}", flush=True)
            continue
        fetched += 1
        total += size
        print(f"  {name}/{key} {size / 1e6:.1f} MB", flush=True)

    missing = [key for key, path in expected.items() if not (dest_dir / Path(path).name).exists()]
    if missing:
        failures.append(f"{name}: missing after fetch: {sorted(missing)}")
    print(f"{name}: {len(expected) - len(missing)}/{len(expected)} files present")
    return fetched, total, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--manifest", type=Path, default=under_root("config/hazard_sources.yaml"))
    parser.add_argument("--dest", type=Path, default=under_root("data/raw/hazard"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--layer", help="fetch only this layer")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    resolve_root(args)

    sources = load_hazard_sources(args.manifest)
    names = [args.layer] if args.layer else list(sources.layers)
    unknown = set(names) - set(sources.layers)
    if unknown:
        print(f"unknown layer(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    fetched = 0
    total = 0
    failures: list[str] = []
    for name in names:
        count, size, layer_failures = fetch_layer(
            sources, name, args.dest, args.timeout, args.force
        )
        fetched += count
        total += size
        failures.extend(layer_failures)

    print(f"\nfetched {fetched} files, {total / 1e6:.0f} MB into {args.dest}")
    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
