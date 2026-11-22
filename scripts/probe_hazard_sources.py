"""Verify that every file `config/hazard_sources.yaml` expects is actually served.

This is the gate behind the manifest, not a one-off. It downloads nothing --
it issues HEAD requests and fails when an expected file is missing, so a stale
year suffix is caught here rather than becoming a quiet hole in the hazard map.

Usage:
    uv run python scripts/probe_hazard_sources.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from _root import add_root_argument, resolve_root, under_root

from landeval.hazard_sources import HazardSources, load_hazard_sources


def probe(url: str, timeout: float) -> tuple[int, int]:
    """Return (status, content_length). Status 0 means the request itself failed."""
    request = urllib.request.Request(  # noqa: S310 - https host fixed by the manifest
        url, method="HEAD", headers={"User-Agent": "landeval/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, int(response.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as error:
        return error.code, 0
    except OSError:
        return 0, 0


def report_layer(
    sources: HazardSources, name: str, timeout: float, quiet: bool
) -> tuple[int, list[str]]:
    """Probe one layer. Returns (total bytes, list of failures)."""
    layer = sources.layers[name]
    if not layer.is_available:
        print(f"\n{name}: 取得不可 — {layer.unavailable}")
        return 0, []

    expected = sources.expected_files(name)
    print(f"\n{name}: {len(expected)} 件を確認")
    for code, reason in sorted(layer.not_designated.items()):
        print(f"  - {code} {sources.prefectures[code]}: 対象外として宣言済み（{reason}）")

    total = 0
    failures: list[str] = []
    for key, path in sorted(expected.items()):
        status, size = probe(sources.url_for(path), timeout)
        total += size
        if status != 200:
            failures.append(f"{name}/{key}: HTTP {status} for {path}")
            print(f"  FAIL {key:<10s} HTTP {status}  {path}")
        elif not quiet:
            print(f"  ok   {key:<10s} {size / 1e6:8.1f} MB")
    return total, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--manifest", type=Path, default=under_root("config/hazard_sources.yaml"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--layer", help="probe only this layer")
    parser.add_argument("--quiet", action="store_true", help="print only failures")
    args = parser.parse_args()
    resolve_root(args)

    sources = load_hazard_sources(args.manifest)
    names = [args.layer] if args.layer else list(sources.layers)
    unknown = set(names) - set(sources.layers)
    if unknown:
        print(f"unknown layer(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    total = 0
    failures: list[str] = []
    for name in names:
        layer_bytes, layer_failures = report_layer(sources, name, args.timeout, args.quiet)
        total += layer_bytes
        failures.extend(layer_failures)

    print(f"\n合計 {total / 1e6:.0f} MB")
    if failures:
        print(f"\n{len(failures)} 件が取得できません:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            "\nマニフェストが期待するファイルが配信されていません。"
            "未指定として扱う前に、年サフィックスと配信一覧を確認してください。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
