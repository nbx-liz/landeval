"""Resolve destination addresses to coordinates, and report what resolved.

The matched address and the derived precision are printed alongside every
result. A geocoder that silently returns the ward centroid for an unrecognised
street number looks exactly like a correct answer, so the match must be
eyeballed before the coordinate is written into the configuration -- that check
is the point of this script, not the lookup itself.

The rules live in `landeval.geocode`; this is the terminal front for them.

Usage:
    uv run python scripts/geocode.py "<prefecture><city><town><number>" "<another address>"
    uv run python scripts/geocode.py --file addresses.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _root import add_root_argument, resolve_root, under_root

from landeval.admin import CODE_COLUMN, load_boundaries, municipality_aliases, municipality_name
from landeval.geocode import GeocodeError, geocode, resolve
from landeval.mesh import mesh_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("addresses", nargs="*")
    parser.add_argument("--file", type=Path, help="one address per line; blank and # lines ignored")
    parser.add_argument("--admin", type=Path, default=under_root("data/raw/admin"))
    args = parser.parse_args()
    resolve_root(args)

    addresses = list(args.addresses)
    if args.file:
        addresses += [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not addresses:
        parser.error("give at least one address, or --file")

    boundaries = load_boundaries(args.admin)
    # Every accepted spelling maps to the canonical name, and the canonical
    # name to the code. The geocoder drops the county, so the alias is what
    # actually reaches a 町 or 村 (see admin.municipality_aliases).
    names = {
        alias: municipality_name(row)
        for _, row in boundaries.iterrows()
        for alias in municipality_aliases(row)
    }
    codes = dict(
        zip(
            boundaries.apply(municipality_name, axis=1),
            boundaries[CODE_COLUMN].astype(str),
            strict=True,
        )
    )

    failures = 0
    for address in addresses:
        try:
            match = geocode(address)
            resolution = resolve(address, match.matched, names)
        except GeocodeError as exc:
            print(f"FAILED  {exc}")
            failures += 1
            continue
        print(f"{match.lat:.6f}, {match.lon:.6f}  mesh={mesh_code(match.lat, match.lon, 500)}")
        print(f"        query={address}  matched={match.matched!r}")
        print(
            f"        precision={resolution.precision.value}  "
            f"muni={resolution.municipality}({codes[resolution.municipality]})"
            + (f"  unresolved={resolution.residue!r}" if resolution.residue else "")
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
