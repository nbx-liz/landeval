"""Record the area units of every supported 250 m row.

`tests/test_mesh.py` compares `mesh.area_units_of_row` against this file on every
run. That comparison is the cross-machine guarantee: two machines whose raw
pyproj areas differ by a few ulps still produce the same integers, and a machine
or library that does not fails loudly instead of shifting a verdict.

Regenerate only when that test fails for a recorded reason (a
pyproj/PROJ upgrade that moves an area), and review the diff of the new
file before committing it.

    uv run python scripts/build_q250_units.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyproj
from _root import add_root_argument, resolve_root, under_root

from landeval.mesh import SUPPORTED_250M_ROWS, area_units_of_row

FIXTURE = "tests/fixtures/mesh/q250_units.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    parser.add_argument("--out", type=Path, default=under_root(FIXTURE))
    args = parser.parse_args()
    resolve_root(args)

    units = [area_units_of_row(row) for row in SUPPORTED_250M_ROWS]
    record = {
        "first_row": SUPPORTED_250M_ROWS.start,
        "pyproj": pyproj.__version__,
        "proj": pyproj.proj_version_str,
        "units": units,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One value per line, so a regenerated file shows in review which rows moved.
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(units)} rows from {SUPPORTED_250M_ROWS.start}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
