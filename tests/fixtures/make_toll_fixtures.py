"""Generate the hand-made Routes toll payloads under ``tests/fixtures/toll_synthetic/``.

Each file mirrors the *shape* of one Google Routes answer the toll reader has to
handle: the request that was sent (so a test can say which shape it was asked
under) and the response that came back. Every coordinate, duration, distance and
price is invented; none was recorded from a live call.

The files are checked in and ``tests/test_tolls.py`` regenerates them in memory
and compares bytes, so this script is the only place their content is defined::

    uv run python tests/fixtures/make_toll_fixtures.py            # rewrite the files
    uv run python tests/fixtures/make_toll_fixtures.py --check    # exit 1 if they differ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent / "toll_synthetic"

ORIGIN = (35.2, 139.9)
DESTINATION = (35.6, 139.7)
DEPARTURE = "2030-01-04T08:00:00+09:00"

BASE_MASK = "routes.duration,routes.distanceMeters"
TOLL_MASK = f"{BASE_MASK},routes.travelAdvisory.tollInfo,routes.legs.travelAdvisory.tollInfo"

# Invented amounts, distinct from one another so a test cannot pass by reading the
# wrong file.
PRICE_NO_PASS = 4000
PRICE_WEEKDAY_PASS = 1500
PRICE_HOLIDAY_PASS = 1200


def _point(lat: float, lon: float) -> dict[str, Any]:
    return {"location": {"latLng": {"latitude": lat, "longitude": lon}}}


def _request(*, etc_unit: str | None, extra_computations: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "origin": _point(*ORIGIN),
        "destination": _point(*DESTINATION),
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "departureTime": DEPARTURE,
    }
    if etc_unit is not None:
        body["routeModifiers"] = {
            "vehicleInfo": {"emissionType": "GASOLINE"},
            "tollPasses": [etc_unit],
        }
    if extra_computations:
        body["extraComputations"] = ["TOLLS"]
    return body


def _priced_advisory(yen: int) -> dict[str, Any]:
    return {"tollInfo": {"estimatedPrice": [{"currencyCode": "JPY", "units": str(yen)}]}}


def _route(
    *, distance_m: int, duration_s: int, advisory: dict[str, Any] | None, legs: bool
) -> dict[str, Any]:
    route: dict[str, Any] = {}
    if legs:
        route["legs"] = [{"travelAdvisory": advisory} if advisory else {}]
    route["distanceMeters"] = distance_m
    route["duration"] = f"{duration_s}s"
    if advisory is not None:
        route["travelAdvisory"] = advisory
    return route


def _case(
    name: str,
    question: str,
    mask: str,
    request: dict[str, Any],
    route: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case": name,
        "question": question,
        "field_mask": mask,
        "request": request,
        "status": 200,
        "response": {"routes": [route]},
    }


def documents() -> dict[str, dict[str, Any]]:
    asked = _request(etc_unit="JP_ETC", extra_computations=True)
    asked_etc2 = _request(etc_unit="JP_ETC2", extra_computations=True)
    asked_no_pass = _request(etc_unit=None, extra_computations=True)

    def tolled(name: str, question: str, request: dict[str, Any], yen: int, seconds: int) -> Any:
        route = _route(
            duration_s=seconds, advisory=_priced_advisory(yen), distance_m=40000, legs=True
        )
        return name, _case(name, question, TOLL_MASK, request, route)

    cases = [
        tolled(
            "tolled-friday-etc",
            "a weekday amount with a pass declared",
            asked,
            PRICE_WEEKDAY_PASS,
            3000,
        ),
        tolled(
            "tolled-friday-etc2",
            "the same weekday amount under the second pass",
            asked_etc2,
            PRICE_WEEKDAY_PASS,
            3000,
        ),
        tolled(
            "tolled-friday-nopass",
            "a weekday amount with no pass declared",
            asked_no_pass,
            PRICE_NO_PASS,
            3000,
        ),
        tolled(
            "tolled-sunday-etc",
            "a holiday amount with a pass declared",
            asked,
            PRICE_HOLIDAY_PASS,
            2400,
        ),
        tolled(
            "tolled-sunday-nopass",
            "a holiday amount with no pass declared",
            asked_no_pass,
            PRICE_NO_PASS,
            2400,
        ),
        (
            "free-friday-etc",
            _case(
                "free-friday-etc",
                "what a route that owes nothing returns: absent field, or zero",
                TOLL_MASK,
                asked,
                _route(distance_m=1000, duration_s=240, advisory={}, legs=True),
            ),
        ),
        (
            "mask-without-extracomputations",
            _case(
                "mask-without-extracomputations",
                "whether asking for the toll path without TOLLS fails, or silently omits",
                TOLL_MASK,
                _request(etc_unit="JP_ETC", extra_computations=False),
                _route(duration_s=3000, advisory={}, distance_m=40000, legs=True),
            ),
        ),
        (
            "extracomputations-without-mask",
            _case(
                "extracomputations-without-mask",
                "whether requesting TOLLS without the path fails, or silently omits",
                BASE_MASK,
                asked,
                _route(duration_s=3000, advisory=None, distance_m=40000, legs=False),
            ),
        ),
        (
            "production-shape-nomodifiers",
            _case(
                "production-shape-nomodifiers",
                "whether the default shape routes identically to a tolled one",
                BASE_MASK,
                _request(etc_unit=None, extra_computations=False),
                _route(duration_s=3000, advisory=None, distance_m=40000, legs=False),
            ),
        ),
    ]
    return dict(cases)


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def expected_files() -> dict[str, str]:
    return {f"{name}.json": render(doc) for name, doc in documents().items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if a file differs")
    args = parser.parse_args(argv)

    expected = expected_files()
    if args.check:
        stale = sorted(
            name
            for name, text in expected.items()
            if not (OUTPUT_DIR / name).is_file()
            or (OUTPUT_DIR / name).read_text(encoding="utf-8") != text
        )
        if stale:
            print(f"stale or missing: {stale}", file=sys.stderr)
            return 1
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in expected.items():
        (OUTPUT_DIR / name).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
