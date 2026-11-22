"""Synthetic archive pins for the hazard tests.

A pin is made the way ``scripts/pin_hazard_archives.py`` makes one: each
archive on disk is fixed by its sha256 and its `.shp` members are matched to
each fragment by name. An archive that is not on disk gets a placeholder digest
and absent entries, so that a test can pin a layout before it writes it. Every
value is synthetic; no real archive or region appears here.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from landeval.hazard_overlay import OverlaySpec, shapefile_members
from landeval.hazard_sources import HazardPin, HazardSources, sha256_of

PLACEHOLDER_SHA = "0" * 64


def pin_dict(
    sources: HazardSources,
    raw: Path,
    specs: Sequence[OverlaySpec],
    *,
    overrides: Mapping[tuple[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The pin for ``specs`` over whatever lies under ``raw``, as plain JSON data.

    ``overrides`` replaces single entries, keyed by (layer, archive, fragment).
    """
    overrides = overrides or {}
    archives: dict[str, dict[str, str]] = {}
    for name, layer in sources.layers.items():
        if not layer.is_available:
            continue
        for key in sorted(sources.archive_keys(name)):
            path = raw / key
            archives[key] = {"sha256": sha256_of(path) if path.is_file() else PLACEHOLDER_SHA}
    layers: dict[str, Any] = {}
    for spec in specs:
        if not sources.layers[spec.source].is_available:
            continue
        entries: dict[str, dict[str, Any]] = {}
        for key in sorted(sources.archive_keys(spec.source)):
            path = raw / key
            names: list[str] = []
            if path.is_file():
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
            by_fragment: dict[str, Any] = {}
            for fragment in spec.member_contains:
                members = shapefile_members(names, fragment)
                entry: dict[str, Any] = {"members": members} if members else {"absent": True}
                by_fragment[fragment] = overrides.get((spec.name, key, fragment), entry)
            entries[key] = by_fragment
        layers[spec.name] = {"fragments": list(spec.member_contains), "entries": entries}
    return {"schema": "hazard-archive-pin/v1", "archives": archives, "layers": layers}


def make_pin(
    sources: HazardSources,
    raw: Path,
    specs: Sequence[OverlaySpec],
    *,
    overrides: Mapping[tuple[str, str, str], dict[str, Any]] | None = None,
) -> HazardPin:
    return HazardPin.model_validate(pin_dict(sources, raw, specs, overrides=overrides))


def write_pin(path: Path, data: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path
