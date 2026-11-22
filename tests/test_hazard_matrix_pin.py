"""The archive pin: what the build may read, and how a pin that cannot be trusted is refused.

A pin that is missing, malformed, or does not fit the manifest and the overlay's
fragments stops before any layer is overlaid. The generator's own refusals are here too.
Every value is synthetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from landeval.hazard_sources import HazardSources, Layer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pin_hazard_archives.py"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def sources() -> HazardSources:
    return HazardSources(
        base_url="https://example.invalid/data",
        primary_meshes=("1234",),
        prefectures={"01": "A", "02": "B"},
        layers={
            "flood": Layer(label="flood", split="primary_mesh", patterns=("f/f_{mesh}.zip",)),
            "landslide": Layer(
                label="landslide", split="prefecture", patterns=("l/A33_{pref}.zip",)
            ),
            "inland": Layer(label="inland", unavailable="not ingested"),
        },
    )


FRAGMENTS = {
    "collapse": ("flood", ("41_a", "42_a")),
    "landslide": ("landslide", ("Polygon",)),
}


def valid() -> dict[str, Any]:
    return {
        "schema": "hazard-archive-pin/v1",
        "archives": {
            "flood/f_1234.zip": {"sha256": SHA_A},
            "landslide/A33_01.zip": {"sha256": SHA_B},
            "landslide/A33_02.zip": {"sha256": SHA_C},
        },
        "layers": {
            "collapse": {
                "fragments": ["41_a", "42_a"],
                "entries": {
                    "flood/f_1234.zip": {
                        "41_a": {"absent": True},
                        "42_a": {"members": ["42_a/x.shp"]},
                    }
                },
            },
            "landslide": {
                "fragments": ["Polygon"],
                "entries": {
                    "landslide/A33_01.zip": {"Polygon": {"members": ["A33_01Polygon.shp"]}},
                    "landslide/A33_02.zip": {"Polygon": {"members": ["A33_02Polygon.shp"]}},
                },
            },
        },
    }


def load(tmp_path: Path, data: object) -> object:
    from landeval.hazard_sources import load_hazard_pin

    path = tmp_path / "pin.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return load_hazard_pin(path, sources=sources(), fragments=FRAGMENTS)


def test_a_pin_that_fits_is_read(tmp_path: Path) -> None:
    pin: Any = load(tmp_path, valid())

    assert pin.entry("collapse", "flood/f_1234.zip", "41_a").absent is True
    assert pin.entry("landslide", "landslide/A33_01.zip", "Polygon").members == (
        "A33_01Polygon.shp",
    )


def test_a_missing_pin_stops_and_is_not_regenerated(tmp_path: Path) -> None:
    from landeval.hazard_sources import HazardPinError, load_hazard_pin

    path = tmp_path / "pin.json"
    with pytest.raises(HazardPinError, match="not found"):
        load_hazard_pin(path, sources=sources(), fragments=FRAGMENTS)
    assert not path.exists()


def _set(data: dict[str, Any], dotted: str, value: object) -> dict[str, Any]:
    node: Any = data
    *parents, last = dotted.split("|")
    for key in parents:
        node = node[key]
    if value is _DELETE:
        del node[last]
    else:
        node[last] = value
    return data


_DELETE = object()
_ENTRY = "layers|landslide|entries|landslide/A33_01.zip|Polygon"

P2_CASES: dict[str, tuple[str, object]] = {
    "schema-version": ("schema", "hazard-archive-pin/v0"),
    "unknown-top-key": ("extra", 1),
    "unknown-entry-key": (_ENTRY + "|rows", 3),
    "members-and-absent": (_ENTRY, {"members": ["A33_01Polygon.shp"], "absent": True}),
    "neither": (_ENTRY, {}),
    # Decided by which keys are present, not by their values: an explicit null
    # is a key the entry carries, so it cannot stand in for the missing one.
    "null-members-and-absent": (_ENTRY, {"members": None, "absent": True}),
    "members-and-null-absent": (_ENTRY, {"members": ["A33_01Polygon.shp"], "absent": None}),
    "null-members": (_ENTRY, {"members": None}),
    "null-absent": (_ENTRY, {"absent": None}),
    "members-empty": (_ENTRY, {"members": []}),
    "absent-false": (_ENTRY, {"absent": False}),
    "bad-sha": ("archives|landslide/A33_01.zip|sha256", "xyz"),
    "unknown-fragment": ("layers|landslide|entries|landslide/A33_01.zip|Other", {"absent": True}),
    "member-under-two-fragments": (
        "layers|collapse|entries|flood/f_1234.zip|41_a",
        {"members": ["42_a/x.shp"]},
    ),
}


def test_text_that_is_not_json_stops(tmp_path: Path) -> None:
    from landeval.hazard_sources import HazardPinError

    with pytest.raises(HazardPinError, match="not valid JSON"):
        load(tmp_path, "{not json")


@pytest.mark.parametrize("case", sorted(P2_CASES))
def test_a_malformed_pin_stops(tmp_path: Path, case: str) -> None:
    from landeval.hazard_sources import HazardPinError

    where, value = P2_CASES[case]
    with pytest.raises(HazardPinError, match="does not satisfy the contract"):
        load(tmp_path, _set(valid(), where, value))


P3_CASES: dict[str, tuple[str, object, str]] = {
    "archive-not-pinned": (
        "layers|landslide|entries|landslide/A33_02.zip",
        "UNPIN",
        "not the manifest",
    ),
    "archive-not-expected": (
        "archives|landslide/A33_03.zip",
        {"sha256": SHA_A},
        "not the manifest",
    ),
    "pair-without-entry": (
        "layers|landslide|entries|landslide/A33_02.zip",
        _DELETE,
        "no pin entry",
    ),
    "fragments-changed": ("layers|landslide|fragments", ["Polygon", "Line"], "regenerate"),
    "layer-missing": ("layers|collapse", _DELETE, "covers layers"),
    "entry-for-another-layer": (
        "layers|landslide|entries|flood/f_1234.zip",
        {"Polygon": {"absent": True}},
        "another layer",
    ),
}


@pytest.mark.parametrize("case", sorted(P3_CASES))
def test_a_pin_that_does_not_fit_the_manifest_or_fragments_stops(tmp_path: Path, case: str) -> None:
    from landeval.hazard_sources import HazardPinError

    where, value, message = P3_CASES[case]
    data = valid()
    if value == "UNPIN":
        # The archive and its entry both go, so the pin is well formed and only
        # short of an archive the manifest expects.
        _set(data, where, _DELETE)
        _set(data, "archives|landslide/A33_02.zip", _DELETE)
    else:
        _set(data, where, value)
    with pytest.raises(HazardPinError, match=message):
        load(tmp_path, data)


# --- the generator ------------------------------------------------------------


def generator() -> ModuleType:
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("pin_hazard_archives_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def inventory(**overrides: Any) -> dict[str, Any]:
    """A hazard-inventory/v2 report over the three archives of ``sources()``."""
    files = [
        {
            "path": "flood/f_1234.zip",
            "sha256": SHA_A,
            "zip": True,
            "members": [{"name": "42_a/x.shp"}, {"name": "42_a/x.dbf"}],
            "shp": [{"member": "42_a/x.shp", "rows": 3}],
        },
        {
            "path": "landslide/A33_01.zip",
            "sha256": SHA_B,
            "zip": True,
            "members": [{"name": "A33_01Polygon.shp"}],
            "shp": [{"member": "A33_01Polygon.shp", "rows": 5}],
        },
        {
            "path": "landslide/A33_02.zip",
            "sha256": SHA_C,
            "zip": True,
            "members": [{"name": "A33_02Polygon.shp"}],
            "shp": [{"member": "A33_02Polygon.shp", "rows": 7}],
        },
        {"path": "roster.csv", "sha256": SHA_A, "zip": False},
    ]
    report: dict[str, Any] = {"schema": "hazard-inventory/v2", "files": files}
    report.update(overrides)
    return report


def run_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, Any], out: Path
) -> int:
    module = generator()
    monkeypatch.setattr(module, "load_hazard_sources", lambda _path: sources())
    monkeypatch.setattr(
        module,
        "pin_fragments",
        lambda _specs, _sources: FRAGMENTS,
    )
    source = tmp_path / "inventory.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    argv = ["pin_hazard_archives.py", "--root", str(tmp_path)]
    argv += ["--inventory", str(source), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    return int(module.main())


def test_the_generator_pins_members_and_declares_absences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "new_pin.json"

    assert run_generator(tmp_path, monkeypatch, inventory(), out) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == valid()


def test_the_generator_refuses_an_inventory_with_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = inventory()
    report["files"][1]["shp"][0]["error"] = "DataSourceError: broken"
    out = tmp_path / "new_pin.json"

    assert run_generator(tmp_path, monkeypatch, report, out) == 1
    assert not out.exists()


def test_the_generator_refuses_a_manifest_file_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = inventory()
    del report["files"][2]
    out = tmp_path / "new_pin.json"

    assert run_generator(tmp_path, monkeypatch, report, out) == 1
    assert not out.exists()


def test_the_generator_refuses_a_member_matching_two_fragments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = inventory()
    report["files"][0]["members"] = [{"name": "41_a_42_a/x.shp"}]
    out = tmp_path / "new_pin.json"

    assert run_generator(tmp_path, monkeypatch, report, out) == 1
    assert not out.exists()


def test_the_generator_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "new_pin.json"
    out.write_text("earlier", encoding="utf-8")

    assert run_generator(tmp_path, monkeypatch, inventory(), out) == 1
    assert out.read_text(encoding="utf-8") == "earlier"


def write_disk(raw: Path, *, landslide_02: str = "A33_02Polygon.shp") -> None:
    for key, members in (
        ("flood/f_1234.zip", ["42_a/x.shp"]),
        ("landslide/A33_01.zip", ["A33_01Polygon.shp"]),
        ("landslide/A33_02.zip", [landslide_02]),
    ):
        path = raw / key
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as writer:
            for member in members:
                writer.writestr(member, "")


def test_check_counts_the_differences_between_the_archives_and_a_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A changed archive is found by its sha256 and by the members it now holds."""
    from landeval.hazard_sources import sha256_of

    raw = tmp_path / "raw"
    write_disk(raw)
    pin = valid()
    for key in pin["archives"]:
        pin["archives"][key]["sha256"] = sha256_of(raw / key)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    module = generator()
    monkeypatch.setattr(module, "load_hazard_sources", lambda _path: sources())
    monkeypatch.setattr(module, "pin_fragments", lambda _specs, _sources: FRAGMENTS)
    argv = ["pin_hazard_archives.py", "--root", str(tmp_path), "--raw", str(raw)]
    argv += ["--check", "--pin", str(pin_path)]
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 0
    assert "0 difference(s)" in capsys.readouterr().out

    write_disk(raw, landslide_02="renamed/Polygon_b.shp")
    assert module.main() == 1
    out = capsys.readouterr().out
    assert "sha256 changed: landslide/A33_02.zip" in out
    assert "members changed: landslide landslide/A33_02.zip Polygon" in out
    assert "2 difference(s)" in out
