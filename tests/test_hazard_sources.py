"""Tests for the hazard source manifest.

The manifest exists to make a missing hazard file loud. Most of these tests are
therefore rejection tests: they pin that an omission cannot pass as an empty
layer, because an empty hazard layer reads as "safe" on the map.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from landeval.hazard_sources import HazardSourceError, load_hazard_sources


def base_manifest() -> dict[str, Any]:
    return {
        "base_url": "https://example.invalid/data",
        "primary_meshes": ["5339", "5340"],
        "prefectures": {"12": "千葉県", "11": "埼玉県"},
        "layers": {
            "flood": {
                "label": "洪水",
                "split": "primary_mesh",
                "patterns": ["A31b/A31b-25/A31b-25_10_{mesh}_GML.zip"],
            },
            "landslide": {
                "label": "土砂",
                "split": "prefecture",
                "patterns": ["A33/A33-25/A33-25_{pref}_GML.zip"],
            },
            "tsunami": {
                "label": "津波",
                "split": "prefecture",
                "files": {"12": "A40/A40-18/A40-18_12_GML.zip"},
                "not_designated": {"11": "内陸県"},
            },
            "inland_flood": {"label": "内水", "unavailable": "全国整備なし"},
        },
    }


def write(tmp_path: Path, manifest: dict[str, Any]) -> Path:
    path = tmp_path / "hazard_sources.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    return path


def load(tmp_path: Path, mutate: Any = None) -> Any:
    manifest = copy.deepcopy(base_manifest())
    if mutate is not None:
        mutate(manifest)
    return load_hazard_sources(write(tmp_path, manifest))


def test_a_valid_manifest_loads(tmp_path: Path) -> None:
    sources = load(tmp_path)

    assert sources.base_url == "https://example.invalid/data"
    assert sources.prefectures["12"] == "千葉県"


def test_a_missing_manifest_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(HazardSourceError, match="not found"):
        load_hazard_sources(tmp_path / "absent.yaml")


def test_malformed_yaml_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("layers: [unclosed", encoding="utf-8")

    with pytest.raises(HazardSourceError, match="not valid YAML"):
        load_hazard_sources(path)


def test_a_non_mapping_manifest_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- flood\n- landslide\n", encoding="utf-8")

    with pytest.raises(HazardSourceError, match="must be a mapping"):
        load_hazard_sources(path)


def test_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(HazardSourceError):
        load(tmp_path, lambda m: m.update(refresh_days=30))


def test_a_non_https_base_url_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(HazardSourceError):
        load(tmp_path, lambda m: m.update(base_url="http://example.invalid/data"))


def test_a_prefecture_missing_from_a_layer_is_an_error(tmp_path: Path) -> None:
    """The whole point of the manifest: silence is not 'not designated'."""

    def drop_saitama(manifest: dict[str, Any]) -> None:
        del manifest["layers"]["tsunami"]["not_designated"]["11"]

    with pytest.raises(HazardSourceError, match="does not say what happens"):
        load(tmp_path, drop_saitama)


def test_a_prefecture_cannot_be_both_present_and_not_designated(tmp_path: Path) -> None:
    def contradict(manifest: dict[str, Any]) -> None:
        manifest["layers"]["tsunami"]["not_designated"]["12"] = "内陸県"

    with pytest.raises(HazardSourceError, match="both present and not designated"):
        load(tmp_path, contradict)


def test_an_unknown_prefecture_reference_is_an_error(tmp_path: Path) -> None:
    def stray(manifest: dict[str, Any]) -> None:
        manifest["layers"]["tsunami"]["not_designated"]["01"] = "北海道"

    with pytest.raises(HazardSourceError, match="unknown prefecture"):
        load(tmp_path, stray)


def test_a_pattern_layer_covers_every_prefecture(tmp_path: Path) -> None:
    """A uniform pattern implies all prefectures, so nothing is left unstated."""
    sources = load(tmp_path)

    assert set(sources.expected_files("landslide")) == {"11_0", "12_0"}


def test_a_layer_cannot_be_both_unavailable_and_fetchable(tmp_path: Path) -> None:
    def both(manifest: dict[str, Any]) -> None:
        manifest["layers"]["inland_flood"]["split"] = "prefecture"

    with pytest.raises(HazardSourceError, match="unavailable but also declares sources"):
        load(tmp_path, both)


def test_a_layer_must_say_how_it_is_obtained(tmp_path: Path) -> None:
    def neither(manifest: dict[str, Any]) -> None:
        manifest["layers"]["orphan"] = {"label": "宙ぶらりん"}

    with pytest.raises(HazardSourceError, match="needs either"):
        load(tmp_path, neither)


def test_patterns_and_files_are_mutually_exclusive(tmp_path: Path) -> None:
    def both(manifest: dict[str, Any]) -> None:
        manifest["layers"]["tsunami"]["patterns"] = ["A40/{pref}.zip"]

    with pytest.raises(HazardSourceError, match="exactly one of"):
        load(tmp_path, both)


def test_a_mesh_pattern_must_reference_the_mesh(tmp_path: Path) -> None:
    def wrong(manifest: dict[str, Any]) -> None:
        manifest["layers"]["flood"]["patterns"] = ["A31b/A31b-25/A31b-25_10_GML.zip"]

    with pytest.raises(HazardSourceError, match=r"\{mesh\}"):
        load(tmp_path, wrong)


def test_a_prefecture_pattern_must_reference_the_prefecture(tmp_path: Path) -> None:
    def wrong(manifest: dict[str, Any]) -> None:
        manifest["layers"]["landslide"]["patterns"] = ["A33/A33-25/A33-25_GML.zip"]

    with pytest.raises(HazardSourceError, match=r"\{pref\}"):
        load(tmp_path, wrong)


def test_a_mesh_layer_rejects_per_prefecture_keys(tmp_path: Path) -> None:
    def mixed(manifest: dict[str, Any]) -> None:
        manifest["layers"]["flood"]["not_designated"] = {"11": "内陸県"}

    with pytest.raises(HazardSourceError, match="mesh-split"):
        load(tmp_path, mixed)


@pytest.mark.parametrize("mesh", ["533", "53390", "abcd"])
def test_a_malformed_mesh_code_is_rejected(tmp_path: Path, mesh: str) -> None:
    with pytest.raises(HazardSourceError, match="four digits"):
        load(tmp_path, lambda m: m.update(primary_meshes=[mesh]))


def test_a_malformed_prefecture_code_is_rejected(tmp_path: Path) -> None:
    def bad(manifest: dict[str, Any]) -> None:
        # Rename 12 -> 012 everywhere, so only the code format is off-contract.
        manifest["prefectures"] = {"012": "千葉県", "11": "埼玉県"}
        manifest["layers"]["tsunami"]["files"] = {"012": "A40/A40-18/A40-18_12_GML.zip"}

    with pytest.raises(HazardSourceError, match="two digits"):
        load(tmp_path, bad)


def test_mesh_files_are_the_cross_product_of_meshes_and_patterns(tmp_path: Path) -> None:
    def two_patterns(manifest: dict[str, Any]) -> None:
        manifest["layers"]["flood"]["patterns"] = [
            "A31b/A31b-25/A31b-25_10_{mesh}_GML.zip",
            "A31b/A31b-25/A31b-25_20_{mesh}_GML.zip",
        ]

    sources = load(tmp_path, two_patterns)
    expected = sources.expected_files("flood")

    assert len(expected) == 4
    assert "A31b-25_20_5340_GML.zip" in expected["5340_1"]


def test_explicit_files_are_used_verbatim(tmp_path: Path) -> None:
    sources = load(tmp_path)

    assert sources.expected_files("tsunami") == {"12": "A40/A40-18/A40-18_12_GML.zip"}


def test_an_unavailable_layer_expects_nothing(tmp_path: Path) -> None:
    sources = load(tmp_path)

    assert sources.expected_files("inland_flood") == {}
    assert not sources.layers["inland_flood"].is_available


def test_urls_join_without_a_doubled_separator(tmp_path: Path) -> None:
    sources = load(tmp_path, lambda m: m.update(base_url="https://example.invalid/data/"))

    assert sources.url_for("A33/x.zip") == "https://example.invalid/data/A33/x.zip"


def test_prefecture_files_follow_the_manifest_structure(tmp_path: Path) -> None:
    """Each prefecture maps to its own files; a not-designated one expects none."""
    sources = load(tmp_path)

    assert sources.prefecture_files("landslide") == {
        "12": ("A33/A33-25/A33-25_12_GML.zip",),
        "11": ("A33/A33-25/A33-25_11_GML.zip",),
    }
    assert sources.prefecture_files("tsunami") == {"12": ("A40/A40-18/A40-18_12_GML.zip",)}


@pytest.mark.parametrize("layer", ["flood", "inland_flood"])
def test_prefecture_files_refuse_a_layer_not_split_by_prefecture(
    tmp_path: Path, layer: str
) -> None:
    with pytest.raises(HazardSourceError, match="prefecture split"):
        load(tmp_path).prefecture_files(layer)
