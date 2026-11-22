"""The example configuration that the repository ships must load and validate."""

from __future__ import annotations

from pathlib import Path

from landeval.config import Config, load_config

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "config" / "example.yaml"


def test_the_example_configuration_exists() -> None:
    assert EXAMPLE.is_file(), f"the example configuration {EXAMPLE} is missing"


def test_the_example_configuration_validates() -> None:
    config = load_config(EXAMPLE)

    assert isinstance(config, Config)
    name, group = config.screening_group()
    assert name == "work"
    assert config.thresholds.commute.target_schedule in group.schedule_ids()


def test_the_example_declares_its_vehicle_and_region_explicitly() -> None:
    config = load_config(EXAMPLE)

    assert config.vehicle is not None
    assert config.vehicle.toll_passes == []
    assert config.region is not None
    assert set(config.region.prefectures) == {"21", "23", "24"}
