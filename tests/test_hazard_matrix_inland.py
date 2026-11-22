"""The inland-flood layer, the declared exception, and its sinks.

Most rows are held by the existing tests, which this file does not repeat:
The roster tests of ``test_inland_flood.py`` hold a missing, empty or
unreadable roster (an error); ``test_inland_tiles.py`` holds the painted and
transparent tiles, the missing tile, the server errors, the tile that is not an
image and the tile of the wrong size. Added here: the request that gets no
answer at all, and the writer of the layer's coverage column, which also holds
the cells outside the roster.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, TracebackType

import httpx
import numpy as np
import pandas as pd
import pytest

from landeval.inland_flood import ABSENT, PRESENT, InlandFloodError, Roster, TileClient
from landeval.mesh import mesh_center, mesh_code

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_inland_flood.py"


def test_a_request_that_gets_no_answer_stops(tmp_path: Path) -> None:
    def no_answer(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = TileClient(transport=httpx.MockTransport(no_answer), cache_dir=tmp_path, delay_s=0.0)

    with pytest.raises(InlandFloodError, match="could not be fetched"):
        client.reading_at(33.59, 130.40)


def load_script() -> ModuleType:
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("fetch_inland_flood_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClient:
    """Paints every point of the roster's first cell; 404 for the rest."""

    def __init__(self, **_kwargs: object) -> None:
        self.requests = 0
        self.calls = 0

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def reading_at(self, _lat: float, _lon: float) -> tuple[str, bool]:
        self.calls += 1
        self.requests += 1
        return (PRESENT, True) if self.calls <= 25 else (ABSENT, False)


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_column: bool) -> list[str]:
    codes = [mesh_code(35.0 + 0.01 * i, 139.0, 500) for i in range(3)]
    centres = [mesh_center(code) for code in codes]
    cells = pd.DataFrame(
        {
            "mesh_id": codes,
            "lat": [c[0] for c in centres],
            "lon": [c[1] for c in centres],
            "muni_code": ["01001", "01001", "02001"],
            "muni_name": ["A", "A", "B"],
        }
    )
    cells.to_parquet(tmp_path / "cells.parquet", index=False)
    hazard = pd.DataFrame(
        {
            "mesh_id": codes,
            "inland_flood_hazard_ratio": [np.nan] * 3,
            "inland_flood_status": ["no_data"] * 3,
        }
    )
    if with_column:
        hazard["inland_flood_covered"] = False
    hazard.to_parquet(tmp_path / "mesh_hazard.parquet", index=False)
    module = load_script()
    roster = Roster(codes=frozenset({"01001"}), names={"01001": "A"}, rows=1)
    monkeypatch.setattr(module, "load_roster", lambda _path: roster)
    monkeypatch.setattr(module, "TileClient", FakeClient)
    argv = ["fetch_inland_flood.py", "--root", str(tmp_path)]
    argv += ["--cells", str(tmp_path / "cells.parquet")]
    argv += ["--hazard", str(tmp_path / "mesh_hazard.parquet")]
    argv += ["--points-out", str(tmp_path / "inland_flood_points.npz")]
    argv += ["--cache-dir", str(tmp_path / "cache"), "--delay", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    return [str(module.main())]


def test_only_the_roster_cells_are_covered_and_the_rest_keep_no_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert setup(tmp_path, monkeypatch, with_column=True) == ["0"]
    frame = pd.read_parquet(tmp_path / "mesh_hazard.parquet")

    assert list(frame["inland_flood_covered"]) == [True, True, False]
    assert list(frame["inland_flood_status"]) == ["risk", "outside", "no_data"]
    assert frame["inland_flood_hazard_ratio"].iloc[:2].tolist() == [1.0, 0.0]
    assert np.isnan(frame["inland_flood_hazard_ratio"].iloc[2])


def test_a_table_without_the_coverage_column_is_refused_and_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mesh_hazard.parquet"

    result = setup(tmp_path, monkeypatch, with_column=False)

    assert result == ["1"]
    assert "inland_flood_covered" not in pd.read_parquet(path).columns
    assert set(pd.read_parquet(path)["inland_flood_status"]) == {"no_data"}
