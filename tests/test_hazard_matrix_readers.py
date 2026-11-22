"""The readers of the stored hazard artefacts, against the per-cell coverage.

The multi-layer table (``unobtained_mask``, ``union_ratio``,
``coverage_class``) over the five set states; the
refusal of a table without the coverage column; the per-layer table row
and the point aggregation over the seven cell states; and the refusal of an
index without the coverage word.

The set states, for a set of layers and one cell of n points, k of them inside
a zone of some layer of the set: all_read_covered k = n and every layer read,
unread_covered k = n and some layer not read, some_inside 0 < k < n read,
none_inside k = 0 read, unread_mixed k < n and some layer not read.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from landeval.hazard import (
    NO_DATA,
    OUTSIDE,
    HazardDataError,
    coverage_class,
    split_by_coverage,
    union_ratio,
    unobtained_mask,
)
from landeval.hazard_points import HazardPointError, SamplePoints
from landeval.mesh import mesh_center, mesh_code

N = 4
# Per cell: (points of layer a inside, a read?, points of layer b inside, b read?)
SET_STATES: dict[str, tuple[int, bool, int, bool]] = {
    "all_read_covered": (4, True, 0, True),
    "unread_covered": (4, True, 0, False),  # the mixed case: b unread, a wholly covering
    "some_inside": (2, True, 1, True),
    "none_inside": (0, True, 0, True),
    "unread_mixed": (2, True, 1, False),  # mixed: the unread layer hits one point, k = 3 < n
}
ORDER = ("all_read_covered", "unread_covered", "some_inside", "none_inside", "unread_mixed")


def table() -> pd.DataFrame:
    rows = [SET_STATES[name] for name in ORDER]
    frame = pd.DataFrame(
        {
            "a_status": ["risk" if hit else OUTSIDE for hit, _, _, _ in rows],
            "a_covered": [read for _, read, _, _ in rows],
            "b_status": [
                ("risk" if hit else OUTSIDE) if read else ("risk" if hit else NO_DATA)
                for _, _, hit, read in rows
            ],
            "b_covered": [read for _, _, _, read in rows],
        }
    )
    return frame


def membership() -> np.ndarray:
    mask = np.zeros(len(ORDER) * N, dtype=np.uint16)
    for cell, name in enumerate(ORDER):
        a_hits, _, b_hits, _ = SET_STATES[name]
        for point in range(a_hits):
            mask[cell * N + point] |= 0b01
        for point in range(b_hits):
            mask[cell * N + N - 1 - point] |= 0b10
    return mask


def ratios() -> np.ndarray:
    return union_ratio(
        membership(),
        0b11,
        np.repeat(np.arange(len(ORDER)), N),
        n_cells=len(ORDER),
        points_per_cell=N,
        unknown=unobtained_mask(table(), ["a", "b"]),
    )


def test_a_cell_is_unobtained_where_a_layer_of_the_set_was_not_read() -> None:
    mask = unobtained_mask(table(), ["a", "b"])

    assert dict(zip(ORDER, mask.tolist(), strict=True)) == {
        "all_read_covered": False,
        "unread_covered": True,
        "some_inside": False,
        "none_inside": False,
        "unread_mixed": True,
    }


def test_reads_the_coverage_column_and_not_the_status() -> None:
    """A `risk` on an unread layer is a hit, not a reading of the whole cell."""
    frame = pd.DataFrame({"a_status": ["risk", "risk"], "a_covered": [True, False]})

    assert list(unobtained_mask(frame, ["a"])) == [False, True]


def test_the_selected_ratio_over_the_five_states() -> None:
    result = dict(zip(ORDER, ratios().tolist(), strict=True))

    assert result["all_read_covered"] == 1.0
    assert result["unread_covered"] == 1.0
    assert result["some_inside"] == 0.75  # three of four points inside a or b
    assert result["none_inside"] == 0.0
    assert math.isnan(result["unread_mixed"])


def test_the_coverage_class_over_the_five_states() -> None:
    classes = [coverage_class(float(value)) for value in ratios()]

    assert dict(zip(ORDER, classes, strict=True)) == {
        "all_read_covered": "covered",
        "unread_covered": "covered",
        "some_inside": "partial",
        "none_inside": "clear",
        "unread_mixed": NO_DATA,
    }


@pytest.mark.parametrize("reader", ["unobtained_mask", "split_by_coverage"])
def test_a_table_without_the_coverage_column_is_refused(reader: str) -> None:
    frame = pd.DataFrame({"a_status": [OUTSIDE, NO_DATA]})
    call = unobtained_mask if reader == "unobtained_mask" else split_by_coverage

    with pytest.raises(HazardDataError, match="a_covered"):
        call(frame, ["a"])


@pytest.mark.parametrize("value", [None, "yes"])
def test_a_coverage_column_that_is_null_or_not_boolean_is_refused(value: object) -> None:
    """A left merge that did not match brings a null, and a null is not a yes."""
    frame = pd.DataFrame({"a_status": [OUTSIDE, OUTSIDE], "a_covered": [True, value]})

    with pytest.raises(HazardDataError, match="a_covered"):
        unobtained_mask(frame, ["a"])


def test_split_by_coverage_reads_the_column() -> None:
    """A layer read for no cell is absent even where a polygon left a `risk`."""
    frame = pd.DataFrame(
        {
            "a_status": ["risk", NO_DATA],
            "a_covered": [False, False],
            "b_status": [OUTSIDE, NO_DATA],
            "b_covered": [True, False],
        }
    )

    assert split_by_coverage(frame, ["a", "b"]) == (["b"], ["a"])


# --- the point index ------------------------------------------------------------

MESH = mesh_code(43.4, 142.5, 500)
LAT, LON = mesh_center(MESH)
# One layer per cell state; the bit is the layer's position.
STATE_LAYERS = ("c_hit", "c_clear", "u_hit", "u_clear", "d_hit", "d_clear", "x")
HIT = 0b0010101  # c_hit, u_hit, d_hit
READ = 0b0110011  # c_hit, c_clear, d_hit, d_clear
AVAILABLE = STATE_LAYERS[:-1]


def write_index(path: Path, *, coverage: np.ndarray | None, hit: int = HIT) -> Path:
    arrays: dict[str, np.ndarray] = {
        "membership": np.full(25, hit, dtype=np.uint16),
        "cell_index": np.zeros(25, dtype=np.int64),
        "mesh_id": np.array([MESH], dtype=str),
        "layers": np.array(STATE_LAYERS),
        "available": np.array(AVAILABLE),
        "points_per_cell": np.array(25, dtype=np.int64),
    }
    if coverage is not None:
        arrays["coverage"] = coverage
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return path


def test_the_point_reading_over_the_seven_states(tmp_path: Path) -> None:
    points = SamplePoints.load(
        write_index(tmp_path / "p.npz", coverage=np.array([READ], dtype=np.uint16))
    )

    reading = points.reading_at(MESH, LAT, LON)

    assert set(reading.inside) == {"c_hit", "u_hit", "d_hit"}
    assert set(reading.unknown) == {"u_clear", "x"}


def test_a_hit_and_a_clear_read_layer_give_one_inside_and_not_clear(tmp_path: Path) -> None:
    """Mixed layers at one point; an unread layer adds to `unknown` and nothing else."""
    only_read = write_index(
        tmp_path / "a.npz", coverage=np.array([0b0000011], dtype=np.uint16), hit=0b0000001
    )
    reading = SamplePoints.load(only_read).reading_at(MESH, LAT, LON)
    assert "c_hit" in reading.inside and "c_clear" not in reading.inside
    assert not reading.is_clear

    clear = write_index(
        tmp_path / "b.npz", coverage=np.array([0b0000010], dtype=np.uint16), hit=0b0000000
    )
    reading = SamplePoints.load(clear).reading_at(MESH, LAT, LON)
    assert reading.inside == ()
    assert reading.is_clear
    assert "u_clear" in reading.unknown


def test_an_index_without_the_coverage_word_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HazardPointError, match="coverage"):
        SamplePoints.load(write_index(tmp_path / "old.npz", coverage=None))


def test_an_index_whose_coverage_does_not_match_its_cells_is_refused(tmp_path: Path) -> None:
    path = write_index(tmp_path / "bad.npz", coverage=np.zeros(2, dtype=np.uint16))

    with pytest.raises(HazardPointError, match="coverage"):
        SamplePoints.load(path)
