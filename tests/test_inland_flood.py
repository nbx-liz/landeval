"""The roster that ingestion is conditional on.

The inland-flood tiles are per municipality, and a 404 tile cannot tell 「this
municipality does not publish」 from 「it publishes and this point is outside
the zone」. Reading 404 as 区域外 without knowing which municipalities publish
is the false-clean shape, so a roster stands in front of it:
inside the roster a 404 means outside the zone, outside the roster the cell
stays `no_data`, and with no roster nothing is ingested at all.

That makes the roster load a gate rather than a convenience, so it fails loudly
on anything it cannot read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from landeval.inland_flood import InlandFloodError, load_roster, publishes, save_points

HEADER = "sheet_id,pref_code,pref,municipality,alert_sewer,whole_area,partial_area,source,published"

# Invented rows in the published sheet's shape.
CITY_A = "01_992011,99,甲県,甲市,,○,,甲市,20251215公開"

CITY_B = "01_992020,99,甲県,乙市,,,○,乙市,20241216公開"


def write(path: Path, *rows: str) -> Path:
    path.write_text("\n".join((HEADER, *rows)) + "\n", encoding="utf-8")
    return path


def test_the_municipality_code_comes_out_of_the_sheet_id(tmp_path: Path) -> None:
    """`01_992011` carries 99201, which is 甲市. The trailing digit is the
    check digit and is not part of the code the cells are keyed by."""
    roster = load_roster(write(tmp_path / "r.csv", CITY_A))

    assert roster.codes == frozenset({"99201"})
    assert roster.names["99201"] == "甲市"


def test_two_sheets_for_one_municipality_are_one_entry(tmp_path: Path) -> None:
    """福岡市 publishes two sheets, one of them 「（天神周辺）」. Two rows for
    one code is a fact about the sheets, not about the municipality."""
    roster = load_roster(
        write(
            tmp_path / "r.csv",
            "01_401307,40,福岡県,福岡市,○,,○,福岡市,20230213公開",
            "01_401307,40,福岡県,福岡市（天神周辺）,○,,○,福岡市（天神周辺）,20250303公開",
        )
    )

    assert roster.codes == frozenset({"40130"})
    assert roster.rows == 2


def test_publishing_is_asked_by_code_not_by_name(tmp_path: Path) -> None:
    """Names are ambiguous across prefectures and for ward names; the code is what
    the cells carry."""
    roster = load_roster(write(tmp_path / "r.csv", CITY_A, CITY_B))

    assert publishes(roster, "99201")
    assert not publishes(roster, "99203")  # a code the roster does not hold


def test_a_sheet_id_that_carries_no_code_is_an_error(tmp_path: Path) -> None:
    """Skipping the row would drop a publishing municipality into the
    "does not publish" set, and that set is what turns a 404 into 区域外."""
    with pytest.raises(InlandFloodError, match="sheet_id"):
        load_roster(write(tmp_path / "r.csv", "01_ABC,99,甲県,甲市,,○,,甲市,20251215公開"))


def test_a_missing_column_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    path.write_text("sheet_id,pref\n01_992011,99\n", encoding="utf-8")

    with pytest.raises(InlandFloodError, match="municipality"):
        load_roster(path)


def test_an_empty_roster_is_an_error(tmp_path: Path) -> None:
    """Zero publishing municipalities is the claim 「該当なし」, and it would
    make every 404 read as 区域外 for nobody while looking like a successful
    load."""
    with pytest.raises(InlandFloodError, match="no municipality"):
        load_roster(write(tmp_path / "r.csv"))


def test_a_roster_that_is_not_there_is_an_error(tmp_path: Path) -> None:
    """No roster, no ingestion. Not an empty roster -- that would
    quietly mean "nobody publishes"."""
    with pytest.raises(InlandFloodError, match="does not exist"):
        load_roster(tmp_path / "absent.csv")


def test_the_saved_overlay_loads_without_pickle(tmp_path: Path) -> None:
    """Written as the object array pandas hands over, `mesh_id` could only be
    loaded with `allow_pickle=True`, and every reader loads with the default
    `allow_pickle=False`. The file would be produced, reported as written, and
    unreadable -- and an ingestion that never reads it back would say nothing."""
    path = tmp_path / "points.npz"

    save_points(
        path,
        layer="inland_flood",
        mesh_ids=["53394611", "53394612"],
        membership=np.array([1, 0, 0, 1], dtype=np.uint8),
        points_per_cell=2,
    )

    with np.load(path) as data:  # allow_pickle defaults to False
        assert list(np.asarray(data["mesh_id"]).astype(str)) == ["53394611", "53394612"]
        assert int(data["points_per_cell"]) == 2


def test_the_overlay_refuses_a_membership_of_the_wrong_length(tmp_path: Path) -> None:
    """Cells and points are two counts of the same measurement. Saving them
    inconsistent puts the disagreement in the artefact, where the reader that
    reshapes it gets a shape error with no idea which count was wrong."""
    with pytest.raises(InlandFloodError, match="membership holds"):
        save_points(
            tmp_path / "points.npz",
            layer="inland_flood",
            mesh_ids=["53394611"],
            membership=np.array([1, 0, 0], dtype=np.uint8),
            points_per_cell=2,
        )
