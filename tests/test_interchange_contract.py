"""The interchange artefact identity contract, on a hand-made table and node set.

The table and the node set are one artefact in two files, and nothing makes two
writes atomic together. A new table beside an old node set parses perfectly and
answers with entries computed against a graph the table never saw, so the
shared build id is what makes the mismatch visible. These tests pin that
refusal on a synthetic pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from landeval.interchange import InterchangeError, InterchangeTable

BUILD_ID = "0123456789abcdef"
OTHER_BUILD_ID = "fedcba9876543210"
HEADER = "build_id,node_id,lon,lat,name,name_en,connected\n"


def table_text(build_id: str = BUILD_ID) -> str:
    rows = [
        f"{build_id},101,139.00,35.00,北IC,North IC,true\n",
        f"{build_id},102,139.10,35.10,南IC,South IC,true\n",
    ]
    return HEADER + "".join(rows)


def write_pair(
    tmp_path: Path, *, table_id: str = BUILD_ID, nodes_id: str = BUILD_ID
) -> tuple[Path, Path]:
    table = tmp_path / "interchanges.csv"
    nodes = tmp_path / "qualified_nodes.json"
    table.write_text(table_text(table_id), encoding="utf-8")
    nodes.write_text(json.dumps({"build_id": nodes_id, "nodes": [101, 102, 103]}), encoding="utf-8")
    return table, nodes


def test_a_matching_pair_loads(tmp_path: Path) -> None:
    table, nodes = write_pair(tmp_path)

    loaded = InterchangeTable.load(table, nodes)

    assert [row.name for row in loaded.interchanges] == ["北IC", "南IC"]
    assert loaded.qualified_nodes == frozenset({101, 102, 103})


def test_a_table_and_a_node_set_from_different_builds_are_refused(tmp_path: Path) -> None:
    table, nodes = write_pair(tmp_path, table_id=BUILD_ID, nodes_id=OTHER_BUILD_ID)

    with pytest.raises(InterchangeError, match="rebuild both"):
        InterchangeTable.load(table, nodes)


def test_a_node_set_without_a_build_id_is_refused(tmp_path: Path) -> None:
    """The pre-build-id shape: a bare list is not an unlabelled match, because "no
    id" and "the same id" are different claims."""
    table, nodes = write_pair(tmp_path)
    nodes.write_text(json.dumps([101, 102, 103]), encoding="utf-8")

    with pytest.raises(InterchangeError, match="carries no build id"):
        InterchangeTable.load(table, nodes)


@pytest.mark.parametrize("spelled", ["", "   ", "not-a-digest", "0123456789ABCDEF", "01234567"])
def test_a_build_id_that_is_not_a_digest_is_refused(tmp_path: Path, spelled: str) -> None:
    """Two empty strings compare equal, so an unlabelled pair would pass as one
    build if only their agreement were checked. The shape is checked as well."""
    table, nodes = write_pair(tmp_path, table_id=spelled, nodes_id=spelled)

    with pytest.raises(InterchangeError, match="build id"):
        InterchangeTable.load(table, nodes)


def test_a_table_carrying_two_build_ids_is_refused(tmp_path: Path) -> None:
    """Half of one build appended to half of another: one file, two generations."""
    table, nodes = write_pair(tmp_path)
    table.write_text(
        table_text() + f"{OTHER_BUILD_ID},103,139.20,35.20,東IC,East IC,true\n", encoding="utf-8"
    )

    with pytest.raises(InterchangeError, match="not one build"):
        InterchangeTable.load(table, nodes)


def test_a_missing_artefact_is_an_error_and_not_an_empty_table(tmp_path: Path) -> None:
    """An empty table would answer "no expressway" for every address."""
    table, nodes = write_pair(tmp_path)

    with pytest.raises(InterchangeError, match="is missing"):
        InterchangeTable.load(tmp_path / "nope.csv", nodes)
    with pytest.raises(InterchangeError, match="is missing"):
        InterchangeTable.load(table, tmp_path / "nope.json")


def test_a_table_with_no_connected_entrance_is_refused(tmp_path: Path) -> None:
    table, nodes = write_pair(tmp_path)
    table.write_text(f"{HEADER}{BUILD_ID},1,139.0,35.0,北IC,,false\n", encoding="utf-8")

    with pytest.raises(InterchangeError, match="no connected named entrance"):
        InterchangeTable.load(table, nodes)


def test_a_table_missing_a_column_is_refused(tmp_path: Path) -> None:
    table, nodes = write_pair(tmp_path)
    table.write_text("node_id,lon,lat,name\n1,139.0,35.0,北IC\n", encoding="utf-8")

    with pytest.raises(InterchangeError, match="expected"):
        InterchangeTable.load(table, nodes)
