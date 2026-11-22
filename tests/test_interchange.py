"""The interchange module's name rules and construction guards, on synthetic input.

Nothing here reads a recorded extract: every name is invented, written to have the
same *shape* as the three spelling families the classifier has to recognise. The
table-and-node-set identity checks, on a synthetic pair, are in
`test_interchange_contract.py`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from landeval.interchange import (
    Interchange,
    InterchangeError,
    RouteNodes,
)

ROOT = Path(__file__).resolve().parents[1]

# Invented names, one per spelling family, with whether the obvious suffix rule
# `endswith("スマートIC")` would have seen each of them. The family decides the
# result, not the place name, so none of these is a real place.
#
# Two of the six the suffix rule does find. They are kept because they defeat
# other plausible matchers instead: a separator rule that expects the name to be
# one token, and a full-width/half-width assumption in the same position.
SMART_SPELLINGS = [
    ("甲PA（上り）、スマートＩＣ", "", "katakana", False),
    ("乙PA（上り）;乙スマートIC", "", "katakana", True),
    ("丙PA(下り),スマートIC", "", "katakana", True),
    ("丁SIC", "", "abbreviation", False),
    ("戊PA/SIC（上り）", "", "abbreviation", False),
    ("己", "Ki Smart", "english-only", False),
]
SUFFIX = "スマートIC"


@pytest.mark.parametrize(("name", "name_en", "family", "_suffix"), SMART_SPELLINGS)
def test_every_measured_smart_spelling_is_recognised(
    name: str, name_en: str, family: str, _suffix: bool
) -> None:
    """Three disjoint families and no tag, so the name is all there is. The
    english-only one carries nothing in Japanese at all and is only visible
    through its English name."""
    assert Interchange(1, name, name_en, 0.0, 0.0).is_smart, family


@pytest.mark.parametrize(("name", "name_en", "family", "suffix"), SMART_SPELLINGS)
def test_the_suffix_rule_is_measured_rather_than_assumed(
    name: str, name_en: str, family: str, suffix: bool
) -> None:
    """What the rejected implementation actually does on each of these, recorded
    so that "a suffix matcher would fail here" is a measurement and not a claim.
    Four of the six defeat it; two do not, and saying so is the point."""
    assert name.endswith(SUFFIX) is suffix


def test_the_suffix_rule_would_miss_a_case_no_other_rule_covers_for_it() -> None:
    """The summary of the rows above: dropping to the suffix rule loses cases the
    katakana, abbreviation and English families each contribute."""
    missed = {family for _, _, family, suffix in SMART_SPELLINGS if not suffix}
    assert missed == {"katakana", "abbreviation", "english-only"}


@pytest.mark.parametrize("name", ["北東IC", "中央IC", "西JCT", "南IC"])
def test_an_ordinary_interchange_is_not_called_smart(name: str) -> None:
    assert not Interchange(1, name, "", 0.0, 0.0).is_smart


@pytest.mark.parametrize(
    "name", ["北JCT", "北ＪＣＴ", "東ＰＡ（上り）", "南SA（下り）", "西ジャンクション"]
)
def test_a_junction_or_a_rest_area_alone_is_not_an_entrance(name: str) -> None:
    assert not Interchange(1, name, "", 0.0, 0.0).is_entrance


@pytest.mark.parametrize(
    "name", ["北IC", "東ＩＣ", "南インター", "第一入口", "中央PA/SIC", "西SA/スマートIC"]
)
def test_a_name_that_says_where_a_car_can_join_is_an_entrance(name: str) -> None:
    """A name with an interchange token is an entrance even when it also names a
    rest area, and so is a name with no token at all (an urban entrance)."""
    assert Interchange(1, name, "", 0.0, 0.0).is_entrance


def test_a_route_whose_arrays_do_not_line_up_is_refused() -> None:
    """The clock is indexed by node. If the two are not the same length then the
    minute figure belongs to some other point on the road."""
    with pytest.raises(InterchangeError, match="not index-aligned"):
        RouteNodes(nodes=(1, 2), coordinates=((35.0, 139.0),), cumulative_s=(0.0, 1.0))
    with pytest.raises(InterchangeError, match="no nodes"):
        RouteNodes(nodes=(), coordinates=(), cumulative_s=())


def test_the_build_script_refuses_a_missing_extract_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """The extract is large and untracked, so the common failure is that it is
    simply not there. A short table would be worse than none."""
    table_path = tmp_path / "interchanges.csv"
    nodes_path = tmp_path / "interchange_nodes.json"
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            str(ROOT / "scripts/build_ic_table.py"),
            "--root",
            str(ROOT),
            "--pbf",
            str(tmp_path / "absent.osm.pbf"),
            "--table",
            str(table_path),
            "--nodes",
            str(nodes_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "is missing" in result.stderr
    assert not table_path.exists()
    assert not nodes_path.exists()


def test_this_indicator_makes_no_paid_call() -> None:
    """Judged on the module's source, never on a row count of some table: nothing
    here constructs a `RoutesClient` at all, which is the strongest form of the
    claim, so what is checked is that the module never imports one."""
    from landeval import interchange

    source = Path(interchange.__file__).read_text(encoding="utf-8")
    assert "routes" not in source.lower().split("\n")[0]
    assert "RoutesClient" not in source
    assert "googleapis" not in source
