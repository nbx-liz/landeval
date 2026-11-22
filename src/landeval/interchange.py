"""Nearest interchange, read off a route rather than from a candidate set.

Two things make this harder than "find the
closest node tagged `highway=motorway_junction`":

* **Not every interchange leads to an expressway.** Some junctions named as
  interchanges are grade-separated crossings on ordinary bypasses, while some
  free roads reach the mainline. Neither `toll` nor `motorroad` separates the
  two, alone or together. The rule is connectivity -- can the
  expressway mainline be reached from here without passing a traffic signal --
  and it is decided at build time, by `scripts/build_ic_table.py`, which writes
  the `connected` column this module reads.
* **The route need not pass through the junction node.** A junction node
  (`highway=motorway_junction`) marks where an exit leaves the carriageway, so
  it sits on the exit ramp rather than on the entrance ramp, and intersecting
  the route with the junction set answers with the *next* interchange down the
  road: the next interchange down the road is named for the one the car
  entered at, which is the wrong answer whenever the entrance node is not on
  the route.

So the entry point and the name come from different places. The entry point is
the first node of the route that can reach the mainline signal-free, which is
exact and carries the direction constraint for free (the route only enters where
a car can). The name is the interchange nearest that node, which is an
attribution and can be wrong; `NearestInterchange.attribution_m` is how far it
had to reach, and callers print it rather than presenting the name as certain.
The reach can be large enough that hiding it behind a threshold would be a
claim, so it is always shown.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

Coordinate = tuple[float, float]  # (lat, lon)

_SMART_KATAKANA = "スマート"
# `SIC` standing alone, not the tail of a longer latin word. Some junctions are
# spelled this way and no katakana rule sees any of them.
_SMART_ABBREVIATION = re.compile(r"(?:^|[^A-Za-z])S\.?I\.?C", re.ASCII)

# A junction whose name says only "junction" or only "rest area" is not a place a
# car can enter the network from. Written as a subtraction rather than as a
# requirement: a connected named junction can carry no interchange token at all
# (urban expressway entrances are often named for the place, as in `〇〇入口`),
# so requiring one would discard the whole urban network.
_JUNCTION_ONLY = re.compile(r"JCT|ＪＣＴ|ジャンクション")
_REST_AREA = re.compile(r"PA|ＰＡ|SA|ＳＡ|パーキング|サービスエリア")
_INTERCHANGE = re.compile(r"IC|ＩＣ|インター|S\.?I\.?C")

# The shape `scripts/build_ic_table.py` writes: the first 16 hex digits of a
# digest over both payloads. Checked rather than merely compared, because two
# empty strings compare equal -- an unlabelled pair would have matched itself
# and passed as one build.
_BUILD_ID = re.compile(r"[0-9a-f]{16}")


class InterchangeError(Exception):
    """Raised when the interchange table is missing or cannot be read."""


@dataclass(frozen=True)
class Interchange:
    """One `highway=motorway_junction` node that reaches the mainline."""

    node_id: int
    name: str
    name_en: str
    lat: float
    lon: float

    @property
    def is_smart(self) -> bool:
        """Whether the name says this is a smart interchange.

        Three disjoint families, because no tag reliably carries it. A suffix
        match on `スマートIC` misses names that carry the token elsewhere; the
        katakana substring misses the `SIC` abbreviations; and a node whose
        Japanese name is a bare place name says `Smart` only in `name:en`.
        """
        return (
            _SMART_KATAKANA in self.name
            or bool(_SMART_ABBREVIATION.search(self.name))
            or "smart" in self.name_en.lower()
        )

    @property
    def is_entrance(self) -> bool:
        """Whether a car can join the network here, as far as the name says."""
        if _INTERCHANGE.search(self.name):
            return True
        return not _JUNCTION_ONLY.search(self.name) and not _REST_AREA.search(self.name)


@dataclass(frozen=True)
class RouteNodes:
    """The part of an OSRM `/route` answer this module reads.

    The three arrays are index-aligned: `coordinates[i]` is where `nodes[i]` is
    and `cumulative_s[i]` is how long the route has taken to get there. OSRM
    returns them that way; construction refuses anything else rather than
    reading a clock that does not line up with the nodes it is timing.
    """

    nodes: tuple[int, ...]
    coordinates: tuple[Coordinate, ...]
    cumulative_s: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.nodes), len(self.coordinates), len(self.cumulative_s)}
        if len(lengths) != 1:
            raise InterchangeError(
                f"route arrays are not index-aligned: {len(self.nodes)} nodes, "
                f"{len(self.coordinates)} coordinates, {len(self.cumulative_s)} times"
            )
        if not self.nodes:
            raise InterchangeError("route has no nodes")


@dataclass(frozen=True)
class NearestInterchange:
    """Where the route joined the expressway network, and what to call it."""

    interchange: Interchange
    minutes: float
    attribution_m: float

    @property
    def is_smart(self) -> bool:
        return self.interchange.is_smart


def haversine_m(a: Coordinate, b: Coordinate) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    chord = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 12742000.0 * math.asin(math.sqrt(chord))


@dataclass(frozen=True)
class InterchangeTable:
    """The connected interchanges, and the corridor nodes that reach the mainline."""

    interchanges: tuple[Interchange, ...]
    qualified_nodes: frozenset[int]

    @classmethod
    def load(cls, table_path: Path, nodes_path: Path) -> InterchangeTable:
        """Read both artifacts, or say which one is missing.

        Neither is tracked -- both are rebuilt from the OSM extract by
        `scripts/build_ic_table.py`. An absent file is an error and not an empty
        table, because an empty table reports "no expressway used" for every
        address in the country.

        The two are one artifact in two files, so they carry a shared build id
        and a pair that disagrees is refused. Two writes cannot be made atomic
        together; a new table beside an old node set parses perfectly and answers
        with entries computed against a graph the table never saw.
        """
        for path in (table_path, nodes_path):
            if not path.is_file():
                raise InterchangeError(
                    f"{path} is missing; build it with scripts/build_ic_table.py"
                )
        rows: list[Interchange] = []
        table_ids: set[str] = set()
        with table_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"build_id", "node_id", "lon", "lat", "name", "name_en", "connected"}
            if reader.fieldnames is None or not expected <= set(reader.fieldnames):
                raise InterchangeError(
                    f"{table_path} has columns {reader.fieldnames}, expected {sorted(expected)}"
                )
            for row in reader:
                table_ids.add(row["build_id"])
                # Two filters, both deliberate. `connected` is the connectivity rule.
                # An unnamed junction is dropped because it cannot supply a name
                # and would otherwise win the attribution and print nothing.
                if row["connected"] != "true" or not row["name"]:
                    continue
                rows.append(
                    Interchange(
                        node_id=int(row["node_id"]),
                        name=row["name"],
                        name_en=row["name_en"],
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                    )
                )
        entrances = tuple(row for row in rows if row.is_entrance)
        if not entrances:
            raise InterchangeError(f"{table_path} holds no connected named entrance")
        if len(table_ids) != 1:
            raise InterchangeError(
                f"{table_path} carries {len(table_ids)} build ids; it is not one build"
            )
        table_id = next(iter(table_ids))
        if not _BUILD_ID.fullmatch(table_id):
            raise InterchangeError(
                f"{table_path} carries {table_id!r} as its build id, which is not a digest; "
                "rebuild it with scripts/build_ic_table.py"
            )
        payload = json.loads(nodes_path.read_text(encoding="utf-8"))
        node_id = payload.get("build_id") if isinstance(payload, dict) else None
        if not isinstance(node_id, str) or not _BUILD_ID.fullmatch(node_id):
            raise InterchangeError(
                f"{nodes_path} carries no build id; rebuild both artifacts with "
                "scripts/build_ic_table.py"
            )
        nodes = payload.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise InterchangeError(f"{nodes_path} does not hold a non-empty list of node ids")
        if node_id != table_id:
            # A new table beside an old node set. Both files exist and both parse,
            # so nothing else here would notice.
            raise InterchangeError(
                f"{table_path} was built as {table_id} and {nodes_path} as "
                f"{node_id}; rebuild both with scripts/build_ic_table.py"
            )
        return cls(interchanges=entrances, qualified_nodes=frozenset(int(n) for n in nodes))

    def entry_index(self, route: RouteNodes) -> int | None:
        """Index of the first route node that reaches the mainline signal-free.

        `None` means the route never joins the expressway network, which is an
        answer ("no expressway used") and not a failure to find one.
        """
        return next((i for i, node in enumerate(route.nodes) if node in self.qualified_nodes), None)

    def nearest(self, route: RouteNodes) -> NearestInterchange | None:
        """The interchange nearest the entry point, and how far away it is.

        One distance, used for both jobs: it chooses the name and it is what
        gets reported. Choosing over a window of route nodes and reporting the
        window's minimum would under-report the entry-to-interchange distance
        while the printed sentence says the number is measured from the entry.
        Selecting on the window and reporting from the entry would be honest but
        buys nothing: selecting from the entry minimises the reported distance
        by construction.
        """
        entry = self.entry_index(route)
        if entry is None:
            return None
        origin = route.coordinates[entry]
        gap, best = min(
            ((haversine_m(origin, (row.lat, row.lon)), row) for row in self.interchanges),
            key=lambda pair: pair[0],
        )
        return NearestInterchange(
            interchange=best,
            minutes=route.cumulative_s[entry] / 60.0,
            attribution_m=gap,
        )
