"""Build the interchange table `landeval.interchange` reads.

Writes two artifacts, neither of them tracked, both rebuilt from the OSM extract:

    data/processed/interchanges.csv        one row per `highway=motorway_junction`
                                           node in the region's bbox, carrying the
                                           connectivity verdict
    data/processed/interchange_nodes.json  every corridor node from which the
                                           expressway mainline is reachable
                                           without passing a traffic signal

Both carry the same `build_id`, a digest of the two payloads. They are read as a
pair and nothing makes two file writes atomic together, so the reader refuses a
pair whose ids differ rather than answering from a table and a graph that were
never built from the same pass.

The rule, as decided:

    graph   = ways with highway in {motorway, motorway_link, trunk, trunk_link},
              plus primary/primary_link ways tagged toll=yes or motorroad=yes.
              DIRECTED, following `oneway`.
    cut     = a `highway=traffic_signals` node blocks only the direction it
              governs (`traffic_signals:direction`); an unrecognised value fails
              closed and blocks both, which errs toward "not connected".
    connected(J) = the mainline (`highway=motorway`) is reachable from J.

Reachability is one reverse breadth-first search seeded with every mainline node,
not a search per junction: connected means the junction can reach the mainline,
which is the direction a driver travels, so an exit-only ramp is not counted as
an entrance. Signals are cut before reachability is asked, not after -- on a road
whose two carriageways are separate ways, asking in the other order cuts the side
that has no signal on it.

Unbuilt interchanges are excluded twice over. A `highway=construction` way is
not in the graph, so nothing lying only on one can reach the mainline -- that is
what puts an `IC予定地` node on the not-connected side.
That alone is not enough, because a junction node **on a built mainline way** can
carry its own `construction` / `proposed` / `planned` tag and would be seeded as
reachable before the way filter ever sees it, so the node's own lifecycle tags
are checked here as well. Emergent exclusion is not exclusion; the rule has to
exclude them itself, not as a side effect.

Both inputs are required and neither is optional. A missing pbf and an absent
osmium both stop the build with a non-zero exit; writing a short table would be
worse than writing none, because the query side cannot tell a short table from a
country with few interchanges.

This file is the one that ships the rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict, deque
from itertools import pairwise
from pathlib import Path
from typing import Any

from _root import add_config_argument, add_root_argument, load_region, resolve_root, under_root

DEFAULT_PBF = "data/raw/japan-latest.osm.pbf"
DEFAULT_TABLE = "data/processed/interchanges.csv"
DEFAULT_NODES = "data/processed/interchange_nodes.json"

CORE = {"motorway", "motorway_link", "trunk", "trunk_link"}
# Lifecycle tags. A junction carrying any of these as a key, or naming one as its
# `highway`, is not open to traffic whatever the ways around it say.
UNBUILT = ("construction", "proposed", "planned")
QUALIFYING_PRIMARY = {"primary", "primary_link"}
MAINLINE = "motorway"

# A signal tagged as governing one direction does not govern the other. Anything
# else -- absent, `both`, or one of the bearing spellings such as `245;335` --
# governs both, which is the fail-closed side.
FORWARD_UNBLOCKED = frozenset({"backward"})
BACKWARD_UNBLOCKED = frozenset({"forward"})

COLUMNS = ["build_id", "node_id", "lon", "lat", "name", "name_en", "ref", "unbuilt", "connected"]


def _oneway(tags: Any, highway: str) -> int:
    """1 forward-only, -1 backward-only, 0 both."""
    value = tags.get("oneway")
    if value in ("yes", "true", "1"):
        return 1
    if value in ("-1", "reverse"):
        return -1
    if value in ("no", "false", "0"):
        return 0
    if tags.get("junction") in ("roundabout", "circular"):
        return 1
    return 1 if highway == MAINLINE else 0


def _is_corridor(tags: Any, highway: str) -> bool:
    if highway in CORE:
        return True
    return highway in QUALIFYING_PRIMARY and (
        tags.get("toll") == "yes" or tags.get("motorroad") == "yes"
    )


def build(root: Path, pbf: Path, table_path: Path, nodes_path: Path, config_path: Path) -> int:
    try:
        import osmium
    except ImportError:
        print("error: osmium is not installed; run `uv sync`.", file=sys.stderr)
        return 1
    if not pbf.is_file():
        print(f"error: {pbf} is missing; nothing to build from.", file=sys.stderr)
        return 1
    # The table is a lookup aid for addresses inside the configured region, so
    # a nationwide table would hold rows that can never be an answer.
    bbox = load_region(config_path).bbox

    class Collector(osmium.SimpleHandler):
        """One pass over nodes: the junctions to report and the signals to cut."""

        def __init__(self) -> None:
            super().__init__()
            self.junctions: list[dict[str, str]] = []
            self.governs: dict[int, str] = {}
            self.unbuilt = 0

        def node(self, node: Any) -> None:
            highway = node.tags.get("highway")
            if highway == "traffic_signals":
                self.governs[node.id] = node.tags.get("traffic_signals:direction") or "both"
                return
            if highway != "motorway_junction":
                return
            lon, lat = node.location.lon, node.location.lat
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                return
            unbuilt = any(node.tags.get(key) for key in UNBUILT)
            self.unbuilt += unbuilt
            self.junctions.append(
                {
                    "node_id": str(node.id),
                    "lon": f"{lon:.6f}",
                    "lat": f"{lat:.6f}",
                    "name": node.tags.get("name", ""),
                    "name_en": node.tags.get("name:en", ""),
                    "ref": node.tags.get("ref", ""),
                    "unbuilt": "true" if unbuilt else "false",
                }
            )

    class Corridor(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.ways: list[tuple[bool, int, list[int]]] = []

        def way(self, way: Any) -> None:
            highway = way.tags.get("highway")
            if highway is None or not _is_corridor(way.tags, highway):
                return
            self.ways.append(
                (highway == MAINLINE, _oneway(way.tags, highway), [n.ref for n in way.nodes])
            )

    collector = Collector()
    collector.apply_file(str(pbf), locations=False)
    print(f"traffic signals   : {len(collector.governs)}")
    print(f"junctions in bbox : {len(collector.junctions)}")
    print(f"  tagged unbuilt  : {collector.unbuilt}")
    if not collector.junctions:
        print("error: no motorway_junction node fell inside the bbox.", file=sys.stderr)
        return 1

    corridor = Corridor()
    corridor.apply_file(str(pbf), locations=False)
    print(f"corridor ways     : {len(corridor.ways)}")

    reverse: dict[int, list[int]] = defaultdict(list)
    mainline: set[int] = set()
    in_graph: set[int] = set()
    for is_mainline, direction, nodes in corridor.ways:
        if is_mainline:
            mainline.update(nodes)
        in_graph.update(nodes)
        for a, b in pairwise(nodes):
            if direction >= 0 and collector.governs.get(b) in (None, *FORWARD_UNBLOCKED):
                reverse[b].append(a)
            if direction <= 0 and collector.governs.get(a) in (None, *BACKWARD_UNBLOCKED):
                reverse[a].append(b)

    reachable = set(mainline & in_graph)
    queue = deque(reachable)
    while queue:
        node = queue.popleft()
        for previous in reverse.get(node, ()):
            if previous not in reachable:
                reachable.add(previous)
                queue.append(previous)
    print(f"graph nodes       : {len(in_graph)}   reaching the mainline: {len(reachable)}")
    if not reachable:
        print("error: no node reaches the mainline; the graph is empty.", file=sys.stderr)
        return 1

    for row in collector.junctions:
        open_to_traffic = row["unbuilt"] == "false" and int(row["node_id"]) in reachable
        row["connected"] = "true" if open_to_traffic else "false"
    connected = sum(row["connected"] == "true" for row in collector.junctions)

    # The two artifacts have to be read as a pair, and nothing about a file
    # system makes two writes atomic together. So they carry a shared build id,
    # derived from both payloads, and the reader refuses a pair that disagrees --
    # a new table beside an old node set would otherwise answer with entries
    # computed against a graph the table never saw, and look consistent.
    # The temporary files narrow the window; the id is what closes it.
    rows = sorted(collector.junctions, key=lambda r: int(r["node_id"]))
    node_list = sorted(reachable)
    digest = hashlib.sha256()
    digest.update(json.dumps(node_list).encode())
    digest.update(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode())
    build_id = digest.hexdigest()[:16]
    for row in rows:
        row["build_id"] = build_id

    for path in (table_path, nodes_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    table_tmp = table_path.with_suffix(table_path.suffix + ".tmp")
    nodes_tmp = nodes_path.with_suffix(nodes_path.suffix + ".tmp")
    try:
        with table_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        nodes_tmp.write_text(
            json.dumps({"build_id": build_id, "nodes": node_list}), encoding="utf-8"
        )
    except OSError as error:
        table_tmp.unlink(missing_ok=True)
        nodes_tmp.unlink(missing_ok=True)
        print(f"error: could not write the artifacts: {error}", file=sys.stderr)
        return 1
    table_tmp.replace(table_path)
    nodes_tmp.replace(nodes_path)
    print(f"build id          : {build_id}")

    print(f"\nconnected         : {connected}/{len(collector.junctions)}")
    print(f"wrote {table_path.relative_to(root)} and {nodes_path.relative_to(root)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_root_argument(parser)
    add_config_argument(parser)
    parser.add_argument("--pbf", type=Path, default=under_root(DEFAULT_PBF))
    parser.add_argument("--table", type=Path, default=under_root(DEFAULT_TABLE))
    parser.add_argument("--nodes", type=Path, default=under_root(DEFAULT_NODES))
    args = parser.parse_args(argv)
    root = resolve_root(args)
    return build(root, args.pbf, args.table, args.nodes, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
