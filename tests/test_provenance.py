"""Tests for the fetch-time record and the batch definition.

Three things decide whether this layer is worth having.

**A measurement whose fetch time cannot be resolved must stop the run.** The
whole point is that a row says when it was fetched; a row that quietly gets a
null instead is the false-clean shape -- the table then reports "recorded" for a value
nobody recorded.

**The digest is computed in one place.** The cache path and the provenance
lookup have to agree on the key, and two implementations of one hash drift
without anything failing.

**A batch is not a time pattern.** One schedule id can be fetched on two days
and shared by several destination groups, so the pair (group, schedule) on its
own can name two different quantities.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from landeval.basedate import JST
from landeval.config import Schedule
from landeval.provenance import (
    SOURCE_CACHE_MTIME,
    SOURCE_RESPONSE,
    ProvenanceError,
    attach_provenance,
    batch_id,
    cache_digests,
    provenance_from_cache,
    schedules_spanning_days,
    write_provenance,
)
from landeval.routes import RoutesClient, cache_digest

DEST = (35.680000, 139.760000)
YOKOHAMA = (35.4657, 139.6222)
DEPARTURE = datetime(2026, 7, 17, 7, 30, tzinfo=JST)
FETCHED = datetime(2026, 7, 8, 22, 15, tzinfo=JST)

OUTBOUND = Schedule(
    id="dep_0730",
    type="depart",
    time="07:30",
    daytype=["friday"],
    direction="outbound",
)
INBOUND = Schedule(
    id="ret_2030",
    type="depart",
    time="20:30",
    daytype=["friday"],
    direction="inbound",
)
SCHEDULES = {
    ("commute", "dep_0730"): OUTBOUND,
    ("commute", "ret_2030"): INBOUND,
}


def measured_row(schedule_id: str = "dep_0730", **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "lat": YOKOHAMA[0],
        "lon": YOKOHAMA[1],
        "schedule_id": schedule_id,
        "destination_group": "commute",
        "destination_lat": DEST[0],
        "destination_lon": DEST[1],
        "departure_time": DEPARTURE,
        "measured_min": 42.0,
    }
    row.update(overrides)
    return row


def provenance_of(frame: pd.DataFrame, fetched_at: datetime = FETCHED) -> pd.DataFrame:
    digests = cache_digests(frame, SCHEDULES).dropna()
    return pd.DataFrame(
        {
            "cache_digest": digests.unique(),
            "fetched_at": fetched_at,
            "source": "cache_mtime",
        }
    )


# --- the key is computed once ----------------------------------------------


def test_the_digest_is_the_one_the_client_caches_under(tmp_path: Path) -> None:
    """One hash, one implementation. A second copy drifts silently."""
    client = RoutesClient(api_key="unused", cache_dir=tmp_path)

    path = client._cache_path(YOKOHAMA, DEST, DEPARTURE)

    assert path.stem == cache_digest(YOKOHAMA, DEST, DEPARTURE)


def test_the_digest_follows_the_direction_the_schedule_declares() -> None:
    """An inbound schedule routes from the destination, and the cache key says
    so. Reading the direction from the id is a known mistake."""
    frame = pd.DataFrame([measured_row(), measured_row("ret_2030")])

    digests = cache_digests(frame, SCHEDULES)

    assert digests.iloc[0] == cache_digest(YOKOHAMA, DEST, DEPARTURE)
    assert digests.iloc[1] == cache_digest(DEST, YOKOHAMA, DEPARTURE)


def test_an_undeclared_schedule_is_an_error_not_a_skip() -> None:
    """A row the configuration does not describe cannot be keyed at all, and
    guessing a group for it invents the destination."""
    frame = pd.DataFrame([measured_row("group_a_depart_0630")])

    with pytest.raises(ProvenanceError, match="group_a_depart_0630"):
        cache_digests(frame, SCHEDULES)


# --- what a batch is --------------------------------------------------------


def test_batch_id_carries_group_schedule_and_fetch_date() -> None:
    assert batch_id("commute", "dep_1730", FETCHED) == "commute/dep_1730/2026-07-08"


def test_one_schedule_fetched_on_two_days_is_two_batches() -> None:
    """One schedule, ``dep_1730``, fetched late on one day and again early the next. If
    the fetch date were not in the key, the "one batch, one day" rule would be
    satisfied in wording by a batch that spans two days."""
    first = batch_id("commute", "dep_1730", datetime(2026, 7, 8, 23, 0, tzinfo=JST))
    second = batch_id("commute", "dep_1730", datetime(2026, 7, 9, 1, 0, tzinfo=JST))

    assert first != second


def test_two_groups_sharing_one_schedule_id_are_two_batches() -> None:
    """``dep_1730`` is declared by commute, group_a and group_c. The key has to
    separate them before any of their quantities are mixed."""
    commute = batch_id("commute", "dep_1730", FETCHED)
    group_a = batch_id("group_a", "dep_1730", FETCHED)

    assert commute != group_a


def test_a_schedule_measured_on_two_days_is_reported() -> None:
    frame = pd.DataFrame(
        [
            measured_row(),
            measured_row(departure_time=DEPARTURE.replace(hour=9)),
        ]
    )
    attached = attach_provenance(
        frame,
        pd.concat(
            [
                provenance_of(frame.iloc[[0]], FETCHED),
                provenance_of(frame.iloc[[1]], FETCHED.replace(day=13)),
            ]
        ),
        SCHEDULES,
    )

    spanning = schedules_spanning_days(attached)

    assert spanning == {("commute", "dep_0730"): 2}


def test_a_schedule_measured_in_one_day_is_not_reported() -> None:
    frame = pd.DataFrame([measured_row()])

    attached = attach_provenance(frame, provenance_of(frame), SCHEDULES)

    assert schedules_spanning_days(attached) == {}


# --- attaching the record ---------------------------------------------------


def test_a_measured_row_gets_its_fetch_time_and_batch() -> None:
    frame = pd.DataFrame([measured_row()])

    attached = attach_provenance(frame, provenance_of(frame), SCHEDULES)

    assert attached.loc[0, "fetched_at"] == pd.Timestamp(FETCHED)
    assert attached.loc[0, "batch_id"] == "commute/dep_0730/2026-07-08"


def test_a_measured_row_with_no_provenance_stops_the_run() -> None:
    """The failure this exists to prevent: many rows recorded, one
    silently null, and a table that says every record carries a fetch time."""
    frame = pd.DataFrame([measured_row(), measured_row(departure_time=DEPARTURE.replace(hour=9))])
    partial = provenance_of(frame.iloc[[0]])

    with pytest.raises(ProvenanceError, match="1"):
        attach_provenance(frame, partial, SCHEDULES)


def test_a_row_without_a_measurement_carries_no_record() -> None:
    """Rows that hold no measurement never reached the API, so they have
    no fetch time -- and that is not the same as one that went missing."""
    frame = pd.DataFrame(
        [measured_row(), measured_row(measured_min=None, departure_time=None)],
    )

    attached = attach_provenance(frame, provenance_of(frame), SCHEDULES)

    assert attached["fetched_at"].isna().loc[1]
    assert attached["batch_id"].isna().loc[1]


# --- reading the fetch times off the cache, once ----------------------------


def write_cache_entry(
    cache_dir: Path, digest: str, mtime: datetime, fetched_at: datetime | None = None
) -> Path:
    path = cache_dir / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {"duration_s": 1800, "distance_m": 27000}
    if fetched_at is not None:
        body["fetched_at"] = fetched_at.isoformat()
    path.write_text(json.dumps(body), encoding="utf-8")
    stamp = mtime.timestamp()
    os.utime(path, (stamp, stamp))
    return path


def test_a_cache_entry_without_a_fetch_time_falls_back_to_its_mtime(tmp_path: Path) -> None:
    """Which is a proxy, and the reason this runs before anything copies the
    tree: mtime does not survive `cp` without -p, a fresh clone, or a move."""
    write_cache_entry(tmp_path, "a" * 32, mtime=FETCHED)

    table = provenance_from_cache(tmp_path)

    assert list(table["cache_digest"]) == ["a" * 32]
    assert table.loc[0, "fetched_at"] == pd.Timestamp(FETCHED)
    assert table.loc[0, "source"] == SOURCE_CACHE_MTIME


def test_a_cache_entry_that_records_its_fetch_time_is_believed_over_mtime(tmp_path: Path) -> None:
    """An entry written after fetch times were recorded knows when it was fetched. Its mtime moves
    if the file is ever rewritten; the recorded value does not."""
    write_cache_entry(tmp_path, "b" * 32, mtime=FETCHED.replace(day=15), fetched_at=FETCHED)

    table = provenance_from_cache(tmp_path)

    assert table.loc[0, "fetched_at"] == pd.Timestamp(FETCHED)
    assert table.loc[0, "source"] == SOURCE_RESPONSE


def test_a_cache_file_that_is_not_readable_stops_the_scan(tmp_path: Path) -> None:
    """A cache tree is an open grammar -- a truncated write leaves a file that
    parses as nothing. Skipping it would drop a measurement's fetch time and
    report a complete table."""
    path = tmp_path / "cc" / f"{'c' * 32}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="c" * 32):
        provenance_from_cache(tmp_path)


def test_a_cache_file_whose_name_is_not_a_digest_stops_the_scan(tmp_path: Path) -> None:
    write_cache_entry(tmp_path, "d" * 32, mtime=FETCHED)
    (tmp_path / "dd" / "notes.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="notes"):
        provenance_from_cache(tmp_path)


def test_a_cache_file_holding_something_other_than_an_object_stops_the_scan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ee" / f"{'e' * 32}.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="list"):
        provenance_from_cache(tmp_path)


@pytest.mark.parametrize("recorded", [1755000000, "not-a-timestamp"])
def test_a_cache_file_with_an_unreadable_fetch_time_stops_the_scan(
    tmp_path: Path, recorded: object
) -> None:
    """Absent means "written before the field existed" and falls back to mtime.
    Malformed means a writer is getting it wrong right now, and answering both
    the same way hides the second."""
    path = tmp_path / "ff" / f"{'f' * 32}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"duration_s": 1, "fetched_at": recorded}), encoding="utf-8")

    with pytest.raises(ProvenanceError, match="fetched_at"):
        provenance_from_cache(tmp_path)


def test_an_empty_cache_tree_is_an_error_not_an_empty_table(tmp_path: Path) -> None:
    """An empty table would attach nulls everywhere and read as "nothing was
    ever fetched", which is a claim about the data rather than about the scan."""
    with pytest.raises(ProvenanceError, match="no cache entries"):
        provenance_from_cache(tmp_path)


def test_an_existing_table_is_not_rebuilt_without_being_asked(tmp_path: Path) -> None:
    """Rebuilding re-reads mtimes. If they have been lost since the first run,
    a silent rebuild replaces a correct record with a wrong one -- so the
    condition refuses rather than skipping."""
    path = tmp_path / "fetch_provenance.parquet"
    table = pd.DataFrame(
        {"cache_digest": ["e" * 32], "fetched_at": [pd.Timestamp(FETCHED)], "source": ["response"]}
    )
    write_provenance(path, table)

    with pytest.raises(ProvenanceError, match="--rebuild"):
        write_provenance(path, table)

    write_provenance(path, table, rebuild=True)


def test_a_frame_missing_the_destination_columns_says_so() -> None:
    """The destination columns are attached first; without them there
    is no coordinate to key on."""
    frame = pd.DataFrame([measured_row()]).drop(columns=["destination_lat"])

    with pytest.raises(ProvenanceError, match="destination_lat"):
        cache_digests(frame, SCHEDULES)


def test_a_frame_that_cannot_say_which_rows_were_measured_says_so() -> None:
    frame = pd.DataFrame([measured_row()]).drop(columns=["measured_min"])

    with pytest.raises(ProvenanceError, match="measured_min"):
        cache_digests(frame, SCHEDULES)


def test_a_measurement_with_no_departure_time_is_neither_kind_of_row() -> None:
    """It cannot be keyed, and calling it unmeasured would give it the same
    null as a row that never reached the API. The writers do not produce such a
    row; this is what makes one arriving a failure."""
    frame = pd.DataFrame([measured_row(departure_time=None)])

    with pytest.raises(ProvenanceError, match="departure_time"):
        cache_digests(frame, SCHEDULES)


def test_a_provenance_table_missing_its_columns_says_so() -> None:
    frame = pd.DataFrame([measured_row()])
    table = provenance_of(frame).drop(columns=["source"])

    with pytest.raises(ProvenanceError, match="source"):
        attach_provenance(frame, table, SCHEDULES)


def test_two_fetch_times_for_one_query_stop_the_run() -> None:
    """Two answers to one question. Taking either one silently picks a batch."""
    frame = pd.DataFrame([measured_row()])
    table = provenance_of(frame)
    doubled = pd.concat([table, table.assign(fetched_at=FETCHED.replace(day=13))])

    with pytest.raises(ProvenanceError, match="twice"):
        attach_provenance(frame, doubled, SCHEDULES)


def test_a_table_with_no_measurements_spans_no_days() -> None:
    frame = pd.DataFrame([measured_row(measured_min=None, departure_time=None)])

    attached = attach_provenance(frame, provenance_of(pd.DataFrame([measured_row()])), SCHEDULES)

    assert schedules_spanning_days(attached) == {}


def test_existing_columns_are_not_overwritten_in_place() -> None:
    """The input frame is a measurement artefact; attaching a record returns a
    new frame rather than editing the one on disk."""
    frame = pd.DataFrame([measured_row()])

    attach_provenance(frame, provenance_of(frame), SCHEDULES)

    assert "fetched_at" not in frame.columns
