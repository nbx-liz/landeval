"""When a measurement was fetched, and which batch it belongs to.

Two things are required: that one evaluation batch is fetched on one day, and
that every record carries its fetch time. The traffic side is where the fetch
date can move the value: the same query re-issued on a different day need not
answer the same.

Three decisions are encoded here.

**A batch is ``(destination_group, schedule_id, fetch date)``.** Not the time
pattern: one schedule id can be fetched on two days, and several destination
groups can declare an id by the same name. Either of those alone makes the pattern
ambiguous as a batch.

**A measured row whose fetch time cannot be resolved stops the run.** The
alternative -- leaving it null and carrying on -- produces a table that claims
to record fetch times while some records have none, which is the exact
false-clean shape the record is meant to close.

**Rows that hold no measurement carry no record.** A measurement table can
hold rows that never reached the API. "No fetch time because
nothing was fetched" and "fetch time went missing" must not look alike.
"""

from __future__ import annotations

import json
import string
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from landeval.basedate import JST
from landeval.config import Schedule, load_config
from landeval.routes import cache_digest

# What a cache file is named: the first 32 hex characters of the key's sha256.
_DIGEST_LENGTH = 32
_DIGEST_ALPHABET = frozenset(string.hexdigits.lower())

# What the provenance table holds: one row per cache entry, keyed by the digest
# the cache filed it under.
PROVENANCE_COLUMNS = ("cache_digest", "fetched_at", "source")

# Where a fetch time came from. The backfilled ones are read from the cache
# file's mtime, which is a proxy; anything written after fetch times were recorded comes from the
# call itself. Keeping the two apart means a later reader can tell which claim
# is which instead of trusting both equally.
SOURCE_CACHE_MTIME = "cache_mtime"
SOURCE_RESPONSE = "response"

FETCHED_AT_COLUMN = "fetched_at"
BATCH_COLUMN = "batch_id"

# What identifies the batch a row belongs to, before the fetch date is known.
BATCH_KEY = ("destination_group", "schedule_id")

# The tables that hold measurements. Named rather than globbed -- a glob would
# quietly adopt a new artefact and quietly miss a renamed one -- and kept here
# rather than in the backfill script, so the script and the tests that check
# its output cannot come to disagree about which tables exist.
MEASUREMENT_TABLES = (
    "calibration",
    "municipality_measured",
    "block_measurements",
    "municipality_rep_target_times",
    "block_rep_target_times",
    "municipality_rep_schedule_times",
    "block_rep_schedule_times",
    "block_rep_afternoon_outbound_times",
    # The group destinations. A measurement table absent from this tuple gets
    # no `batch_id`, and the unit of an evaluation would be missing for every
    # one of its rows without anything saying so.
    "group_destination_times",
)

# The columns a measurement row must carry before its cache key can be rebuilt.
# The destination is a column rather than a configuration lookup because the
# configured coordinate can change later, and the key was made from the one
# actually queried.
_KEY_COLUMNS = (
    "lat",
    "lon",
    "schedule_id",
    "destination_group",
    "destination_lat",
    "destination_lon",
    "departure_time",
)


class ProvenanceError(Exception):
    """Raised when a fetch time cannot be established for a measured row."""


def provenance_from_cache(cache_dir: Path) -> pd.DataFrame:
    """Read the fetch time of every cached response under ``cache_dir``.

    An entry written after fetch times were recorded states its own fetch time and is believed.
    An older one has none, so its file mtime stands in -- a proxy, and the
    reason this runs before anything copies the tree: ``cp`` without ``-p``, a
    fresh clone and a move all reset it.

    A file that cannot be read, or whose name is not a digest, raises. The
    cache directory is not a closed grammar -- a truncated write leaves a file
    that parses as nothing -- and skipping such a file would drop one
    measurement's fetch time while reporting a complete table.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(cache_dir.rglob("*.json")):
        digest = path.stem
        if len(digest) != _DIGEST_LENGTH or not set(digest) <= _DIGEST_ALPHABET:
            raise ProvenanceError(
                f"{path} is not named after a cache digest; a stray file in the cache "
                "tree means the scan is reading something it does not understand"
            )
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProvenanceError(f"{path} could not be read: {type(exc).__name__}") from None
        if not isinstance(body, dict):
            raise ProvenanceError(f"{path} holds a {type(body).__name__}, not an object")

        recorded = body.get("fetched_at")
        if recorded is None:
            moment = datetime.fromtimestamp(path.stat().st_mtime, JST)
            source = SOURCE_CACHE_MTIME
        else:
            if not isinstance(recorded, str):
                raise ProvenanceError(f"{path} holds a non-string fetched_at")
            try:
                moment = datetime.fromisoformat(recorded)
            except ValueError:
                raise ProvenanceError(
                    f"{path} holds a malformed fetched_at: {recorded!r}"
                ) from None
            source = SOURCE_RESPONSE
        rows.append({"cache_digest": digest, "fetched_at": moment, "source": source})

    if not rows:
        raise ProvenanceError(
            f"no cache entries under {cache_dir}; an empty table would record that "
            "nothing was ever fetched"
        )
    return pd.DataFrame(rows, columns=list(PROVENANCE_COLUMNS))


def write_provenance(path: Path, table: pd.DataFrame, *, rebuild: bool = False) -> None:
    """Write the provenance table, refusing to overwrite one that exists.

    Not "rebuild if missing": rebuilding re-reads mtimes, and if those have
    been lost since the first run the rebuild replaces a correct record with a
    wrong one. So the condition rejects rather than skipping.
    """
    if path.exists() and not rebuild:
        raise ProvenanceError(
            f"{path} already exists. Re-deriving it reads file mtimes again, which may "
            "since have been reset by a copy or a clone. Pass --rebuild to say that is "
            "what you mean."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def batch_id(destination_group: str, schedule_id: str, fetched_at: datetime) -> str:
    """The batch a row measured at ``fetched_at`` belongs to.

    The fetch date is part of the identity, so a schedule that ran on two days
    is two batches rather than one batch that breaks the one-day rule.
    """
    return f"{destination_group}/{schedule_id}/{fetched_at.date().isoformat()}"


def _measured(frame: pd.DataFrame) -> pd.Series:
    """Which rows hold a measurement, and therefore reached the API."""
    if "measured_min" not in frame.columns:
        raise ProvenanceError(
            "the frame has no measured_min column, so measured and unmeasured rows "
            f"cannot be told apart; it has {sorted(frame.columns)}"
        )
    measured = frame["measured_min"].notna()
    keyable = frame["departure_time"].notna()
    # A row that holds a measurement but no departure time cannot be keyed, and
    # treating it as unmeasured would give it a null fetch time on the same
    # footing as a row that never reached the API. The writers do not
    # produce such a row; this is here so that one arriving is a failure rather
    # than a quiet exemption.
    half = measured & ~keyable
    if half.any():
        raise ProvenanceError(
            f"{int(half.sum())} rows hold a measurement with no departure_time, so their "
            "cache key cannot be rebuilt; they are neither measured nor unmeasured as far "
            "as the fetch record is concerned"
        )
    return measured & keyable


def cache_digests(frame: pd.DataFrame, schedules: Mapping[tuple[str, str], Schedule]) -> pd.Series:
    """The cache key of every measured row, and ``NA`` for the rest.

    ``schedules`` is keyed by ``(destination_group, schedule_id)`` because the
    id alone does not name one schedule: several groups can declare the same
    id, and they can run in different directions. A row whose pair is not
    declared raises rather than being dropped -- an unkeyed row is a row whose
    destination we would have to invent.
    """
    missing = [column for column in _KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ProvenanceError(
            f"the columns {missing} are absent, so the cache key cannot be rebuilt; "
            "the destination columns are attached first"
        )

    measured = _measured(frame)
    pairs = list(
        zip(
            frame["destination_group"].astype(str),
            frame["schedule_id"].astype(str),
            strict=True,
        )
    )
    undeclared = sorted(
        {pair for pair, keep in zip(pairs, measured, strict=True) if keep and pair not in schedules}
    )
    if undeclared:
        raise ProvenanceError(
            f"the configuration declares no schedule for {undeclared}; "
            "a row whose direction is unknown cannot be keyed"
        )

    digests: list[str | None] = []
    for position, (pair, keep) in enumerate(zip(pairs, measured, strict=True)):
        if not keep:
            digests.append(None)
            continue
        row = frame.iloc[position]
        origin, destination = schedules[pair].endpoints(
            (float(row["lat"]), float(row["lon"])),
            (float(row["destination_lat"]), float(row["destination_lon"])),
        )
        digests.append(cache_digest(origin, destination, pd.Timestamp(row["departure_time"])))
    return pd.Series(digests, index=frame.index, dtype="object")


def attach_provenance(
    frame: pd.DataFrame,
    provenance: pd.DataFrame,
    schedules: Mapping[tuple[str, str], Schedule],
) -> pd.DataFrame:
    """Return ``frame`` with ``fetched_at`` and ``batch_id`` attached.

    A measured row that the provenance table does not cover raises. That is the
    whole guarantee: after this returns, every measured row has a fetch time.
    """
    missing_columns = [column for column in PROVENANCE_COLUMNS if column not in provenance.columns]
    if missing_columns:
        raise ProvenanceError(
            f"the provenance table lacks {missing_columns}; it has {sorted(provenance.columns)}"
        )
    duplicated = provenance["cache_digest"].duplicated()
    if duplicated.any():
        raise ProvenanceError(
            f"{int(duplicated.sum())} cache digests appear twice in the provenance table; "
            "two fetch times for one query is two answers to one question"
        )

    digests = cache_digests(frame, schedules)
    times = provenance.set_index("cache_digest")["fetched_at"]
    fetched = digests.map(times)

    unresolved = int((digests.notna() & fetched.isna()).sum())
    if unresolved:
        sample = sorted(set(digests[digests.notna() & fetched.isna()]))[:3]
        raise ProvenanceError(
            f"{unresolved} measured rows have no fetch time in the provenance table "
            f"(for example {sample}); recording a null for them would say the fetch "
            "time was recorded when it was not"
        )

    fetched = pd.to_datetime(fetched)
    batches = [
        batch_id(group, schedule, moment) if pd.notna(moment) else None
        for group, schedule, moment in zip(
            frame["destination_group"].astype(str),
            frame["schedule_id"].astype(str),
            fetched,
            strict=True,
        )
    ]
    # A string dtype rather than object for the batch, so that a row with no
    # batch reads back as the same null it was written as after a parquet round
    # trip. `None` written and `NaN` read is a difference that tempts the next
    # reader into treating one of them as a value.
    attached: dict[str, pd.Series[Any]] = {
        FETCHED_AT_COLUMN: fetched,
        BATCH_COLUMN: pd.Series(batches, index=frame.index, dtype="string"),
    }
    return frame.assign(**attached)


def schedules_spanning_days(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Every ``(group, schedule)`` fetched on more than one day, and on how many.

    One batch is fetched on one day. This does not enforce it -- by the time it
    runs, the measurements are already paid for. It reports, so that a batch
    spanning days cannot arrive unnoticed.
    """
    covered = frame[frame[FETCHED_AT_COLUMN].notna()]
    if covered.empty:
        return {}
    days = pd.to_datetime(covered[FETCHED_AT_COLUMN]).dt.date
    counts = (
        pd.DataFrame(
            {
                "destination_group": covered["destination_group"].astype(str),
                "schedule_id": covered["schedule_id"].astype(str),
                "day": days,
            }
        )
        .groupby(["destination_group", "schedule_id"])["day"]
        .nunique()
    )
    spanning: dict[tuple[str, str], int] = {}
    for key, value in counts.items():
        group, schedule_id = cast(tuple[str, str], key)
        if value > 1:
            spanning[(group, schedule_id)] = int(value)
    return spanning


def schedule_index(config_path: Path) -> dict[tuple[str, str], Schedule]:
    """Every declared schedule, keyed by (destination group, id).

    The id alone is not a key: one id can be declared by several groups and
    run in a different direction in each. Reading the direction from
    the id is a known mistake, so the pair is what callers pass around.
    """
    config = load_config(config_path)
    return {
        (group_name, schedule.id): schedule
        for group_name, group in config.destinations.items()
        for schedule in group.schedules
    }
