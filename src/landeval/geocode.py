"""Resolve a Japanese address, and say how much of it actually resolved.

The GSI address search returns a coordinate for almost anything. Ask it for a
street number it does not hold and it answers with the town, or with the
municipality, in the same shape and with no field saying which happened. That is
the false-clean shape -- a coarser answer wearing the clothes of a precise one -- and
downstream it decides everything: a street-number resolution earns a point
estimate and a 100 m hazard sample, a coarser one earns a distribution over an
area.

So this module answers two questions, and keeps them apart. `geocode` asks the
network where an address is. `resolve` is pure and asks how much of the query
the answer accounted for, by reading the returned string against the query and
against the municipality boundary set. A municipality the boundary set does not
contain is an error rather than a guess: every figure downstream is
keyed on that code, and inventing one puts the answer on a map that never had it.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from landeval.mesh import JAPAN_LAT_RANGE, JAPAN_LON_RANGE

ENDPOINT = "https://msearch.gsi.go.jp/address-search/AddressSearch"

_FULL_WIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
# Hyphen-like codepoints that appear between the parts of a street number. The
# katakana prolonged mark (ー) is deliberately absent: it is a letter in names
# such as ラーメン and folding it would corrupt them.
_HYPHENS = re.compile(r"[－−‐‑–—―]")
_WHITESPACE = re.compile(r"\s|　")
_DIGIT_RUN = re.compile(r"\d+")
# A digit run that names a 丁目 is naming a block of the town, not a plot. The
# whole token is matched, not just its number, because the number has to be
# removed where it stands rather than wherever its value happens to appear.
_CHOME_TOKEN = re.compile(r"\d+丁目")
# A pasted listing often carries the postal code in front of the address. It is
# seven digits that mean a delivery area, not a plot, and leaving them in
# reported a code such as '987-6543' as an unresolved part of the address.
_POSTAL_PREFIX = re.compile(r"^〒?\d{3}-?\d{4}(?=[^\d-])")
# ケ, ヶ and が between two non-hiragana characters are one morpheme with three
# spellings: 雪が谷 and 雪ケ谷 are the same town, 茅ケ崎 and 茅ヶ崎 the same
# city. The lookbehind keeps ordinary hiragana out of it, so みどりが丘 is left
# exactly as it was written on both sides.
_KE_MORPHEME = re.compile(r"(?<=[^぀-ゟ])[ヶが](?=[^぀-ゟ])")
# What a query may legitimately carry below the town name that the geocoder
# drops: the 小字 marker, and the 甲乙丙 parcel letters. A closed list, because
# the alternative -- accepting any prefix of the town -- would let an answer for
# 石川 satisfy a query about 石川台.
_DROPPED_TOWN_DETAIL = re.compile(r"^(?:字|[甲乙丙丁戊])")
# The geocoder writes 丁目 in kanji numerals (丸の内一丁目) where the query is
# typed in digits (丸の内1-9-1). Same block, two alphabets.
_KANJI_CHOME = re.compile(r"([一二三四五六七八九十]+)丁目")
_KANJI_DIGITS = "一二三四五六七八九"


def _kanji_number(text: str) -> int:
    """Read a kanji numeral in the range a 丁目 can take (1-99)."""
    if "十" not in text:
        return _KANJI_DIGITS.index(text) + 1
    tens, _, ones = text.partition("十")
    return (_KANJI_DIGITS.index(tens) + 1 if tens else 1) * 10 + (
        _KANJI_DIGITS.index(ones) + 1 if ones else 0
    )


class GeocodeError(Exception):
    """Raised when an address cannot be resolved to a usable, placeable answer."""


class Precision(StrEnum):
    """How much of the query the geocoder's answer accounted for."""

    STREET_NUMBER = "street_number"
    TOWN = "town"
    MUNICIPALITY = "municipality"

    @property
    def is_point(self) -> bool:
        """Whether the answer names a plot rather than an area.

        This is the gate the measured route sits behind, so both of its branches
        have to stay live: if street-number precision never occurred, the point
        path would be dead and the distribution path the only real one.

        The geocoder answers at street-number precision only when it holds the
        plot. An address written all the way down to the plot number can still
        come back at *town* precision with the plot number unresolved, so either
        branch can answer any given address and neither is a fallback for the
        other.
        """
        return self is Precision.STREET_NUMBER


@dataclass(frozen=True)
class Resolution:
    """What the geocoder accounted for, and what it silently did not."""

    precision: Precision
    municipality: str
    residue: str
    """What the answer never accounted for. Empty when it accounted for all of
    the query. Reported rather than dropped, because "5831" going missing is the
    difference between a plot and a town."""


@dataclass(frozen=True)
class Match:
    """A geocoder answer, before anything is judged about its precision."""

    lat: float
    lon: float
    matched: str


def normalize(text: str) -> str:
    """Fold the spelling differences that are not differences of place.

    The query is typed in half width and the answer comes back in full width;
    ヶ and ケ are the same ward in 茅ケ崎市; 大字 is written by people and
    omitted by the geocoder. None of these are places, so none of them may
    decide whether two strings are the same place.
    """
    folded = text.translate(_FULL_WIDTH_DIGITS)
    folded = _HYPHENS.sub("-", folded)
    folded = _WHITESPACE.sub("", folded)
    folded = _POSTAL_PREFIX.sub("", folded)
    folded = _KE_MORPHEME.sub("ケ", folded)
    folded = _KANJI_CHOME.sub(lambda m: f"{_kanji_number(m.group(1))}丁目", folded)
    return folded.replace("大字", "")


def _municipality_of(matched: str, municipalities: Mapping[str, str]) -> tuple[str, int]:
    """Return the canonical name for ``matched`` and the length it consumed.

    The alias, not the canonical name, is what the address is matched against:
    the geocoder omits the county, so 〇〇県〇〇郡〇〇町 arrives as 〇〇県〇〇町,
    and a user may omit the prefecture entirely. Only an alias reaches either.

    Longest wins. The choice matters only when one alias prefixes another: a
    reference set that held both a designated city and its wards would
    otherwise resolve to whichever row the dict happened to reach first, and
    the ward is the unit every stored measurement is keyed on.

    The designated cities fail the other way round: ``municipality_name``
    always writes the ward out, so the set holds one entry per ward and no
    city-only entry, and a GSI title that stops at 横浜市 raises rather than
    resolving. That is the right
    answer -- 横浜市 has no single code to answer for -- but it is a refusal,
    not a tiebreak.
    """
    hits = [
        (canonical, len(normalize(alias)))
        for alias, canonical in municipalities.items()
        if matched.startswith(normalize(alias))
    ]
    if not hits:
        raise GeocodeError(
            f"{matched!r}: no municipality in the reference set prefixes this address; "
            "refusing to guess one"
        )
    return max(hits, key=lambda hit: hit[1])


def _municipality_of_or_none(
    text: str, municipalities: Mapping[str, str]
) -> tuple[str | None, int]:
    """As ``_municipality_of``, but tolerant: the query may omit it entirely.

    A user typing a bare street address is not an error; a user typing one
    municipality and being answered about another is.
    """
    try:
        return _municipality_of(text, municipalities)
    except GeocodeError:
        return None, 0


def _unaccounted(rest_query: str, rest_matched: str) -> list[str]:
    """Digit runs the query asked about that the answer did not carry.

    Read in order, so 丸の内1-9-1 against 丸の内1丁目9番 reports the 号 alone
    rather than the whole tail: a longest-common-prefix reading calls the 6
    unresolved even though the answer names it, which overstates the loss.
    """
    carried = _DIGIT_RUN.findall(rest_matched)
    missing: list[str] = []
    index = 0
    for run in _DIGIT_RUN.findall(rest_query):
        if index < len(carried) and carried[index] == run:
            index += 1
        else:
            missing.append(run)
    return missing


def _town_of(rest: str) -> str:
    """The text before the first number: the town name, without any of it."""
    return _DIGIT_RUN.split(rest, maxsplit=1)[0]


def _same_town(town_matched: str, town_query: str) -> bool:
    """Whether the two names are the same town written at different depth.

    Neither side may be empty. A guard written as
    ``if town_matched and town_query and town_matched != town_query``
    reads like a guard and behaves like one only when both sides are populated:
    a query of ``1-1`` names no town at all, would skip the comparison entirely,
    and come back at street-number precision for whatever plot the geocoder
    chose to answer with.

    Above that, the query is allowed to be *more* specific than the answer and
    the reverse, but only by the 小字 marker or a 甲乙丙 parcel letter -- the
    two things the geocoder is known to drop. Accepting any prefix instead
    would let an answer for 石川 satisfy a query about 石川台, which is a
    different town whose name happens to start the same way.
    """
    if not town_matched or not town_query:
        return False
    if town_matched == town_query:
        return True
    short, long_ = sorted((town_matched, town_query), key=len)
    return long_.startswith(short) and bool(_DROPPED_TOWN_DETAIL.match(long_[len(short) :]))


def _plot_runs(text: str, *, bare_chome: str | None) -> list[str]:
    """The digit runs that name a plot, in the order they were written.

    The 丁目 is a block of the town rather than a plot, so it is removed --
    *where it stands*, not wherever its value occurs. Removing by value is the
    defect to avoid: 丸の内1-1-2 asks about 1丁目1番2号, so filtering out every "1"
    deleted the 番地 as well, and the 号 that survived lined up with the 番地 of
    an answer for 一丁目2番. A different plot would come back as street-number
    precision with only the 号 reported unresolved.

    ``bare_chome`` covers the spelling difference between the two sides: the
    geocoder writes 丸の内一丁目9番 while the query is typed 丸の内1-9-1, so the
    query's leading run is its 丁目 even though nothing says so. It is dropped
    only in that leading position and only when it is the block the answer
    actually named.
    """
    without_chome = _CHOME_TOKEN.sub("", text)
    runs = _DIGIT_RUN.findall(without_chome)
    query_wrote_chome_out = without_chome != text
    if bare_chome is not None and not query_wrote_chome_out and runs and runs[0] == bare_chome:
        return runs[1:]
    return runs


def _has_street_number(rest_matched: str, rest_query: str) -> bool:
    """Whether the answer carries the plot number the query actually asked for.

    Three conditions, all necessary: the same town (``_same_town``), a plot
    number on each side once the 丁目 is taken out (``_plot_runs``), and the
    first of them equal. Ordered from the front, so a trailing 号 may go missing
    without turning a resolved 番地 into a different one -- while set membership
    let a truncated answer pass, 831番地 satisfying a query about 5831 because
    the query's own text contains "831".
    """
    if not _same_town(_town_of(rest_matched), _town_of(rest_query)):
        return False
    chome_token = _CHOME_TOKEN.search(rest_matched)
    chome = _DIGIT_RUN.findall(chome_token.group())[0] if chome_token else None
    carried = _plot_runs(rest_matched, bare_chome=None)
    asked = _plot_runs(rest_query, bare_chome=chome)
    return bool(carried) and bool(asked) and carried[0] == asked[0]


def resolve(query: str, matched: str, municipalities: Mapping[str, str]) -> Resolution:
    """Judge how much of ``query`` the geocoder's ``matched`` answer reached.

    ``municipalities`` maps every accepted spelling to the canonical name, so
    that a county the geocoder dropped is a spelling difference rather than an
    unknown municipality.
    """
    normal_query = normalize(query)
    normal_matched = normalize(matched)
    if not normal_query:
        raise GeocodeError(f"{query!r}: empty address")
    if not normal_matched:
        raise GeocodeError(f"{query!r}: no match")

    municipality, boundary = _municipality_of(normal_matched, municipalities)
    rest_matched = normal_matched[boundary:]
    # The query is cut at its OWN municipality, not at the answer's. One offset
    # for both would make a query typed without its prefecture lose four more
    # characters than it should: 〇〇市〇〇町5831 would become 831, which would
    # then "match" an answer for plot 831 -- a different plot, reported as
    # street-number precision with nothing unresolved.
    query_municipality, query_boundary = _municipality_of_or_none(normal_query, municipalities)
    if query_municipality is not None and query_municipality != municipality:
        # The answer is in a different municipality from the one asked about.
        # Every stored figure is keyed on that code, so continuing would report
        # another town's commute and hazard at plot-level confidence.
        raise GeocodeError(
            f"{query!r} names {query_municipality} but the geocoder answered "
            f"{matched!r} in {municipality}; refusing to answer for a different municipality"
        )
    rest_query = normal_query[query_boundary:]
    if query_municipality is None and not _town_of(rest_query):
        # Nothing in the query names a place: no municipality the reference set
        # holds, and no text before its first number. Whatever the geocoder
        # decided to answer with, it is not an answer to this. Returning it
        # reported plot 1番地 of a town nobody typed at street-number precision.
        raise GeocodeError(
            f"{query!r} names no municipality and no town; refusing to report "
            f"{matched!r} as an answer to it"
        )

    if not rest_matched:
        # Nothing below the municipality was reached, so the whole tail of the
        # query is unaccounted for -- the town name included, not just numbers.
        return Resolution(
            precision=Precision.MUNICIPALITY, municipality=municipality, residue=rest_query
        )

    precision = (
        Precision.STREET_NUMBER if _has_street_number(rest_matched, rest_query) else Precision.TOWN
    )
    return Resolution(
        precision=precision,
        municipality=municipality,
        residue="-".join(_unaccounted(rest_query, rest_matched)),
    )


# A coordinate input, which is a different door into the same answer.
# Strict on purpose: an address never looks like two decimal numbers separated
# by a comma, so anything matching this was meant as a coordinate, and anything
# not matching goes to the geocoder untouched.
_COORDINATE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_coordinate(text: str) -> Match | None:
    """``lat,lon`` typed directly, or ``None`` when this is an address.

    A query is either an address or a **coordinate**. This door stays open
    for the inputs the residue check refuses -- a new development or a
    re-numbered street the reference set has not caught up with has no other
    way in.

    Treated as street-number precision, because nothing was resolved: the
    geocoder is the only thing that can downgrade an answer, and it was never
    asked. A pair outside Japan raises rather than falling back to the
    geocoder: it was typed as a coordinate, and geocoding "35.4,999" would
    answer a question nobody asked.
    """
    found = _COORDINATE.match(text)
    if found is None:
        return None
    lat, lon = float(found.group(1)), float(found.group(2))
    if not (
        JAPAN_LAT_RANGE[0] <= lat <= JAPAN_LAT_RANGE[1]
        and JAPAN_LON_RANGE[0] <= lon <= JAPAN_LON_RANGE[1]
    ):
        raise GeocodeError(
            f"({lat}, {lon}) is outside Japan (lat {JAPAN_LAT_RANGE}, lon {JAPAN_LON_RANGE}); "
            "a transposed lat/lon pair lands here"
        )
    return Match(lat=lat, lon=lon, matched=f"{lat}, {lon}")


def geocode(address: str) -> Match:
    """Return the geocoder's answer for ``address``.

    No match, a malformed answer, or a coordinate outside Japan is an error
    rather than a skipped row: a lookup that quietly returns nothing looks
    exactly like a lookup that found nothing to worry about.
    """
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'q': address})}"
    request = urllib.request.Request(url, headers={"User-Agent": "landeval/0.1"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https host
        body: bytes = response.read()

    try:
        results = json.loads(body)
    except ValueError as exc:
        raise GeocodeError(f"{address}: non-JSON response") from exc

    if not isinstance(results, list) or not results:
        raise GeocodeError(f"{address}: no match")

    best = results[0]
    coordinates = best.get("geometry", {}).get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 2:
        raise GeocodeError(f"{address}: malformed geometry {coordinates!r}")

    lon, lat = float(coordinates[0]), float(coordinates[1])
    if not (
        JAPAN_LAT_RANGE[0] <= lat <= JAPAN_LAT_RANGE[1]
        and JAPAN_LON_RANGE[0] <= lon <= JAPAN_LON_RANGE[1]
    ):
        raise GeocodeError(f"{address}: resolved to ({lat}, {lon}), outside Japan")

    return Match(lat=lat, lon=lon, matched=str(best.get("properties", {}).get("title", "")))
