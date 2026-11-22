"""Deriving how much of an address the geocoder actually resolved.

A geocoder that cannot find a street number does not say so. It returns the
town, or the ward centroid, with the same shape of answer and the same
confidence -- which is the false-clean shape: a coarser answer wearing the clothes of a
precise one. Everything downstream keys off this: a street-number resolution
gets a point estimate and a 100 m hazard sample, a coarser one gets a
distribution over the area.

So the question these tests pin is not "where is it" but "how much of what I
asked for came back".
"""

from __future__ import annotations

import pytest

from landeval.geocode import (
    GeocodeError,
    Precision,
    normalize,
    parse_coordinate,
    resolve,
)

# The reference set maps every accepted spelling to the canonical name. Real
# entries, so a passing test says something about real input.
MUNICIPALITIES = {
    name: name
    for name in (
        "愛知県岡崎市",
        "愛知県豊橋市",
        "青森県上北郡六ケ所村",
        "愛知県名古屋市中区",
        "愛知県名古屋市東区",
        "愛知県名古屋市港区",
    )
} | {
    # The county-less spelling the geocoder actually returns.
    "愛知県額田郡幸田町": "愛知県額田郡幸田町",
    "愛知県幸田町": "愛知県額田郡幸田町",
    # A query may be typed without its prefecture. These entries have the
    # shape `admin.alias_map` produces; a fixture that invented its own would
    # let the case below pass while the builder refused the address
    # (test_admin.py pins the builder itself).
    "豊橋市": "愛知県豊橋市",
    "久留米市": "福岡県久留米市",
    "豊田市": "愛知県豊田市",
    "福岡県久留米市": "福岡県久留米市",
    "愛知県豊田市": "愛知県豊田市",
}


# --- normalize -------------------------------------------------------------


def test_full_width_digits_become_ascii() -> None:
    """The geocoder answers in full width; the query is typed in half width."""
    assert normalize("石川１２３４") == "石川1234"


def test_whitespace_is_removed() -> None:
    assert normalize("愛知県 岡崎市 石川") == "愛知県岡崎市石川"


def test_the_two_spellings_of_ke_are_folded() -> None:
    """六ヶ所村 and 六ケ所村 are the same place; the boundary set picks one."""
    assert normalize("青森県上北郡六ヶ所村") == normalize("青森県上北郡六ケ所村")


@pytest.mark.parametrize(
    ("kanji", "digits"),
    [("一丁目", "1丁目"), ("九丁目", "9丁目"), ("十丁目", "10丁目"), ("十一丁目", "11丁目")],
)
def test_a_chome_written_in_kanji_becomes_the_same_digits(kanji: str, digits: str) -> None:
    """丁目 above ten exists in some towns, so the
    reading cannot stop at nine."""
    assert normalize(f"栄{kanji}") == f"栄{digits}"


def test_oaza_is_dropped_because_the_geocoder_omits_it() -> None:
    assert normalize("愛知県豊橋市大字大町") == "愛知県豊橋市大町"


# --- precision -------------------------------------------------------------


def test_a_resolved_street_number_is_street_number_precision() -> None:
    resolution = resolve("愛知県岡崎市石川1234", "愛知県岡崎市石川１２３４番地", MUNICIPALITIES)

    assert resolution.precision is Precision.STREET_NUMBER
    assert resolution.municipality == "愛知県岡崎市"
    assert resolution.residue == ""


def test_a_dropped_street_number_degrades_to_town_precision() -> None:
    """The geocoder can answer a plot query with the town only. Nothing in the
    response says so."""
    resolution = resolve("愛知県豊橋市大町5831", "愛知県豊橋市大町", MUNICIPALITIES)

    assert resolution.precision is Precision.TOWN
    assert resolution.residue == "5831"


def test_a_ward_centroid_answer_is_municipality_precision() -> None:
    """The failure the whole module exists for: the geocoder falls back to the
    municipality and the coordinate looks just as authoritative."""
    resolution = resolve("愛知県豊橋市大町5831", "愛知県豊橋市", MUNICIPALITIES)

    assert resolution.precision is Precision.MUNICIPALITY
    assert resolution.residue == "大町5831"


def test_a_chome_number_is_not_a_street_number() -> None:
    """4丁目 names a block of the town, not a plot. Treating it as a street
    number would put a 100 m hazard sample on a point nobody asked about."""
    resolution = resolve("愛知県名古屋市中区栄4丁目", "愛知県名古屋市中区栄４丁目", MUNICIPALITIES)

    assert resolution.precision is Precision.TOWN


def test_a_street_number_after_a_chome_is_still_a_street_number() -> None:
    resolution = resolve(
        "愛知県名古屋市中区栄4丁目1番", "愛知県名古屋市中区栄４丁目１番", MUNICIPALITIES
    )

    assert resolution.precision is Precision.STREET_NUMBER


def test_a_dropped_trailing_go_does_not_degrade_the_precision() -> None:
    """The 番地 resolved and only the 号 was dropped. The point is the plot,
    which is what a 100 m sample needs; the residue still reports the loss."""
    resolution = resolve("愛知県岡崎市石川1234-5", "愛知県岡崎市石川１２３４番地", MUNICIPALITIES)

    assert resolution.precision is Precision.STREET_NUMBER
    assert resolution.residue == "5"


def test_a_street_number_the_query_never_asked_for_is_not_a_match() -> None:
    """The geocoder answering 1235 for 1234 is a different plot, and calling
    that street-number precision would claim a number the user never gave."""
    resolution = resolve("愛知県岡崎市石川1234", "愛知県岡崎市石川１２３５番地", MUNICIPALITIES)

    assert resolution.precision is Precision.TOWN


def test_a_town_with_no_number_anywhere_is_town_precision() -> None:
    resolution = resolve("愛知県豊橋市大町", "愛知県豊橋市大町", MUNICIPALITIES)

    assert resolution.precision is Precision.TOWN
    assert resolution.residue == ""


def test_a_bare_municipality_is_municipality_precision() -> None:
    resolution = resolve("愛知県岡崎市", "愛知県岡崎市", MUNICIPALITIES)

    assert resolution.precision is Precision.MUNICIPALITY


# --- the reference set -----------------------------------------------------


def test_a_municipality_outside_the_reference_set_is_an_error() -> None:
    """A gap in the reference set is an exception. Guessing the municipality from the string
    would put the answer on a boundary set that never contained it."""
    with pytest.raises(GeocodeError, match="reference"):
        resolve("北海道札幌市中央区北1条西2", "北海道札幌市中央区北１条西２", MUNICIPALITIES)


def test_the_longest_municipality_name_wins() -> None:
    """A designated city and its wards share a prefix; the ward is the answer."""
    reference = {"愛知県名古屋市": "愛知県名古屋市", "愛知県名古屋市中区": "愛知県名古屋市中区"}

    resolution = resolve("愛知県名古屋市中区栄1", "愛知県名古屋市中区栄１番", reference)

    assert resolution.municipality == "愛知県名古屋市中区"


def test_an_empty_reference_set_is_an_error_rather_than_a_pass() -> None:
    with pytest.raises(GeocodeError, match="reference"):
        resolve("愛知県岡崎市石川1234", "愛知県岡崎市石川１２３４番地", {})


def test_a_county_the_geocoder_dropped_still_resolves_to_the_canonical_name() -> None:
    """The geocoder drops the county: 〇〇郡〇〇町 comes back as 〇〇町.
    Rejecting that as an unknown municipality is a
    false negative wearing the clothes of a strict check -- the address resolved
    fine and only the spelling differs."""
    resolution = resolve("愛知県額田郡幸田町相見639", "愛知県幸田町相見６３９番地", MUNICIPALITIES)

    assert resolution.municipality == "愛知県額田郡幸田町"
    assert resolution.precision is Precision.STREET_NUMBER


# --- what the answer accounted for -----------------------------------------


def test_a_chome_in_kanji_is_the_same_block_as_one_in_digits() -> None:
    """The geocoder writes the 丁目 in kanji and can stop at the 番, as in
    丸の内一丁目９番 for 丸の内1-9-1. Only the 号 went missing, so only the 号 is reported --
    reading the tail as unresolved would call the 6 lost when the answer names
    it."""
    resolution = resolve(
        "愛知県名古屋市中区丸の内1-9-1", "愛知県名古屋市中区丸の内一丁目９番", MUNICIPALITIES
    )

    assert resolution.precision is Precision.STREET_NUMBER
    assert resolution.residue == "1"


def test_an_empty_match_is_an_error() -> None:
    """No match is not a coarse match."""
    with pytest.raises(GeocodeError, match="no match"):
        resolve("愛知県岡崎市石川1234", "", MUNICIPALITIES)


# --- the gate downstream reads ---------------------------------------------


def test_only_street_number_precision_opens_the_point_estimate() -> None:
    """The measured path is gated on street-number precision, and the
    firing rate of that gate is a risk: a gate whose condition real input never meets is dead."""
    assert Precision.STREET_NUMBER.is_point is True
    assert Precision.TOWN.is_point is False
    assert Precision.MUNICIPALITY.is_point is False


# --- wrong answers the basic cases do not catch --------------------------------
#
# Each case below reproduces a defect that the cases above do not catch.


def test_a_query_without_its_prefecture_is_cut_at_its_own_municipality() -> None:
    """The query is cut at the length of its own alias, not the alias matched
    in the ANSWER. Typed without its prefecture, 豊橋市大町5831 must not lose
    four more characters than it should and become '831'."""
    resolution = resolve("豊橋市大町5831", "愛知県豊橋市大町5831番地", MUNICIPALITIES)

    assert resolution.precision is Precision.STREET_NUMBER
    assert resolution.residue == ""


def test_a_truncated_plot_number_is_not_the_plot_that_was_asked_for() -> None:
    """The dangerous half of the same defect: when the geocoder answers plot 831
    for a query about 5831, an offset slice would make '831' match '831'. The
    result would be street-number precision with nothing reported unresolved --
    a point estimate and a per-cell hazard reading for a plot nobody asked
    about."""
    resolution = resolve("豊橋市大町5831", "愛知県豊橋市大町831番地", MUNICIPALITIES)

    assert resolution.precision is Precision.TOWN


def test_an_answer_in_a_different_municipality_is_refused() -> None:
    """Every stored figure is keyed on the municipality code. Continuing would
    report another town's commute and hazard at plot-level confidence."""
    with pytest.raises(GeocodeError, match="different municipality"):
        resolve("福岡県久留米市大和田1-2", "愛知県豊田市大和田１番", MUNICIPALITIES)


def test_a_plot_number_in_a_different_town_is_not_a_match() -> None:
    """A digit run of the query that is not a 丁目 must stand where the answer
    puts its plot number: an answer for 辻堂5番 does not satisfy a query about
    石川1234-5."""
    resolution = resolve("愛知県岡崎市石川1234-5", "愛知県岡崎市辻堂５番", MUNICIPALITIES)

    assert resolution.precision is not Precision.STREET_NUMBER


# --- wrong answers the cases above still let through ---------------------------
#
# Each case below reproduces a defect that the cases above, including the
# wrong-answer cases, do not catch.


def test_a_chome_number_repeated_as_the_plot_does_not_admit_a_different_plot() -> None:
    """Filtering the 丁目 out by *value* would make a query about 1丁目1番2号 lose
    its 番地 as well, and the surviving 号 would line up with an answer for 1丁目2番.
    A different plot would come back at street-number precision, which opens the
    point estimate and the per-cell hazard reading for ground nobody asked about."""
    resolution = resolve(
        "愛知県名古屋市中区丸の内1-1-2", "愛知県名古屋市中区丸の内一丁目２番", MUNICIPALITIES
    )

    assert resolution.precision is not Precision.STREET_NUMBER


def test_a_chome_number_repeated_as_the_plot_still_resolves_when_it_is_right() -> None:
    """The same defect would refuse a correct answer: 1丁目1番 for 1-1-1 is
    exactly what was asked for, and filtering every "1" leaves nothing to compare."""
    resolution = resolve(
        "愛知県名古屋市中区丸の内1-1-1", "愛知県名古屋市中区丸の内一丁目１番", MUNICIPALITIES
    )

    assert resolution.precision is Precision.STREET_NUMBER
    assert resolution.residue == "1"


def test_a_postal_code_pasted_in_front_is_not_part_of_the_address() -> None:
    """A listing is pasted whole. Reading 〒460-0001 as address digits would
    demote the answer to town precision and report '460-0001' as the part that
    did not resolve."""
    resolution = resolve(
        "〒460-0001 愛知県名古屋市中区丸の内1-9-1",
        "愛知県名古屋市中区丸の内一丁目９番",
        MUNICIPALITIES,
    )

    assert resolution.precision is Precision.STREET_NUMBER
    assert resolution.residue == "1"


def test_the_three_spellings_of_the_ke_morpheme_are_one_town() -> None:
    """雪ケ谷 and 雪が谷 are the same place: every spelling of the morpheme folds."""
    resolution = resolve(
        "愛知県名古屋市東区雪ケ谷2-1-1", "愛知県名古屋市東区雪が谷二丁目１番", MUNICIPALITIES
    )

    assert resolution.precision is Precision.STREET_NUMBER


def test_a_hiragana_ga_inside_a_name_is_left_alone() -> None:
    """The fold is between two non-hiragana characters, so みどりが丘 keeps its
    が on both sides rather than becoming みどりケ丘."""
    assert normalize("みどりが丘") == "みどりが丘"


@pytest.mark.parametrize("query", ["愛知県岡崎市石川字向台1234", "愛知県岡崎市石川甲1234"])
def test_detail_below_the_town_that_the_geocoder_drops_is_not_a_different_town(
    query: str,
) -> None:
    """字 and the 甲乙丙 parcel letters are written by people and omitted by the
    geocoder. Addresses carry them, and an exact town comparison would refuse
    them."""
    resolution = resolve(query, "愛知県岡崎市石川１２３４番地", MUNICIPALITIES)

    assert resolution.precision is Precision.STREET_NUMBER


def test_a_town_whose_name_merely_starts_the_same_is_still_a_different_town() -> None:
    """Which is why the tolerance above is a closed list rather than a prefix:
    石川台 is not 石川 written in more detail."""
    resolution = resolve("愛知県岡崎市石川台1234", "愛知県岡崎市石川１２３４番地", MUNICIPALITIES)

    assert resolution.precision is not Precision.STREET_NUMBER


def test_a_query_that_names_no_place_at_all_is_refused() -> None:
    """A town guard written `if town_matched and town_query and ...` skips
    itself when the query names no town, and '1-1' would come back at
    street-number precision for whatever plot the geocoder chose."""
    with pytest.raises(GeocodeError, match="no municipality and no town"):
        resolve("1-1", "愛知県豊橋市大町１番地", MUNICIPALITIES)


@pytest.mark.parametrize("query", ["", "  　 "])
def test_an_empty_query_is_an_error_rather_than_an_answer(query: str) -> None:
    with pytest.raises(GeocodeError, match="empty address"):
        resolve(query, "愛知県豊橋市大町１番地", MUNICIPALITIES)


def test_the_plot_numbers_have_to_line_up_from_the_front() -> None:
    """Ordered from the first plot run, so a trailing 号 may go missing without
    turning a resolved 番地 into a different one."""
    resolution = resolve("愛知県岡崎市石川1234-5", "愛知県岡崎市石川１２３４番地", MUNICIPALITIES)

    assert resolution.precision is Precision.STREET_NUMBER
    assert resolution.residue == "5"


# --- the coordinate door --------------------


def test_a_coordinate_pair_is_read_as_a_point() -> None:
    """A query is an address *or* a coordinate, and the residue check refuses
    new developments the reference set has not caught up with. A coordinate is
    their only way in."""
    match = parse_coordinate("43.4, 142.5")

    assert match is not None
    assert (match.lat, match.lon) == (43.4, 142.5)


def test_an_address_is_not_mistaken_for_a_coordinate() -> None:
    assert parse_coordinate("愛知県岡崎市本町512-3") is None


def test_a_number_that_is_not_a_pair_is_an_address() -> None:
    """`3-5` is a street number, not a coordinate. Reading it as one would
    answer about a point in the ocean."""
    assert parse_coordinate("3-5") is None


def test_a_coordinate_outside_japan_is_an_error_not_an_address() -> None:
    """It was typed as a coordinate. Handing "43.4,999" to the geocoder would
    answer a question nobody asked."""
    with pytest.raises(GeocodeError, match="outside Japan"):
        parse_coordinate("43.4, 999")


def test_a_transposed_pair_is_refused() -> None:
    """142.5, 43.4 is the same place with the two numbers swapped, and it lands
    outside the box for exactly that reason."""
    with pytest.raises(GeocodeError, match="outside Japan"):
        parse_coordinate("142.5, 43.4")
