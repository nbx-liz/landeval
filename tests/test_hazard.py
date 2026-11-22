"""Tests for the hazard overlay.

Two things are being pinned here. First, the parsing of the source vocabularies:
the depth ranges arrive as integer codes in one dataset and as free text in
another, and the free text is not even spelled consistently between the tsunami
and storm-surge layers, so an unrecognised form must raise rather than fall
through to "no hazard".

Second, the arithmetic of the sub-cell sampling, because the whole point of
sampling 25 points per cell rather than testing the centre is that a cell is
allowed to be partly hazardous.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from landeval.hazard import (
    APPLIED_LAYERS_COLUMN,
    DEPTH_RANK_LOWER_M,
    HIT_WITHOUT_DEPTH,
    NO_DATA,
    NOT_DESIGNATED,
    OUTSIDE,
    SUBGRID,
    HazardDataError,
    PointIndex,
    applied_layers,
    coverage_class,
    covered_column,
    depth_from_range_text,
    depth_from_rank_code,
    layer_status,
    max_severity_at_points,
    record_applied_layers,
    split_by_coverage,
    status_layers,
    subcell_lonlat,
    summarise_layer,
    union_ratio,
    unobtained_mask,
)
from landeval.mesh import mesh_bounds, mesh_center

# A 500m cell near a station. The centre is derived rather than typed, so
# the containment test below checks the sampler and not the fixture.
MESH_ID = "533546113"
_CENTRE = mesh_center(MESH_ID)
CELL = pd.DataFrame(
    {
        "mesh_id": pd.array([MESH_ID], dtype="string"),
        "lat": [_CENTRE[0]],
        "lon": [_CENTRE[1]],
    }
)


# --- depth vocabularies ----------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [(1, 0.0), (2, 0.5), (3, 3.0), (4, 5.0), (5, 10.0), (6, 20.0)],
)
def test_every_flood_rank_code_maps_to_its_lower_bound(code: int, expected: float) -> None:
    assert depth_from_rank_code(code) == expected


def test_rank_codes_arriving_as_strings_are_accepted() -> None:
    """Shapefile attribute dtypes vary; the vocabulary does not."""
    assert depth_from_rank_code("3") == 3.0


def test_an_unknown_rank_code_raises_rather_than_reading_as_shallow() -> None:
    with pytest.raises(HazardDataError, match="unknown"):
        depth_from_rank_code(7)


@pytest.mark.parametrize("code", [None, "", "深い", float("nan")])
def test_a_non_numeric_rank_code_raises(code: object) -> None:
    with pytest.raises(HazardDataError, match="unknown"):
        depth_from_rank_code(code)


@pytest.mark.parametrize("value", [None, float("nan")])
def test_a_missing_depth_range_raises(value: object) -> None:
    with pytest.raises(HazardDataError, match="depth"):
        depth_from_range_text(value)


def test_the_rank_table_is_monotonic() -> None:
    bounds = [DEPTH_RANK_LOWER_M[code] for code in sorted(DEPTH_RANK_LOWER_M)]

    assert bounds == sorted(bounds)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A40 (tsunami) spelling: spaces, one decimal place.
        ("0.01m以上 0.3m未満", 0.01),
        ("0.5m以上 1.0m未満", 0.5),
        ("20.0m以上", 20.0),
        # A49 (storm surge) spelling: no space, integers unpadded.
        ("0.5m以上1m未満", 0.5),
        ("10m以上20m未満", 10.0),
        # Open lower bound, only present in A49.
        ("0.3m未満", 0.0),
        # Wave-dash separator, used by some prefectures.
        ("0.01m以上 ～ 0.3m未満", 0.01),
        ("2.0m以上 ～ 5.0m未満", 2.0),
        ("10m以上 ～ 20m未満", 10.0),
        # Leading wave dash for an open lower bound.
        ("～0.3m未満", 0.0),
        # The full-width tilde NFKC folds to ASCII, the wave dash does not.
        ("0.3m以上〜1m未満", 0.3),
    ],
)
def test_both_depth_spellings_parse_to_the_same_lower_bound(text: str, expected: float) -> None:
    assert depth_from_range_text(text) == pytest.approx(expected)


def test_full_width_digits_and_spaces_do_not_silently_fail() -> None:
    assert depth_from_range_text("０.５ｍ以上　１.０ｍ未満") == pytest.approx(0.5)


@pytest.mark.parametrize("text", ["", "不明", "0.5m以上 1.0m以下", "浸水なし", "3.0"])
def test_an_unparseable_depth_raises(text: str) -> None:
    """A depth the parser cannot read must not become 'not inundated'."""
    with pytest.raises(HazardDataError, match="depth"):
        depth_from_range_text(text)


# --- sub-cell sampling -----------------------------------------------------


def test_each_cell_yields_one_point_per_subgrid_square() -> None:
    lon, lat, index = subcell_lonlat(CELL, mesh_size_m=500)

    assert len(lon) == len(lat) == len(index) == SUBGRID**2
    assert set(index) == {0}


def test_sample_points_stay_inside_their_own_cell() -> None:
    south, west, north, east = mesh_bounds(MESH_ID)
    lon, lat, _ = subcell_lonlat(CELL, mesh_size_m=500)

    assert (lat > south).all() and (lat < north).all()
    assert (lon > west).all() and (lon < east).all()


def test_sample_points_are_evenly_spread_rather_than_stacked_on_the_centre() -> None:
    lon, lat, _ = subcell_lonlat(CELL, mesh_size_m=500)

    assert len(np.unique(lon.round(9))) == SUBGRID
    assert len(np.unique(lat.round(9))) == SUBGRID


def test_a_degenerate_subgrid_is_rejected() -> None:
    with pytest.raises(HazardDataError, match="subgrid"):
        subcell_lonlat(CELL, mesh_size_m=500, subgrid=0)


def test_cells_are_indexed_in_order() -> None:
    two = pd.concat([CELL, CELL.assign(lat=35.69)], ignore_index=True)
    _, _, index = subcell_lonlat(two, mesh_size_m=500)

    assert index[0] == 0
    assert index[-1] == 1
    assert (np.bincount(index) == SUBGRID**2).all()


# --- point-in-polygon ------------------------------------------------------


def test_points_outside_every_polygon_are_nan() -> None:
    severity = max_severity_at_points(
        np.array([139.0, 140.0]),
        np.array([35.0, 36.0]),
        [box(138.0, 34.0, 138.5, 34.5)],
        np.array([3.0]),
    )

    assert np.isnan(severity).all()


def test_a_point_inside_takes_the_polygon_severity() -> None:
    severity = max_severity_at_points(
        np.array([139.0]), np.array([35.0]), [box(138.5, 34.5, 139.5, 35.5)], np.array([5.0])
    )

    assert severity[0] == 5.0


def test_overlapping_polygons_yield_the_worst_severity() -> None:
    """Planned-scale and maximum-scale polygons overlap; the deeper one wins."""
    severity = max_severity_at_points(
        np.array([139.0]),
        np.array([35.0]),
        [box(138.5, 34.5, 139.5, 35.5), box(138.9, 34.9, 139.1, 35.1)],
        np.array([0.5, 10.0]),
    )

    assert severity[0] == 10.0


def test_a_zero_severity_polygon_is_still_a_hit() -> None:
    """Depth rank 1 is 0.0 m; it must not be confused with being outside."""
    severity = max_severity_at_points(
        np.array([139.0]), np.array([35.0]), [box(138.5, 34.5, 139.5, 35.5)], np.array([0.0])
    )

    assert severity[0] == 0.0
    assert not np.isnan(severity[0])


def test_an_empty_polygon_set_leaves_every_point_outside() -> None:
    severity = max_severity_at_points(np.array([139.0]), np.array([35.0]), [], np.array([]))

    assert np.isnan(severity).all()


def test_a_point_on_the_boundary_counts_as_inside() -> None:
    """Erring toward hazard: a boundary point must not read as clear."""
    severity = max_severity_at_points(
        np.array([139.0]), np.array([35.0]), [box(139.0, 35.0, 139.5, 35.5)], np.array([2.0])
    )

    assert severity[0] == 2.0


def test_severity_length_must_match_the_geometry_count() -> None:
    with pytest.raises(HazardDataError, match="severit"):
        max_severity_at_points(
            np.array([139.0]),
            np.array([35.0]),
            [Point(139.0, 35.0).buffer(1)],
            np.array([1.0, 2.0]),
        )


# --- streaming index -------------------------------------------------------


def test_accumulating_two_batches_keeps_the_worse_one() -> None:
    """Polygons arrive in chunks; the running worst must survive across them."""
    index = PointIndex(np.array([139.0]), np.array([35.0]))
    out = index.empty_severity()

    index.accumulate([box(138.5, 34.5, 139.5, 35.5)], np.array([0.5]), out)
    index.accumulate([box(138.9, 34.9, 139.1, 35.1)], np.array([10.0]), out)

    assert out[0] == 10.0


def test_a_later_shallower_batch_does_not_overwrite_a_deeper_one() -> None:
    index = PointIndex(np.array([139.0]), np.array([35.0]))
    out = index.empty_severity()

    index.accumulate([box(138.9, 34.9, 139.1, 35.1)], np.array([10.0]), out)
    index.accumulate([box(138.5, 34.5, 139.5, 35.5)], np.array([0.5]), out)

    assert out[0] == 10.0


def test_accumulate_reports_how_many_points_were_hit() -> None:
    """A layer that matched nothing must be distinguishable from one not run."""
    index = PointIndex(np.array([139.0, 145.0]), np.array([35.0, 43.0]))
    out = index.empty_severity()

    hits = index.accumulate([box(138.5, 34.5, 139.5, 35.5)], np.array([1.0]), out)

    assert hits == 1


def test_a_mismatched_accumulator_is_rejected() -> None:
    index = PointIndex(np.array([139.0]), np.array([35.0]))

    with pytest.raises(HazardDataError, match="accumulator"):
        index.accumulate([box(138.5, 34.5, 139.5, 35.5)], np.array([1.0]), np.zeros(5))


def test_mismatched_coordinate_arrays_are_rejected() -> None:
    with pytest.raises(HazardDataError, match="latitude"):
        PointIndex(np.array([139.0, 140.0]), np.array([35.0]))


# --- aggregation -----------------------------------------------------------


def test_a_wholly_clear_cell_has_a_zero_hazard_ratio() -> None:
    result = summarise_layer(
        np.zeros(25, dtype=np.int64), np.full(25, np.nan), n_cells=1, points_per_cell=25
    )

    assert result.loc[0, "hazard_ratio"] == 0.0
    assert result["max_depth_m"].isna().loc[0]


def test_a_partly_flooded_cell_reports_the_fraction() -> None:
    severity = np.full(25, np.nan)
    severity[:5] = 3.0

    result = summarise_layer(np.zeros(25, dtype=np.int64), severity, n_cells=1, points_per_cell=25)

    assert result.loc[0, "hazard_ratio"] == pytest.approx(0.2)
    assert result.loc[0, "max_depth_m"] == 3.0


def test_the_reported_depth_is_the_worst_in_the_cell() -> None:
    severity = np.array([0.5, 10.0, np.nan, np.nan] + [np.nan] * 21)

    result = summarise_layer(np.zeros(25, dtype=np.int64), severity, n_cells=1, points_per_cell=25)

    assert result.loc[0, "max_depth_m"] == 10.0


def test_a_cell_with_no_sampled_points_is_not_reported_as_clear() -> None:
    """A cell the sampler never reached is unknown, not hazard-free."""
    result = summarise_layer(
        np.array([0] * 25, dtype=np.int64), np.full(25, np.nan), n_cells=2, points_per_cell=25
    )

    assert result.loc[0, "hazard_ratio"] == 0.0
    assert result["hazard_ratio"].isna().loc[1]


def test_a_depthless_hit_counts_as_hazard_without_inventing_a_depth() -> None:
    """The house-collapse and landslide layers designate areas but no depth."""
    severity = np.full(25, np.nan)
    severity[:10] = HIT_WITHOUT_DEPTH

    result = summarise_layer(np.zeros(25, dtype=np.int64), severity, n_cells=1, points_per_cell=25)

    assert result.loc[0, "hazard_ratio"] == pytest.approx(0.4)
    assert result["max_depth_m"].isna().loc[0]


def test_cells_are_summarised_independently() -> None:
    index = np.repeat([0, 1], 25)
    severity = np.concatenate([np.full(25, 5.0), np.full(25, np.nan)])

    result = summarise_layer(index, severity, n_cells=2, points_per_cell=25)

    assert result.loc[0, "hazard_ratio"] == 1.0
    assert result.loc[1, "hazard_ratio"] == 0.0


# --- subset union ----------------------------------------------------------


def test_a_union_is_not_the_sum_of_its_layers() -> None:
    """Two layers over the same half of a cell cover half of it, not all of it."""
    index = np.zeros(4, dtype=np.int64)
    # Points 0 and 1 are hit by both layer 0 (bit 1) and layer 1 (bit 2).
    mask = np.array([0b11, 0b11, 0, 0], dtype=np.uint16)

    ratio = union_ratio(
        mask, 0b11, index, n_cells=1, points_per_cell=4, unknown=np.zeros(1, dtype=bool)
    )

    assert ratio[0] == pytest.approx(0.5)


def test_selecting_one_layer_ignores_the_others() -> None:
    index = np.zeros(4, dtype=np.int64)
    mask = np.array([0b01, 0b10, 0b10, 0], dtype=np.uint16)

    assert union_ratio(
        mask, 0b01, index, n_cells=1, points_per_cell=4, unknown=np.zeros(1, dtype=bool)
    )[0] == pytest.approx(0.25)
    assert union_ratio(
        mask, 0b10, index, n_cells=1, points_per_cell=4, unknown=np.zeros(1, dtype=bool)
    )[0] == pytest.approx(0.5)


def test_adding_a_layer_never_shrinks_the_union() -> None:
    index = np.zeros(4, dtype=np.int64)
    mask = np.array([0b01, 0b10, 0b10, 0], dtype=np.uint16)

    one = union_ratio(
        mask, 0b01, index, n_cells=1, points_per_cell=4, unknown=np.zeros(1, dtype=bool)
    )[0]
    both = union_ratio(
        mask, 0b11, index, n_cells=1, points_per_cell=4, unknown=np.zeros(1, dtype=bool)
    )[0]

    assert both >= one
    assert both == pytest.approx(0.75)


def test_selecting_no_layers_is_an_error() -> None:
    with pytest.raises(HazardDataError, match="no layers"):
        union_ratio(
            np.zeros(4, dtype=np.uint16),
            0,
            np.zeros(4, dtype=np.int64),
            n_cells=1,
            points_per_cell=4,
            unknown=np.zeros(1, dtype=bool),
        )


def test_a_mask_that_does_not_match_the_points_is_rejected() -> None:
    with pytest.raises(HazardDataError, match="mask"):
        union_ratio(
            np.zeros(3, dtype=np.uint16),
            0b1,
            np.zeros(4, dtype=np.int64),
            n_cells=1,
            points_per_cell=4,
            unknown=np.zeros(1, dtype=bool),
        )


# --- a cell where a selected layer was never obtained ---------
#
# The membership mask records where polygons *were* found. It cannot record
# where none were looked for: a cell outside every prefecture the tsunami layer
# was fetched for has no bit set, exactly like a cell that was checked and
# found dry. Reading the first as the second is the silent-pass this project
# exists to avoid, so the caller has to say which cells it could not answer for.


def test_a_cell_with_an_unobtained_layer_is_unknown_not_clear() -> None:
    index = np.zeros(4, dtype=np.int64)
    mask = np.zeros(4, dtype=np.uint16)  # no polygon hit anywhere

    ratio = union_ratio(
        mask,
        0b1,
        index,
        n_cells=1,
        points_per_cell=4,
        unknown=np.array([True]),
    )

    assert math.isnan(ratio[0])
    assert coverage_class(ratio[0]) == NO_DATA


def test_an_unobtained_layer_outranks_a_polygon_hit() -> None:
    """Partial knowledge of a cell is still not knowledge of the cell."""
    index = np.zeros(4, dtype=np.int64)
    mask = np.array([0b1, 0b1, 0, 0], dtype=np.uint16)

    ratio = union_ratio(mask, 0b1, index, n_cells=1, points_per_cell=4, unknown=np.array([True]))

    assert math.isnan(ratio[0])


def test_cells_the_caller_could_answer_for_are_unaffected() -> None:
    index = np.array([0, 0, 1, 1], dtype=np.int64)
    mask = np.array([0b1, 0, 0b1, 0], dtype=np.uint16)

    ratio = union_ratio(
        mask,
        0b1,
        index,
        n_cells=2,
        points_per_cell=2,
        unknown=np.array([False, True]),
    )

    assert ratio[0] == pytest.approx(0.5)
    assert math.isnan(ratio[1])


def test_omitting_the_unknown_mask_keeps_the_old_behaviour() -> None:
    index = np.zeros(4, dtype=np.int64)
    mask = np.array([0b1, 0, 0, 0], dtype=np.uint16)

    assert union_ratio(
        mask, 0b1, index, n_cells=1, points_per_cell=4, unknown=np.zeros(1, dtype=bool)
    )[0] == pytest.approx(0.25)


def test_an_unknown_mask_of_the_wrong_length_is_rejected() -> None:
    with pytest.raises(HazardDataError, match="unknown"):
        union_ratio(
            np.zeros(4, dtype=np.uint16),
            0b1,
            np.zeros(4, dtype=np.int64),
            n_cells=2,
            points_per_cell=4,
            unknown=np.array([True]),
        )


def test_the_unknown_mask_cannot_be_omitted() -> None:
    """The whole point of requiring it is that a caller cannot forget it.

    A default would restore the false-clean answer for anyone who did not know
    to ask, which is the shape of gate that disables itself when an input is
    absent. Pinned so the keyword cannot quietly regain a default.
    """
    with pytest.raises(TypeError, match="unknown"):
        union_ratio(  # type: ignore[call-arg]
            np.zeros(4, dtype=np.uint16),
            0b1,
            np.zeros(4, dtype=np.int64),
            n_cells=1,
            points_per_cell=4,
        )


# --- unobtained_mask: the shared per-cell availability rule -----------------
#
# A call site that re-derives this from the status columns can get it wrong, so
# the rule lives in one tested place.


def statuses(**columns: list[str]) -> pd.DataFrame:
    """Status columns, each with the coverage column a build writes beside it.

    A cell reads ``no_data`` exactly where its layer was not read, which is the
    shape every frame in this section stands for.
    """
    frame = pd.DataFrame({f"{name}_status": values for name, values in columns.items()})
    for name, values in columns.items():
        frame[covered_column(name)] = [value != NO_DATA for value in values]
    return frame


def test_a_cell_is_unobtained_when_any_selected_layer_is_no_data() -> None:
    frame = statuses(
        flood_max=[OUTSIDE, OUTSIDE, NO_DATA],
        tsunami=[OUTSIDE, NO_DATA, NO_DATA],
    )

    mask = unobtained_mask(frame, ["flood_max", "tsunami"])

    assert list(mask) == [False, True, True]


def test_a_layer_the_caller_did_not_select_does_not_make_a_cell_unobtained() -> None:
    """Which layers are applied is a display-time choice, and so is this."""
    frame = statuses(flood_max=[OUTSIDE], tsunami=[NO_DATA])

    assert list(unobtained_mask(frame, ["flood_max"])) == [False]


def test_not_designated_is_an_answer_and_not_a_gap() -> None:
    """A prefecture that publishes no such zone has been asked; a gap has not."""
    frame = statuses(tsunami=[NOT_DESIGNATED])

    assert list(unobtained_mask(frame, ["tsunami"])) == [False]


def test_a_missing_status_column_is_rejected_rather_than_read_as_obtained() -> None:
    """Without the column the question cannot be answered, so it must not be."""
    frame = statuses(flood_max=[OUTSIDE])

    with pytest.raises(HazardDataError, match="tsunami_status"):
        unobtained_mask(frame, ["flood_max", "tsunami"])


def test_a_null_status_is_rejected_rather_than_read_as_obtained() -> None:
    """A cell with no row in the hazard table at all.

    It arrives as NaN from a left merge that did not match, and ``== "no_data"``
    reads NaN as False -- obtained, nothing found. The original fault wearing a
    different coat, one level up.
    """
    frame = pd.DataFrame({"flood_max_status": [OUTSIDE, None]})

    with pytest.raises(HazardDataError, match="null"):
        unobtained_mask(frame, ["flood_max"])


def test_selecting_no_layers_is_rejected() -> None:
    with pytest.raises(HazardDataError, match="no layers"):
        unobtained_mask(statuses(flood_max=[OUTSIDE]), [])


def test_a_layer_obtained_for_some_cells_is_covered_and_flags_only_the_gaps() -> None:
    frame = statuses(tsunami=[OUTSIDE, NO_DATA])

    covered, absent = split_by_coverage(frame, ["tsunami"])

    assert (covered, absent) == (["tsunami"], [])
    assert list(unobtained_mask(frame, covered)) == [False, True]


def test_a_layer_obtained_for_no_cell_at_all_is_a_map_wide_caveat() -> None:
    """A layer obtained for no cell is a statement about the whole map.

    Folding that into the per-cell flag marks every cell unanswerable, which
    tells the reader nothing and buries the cells that really do have a gap.
    """
    frame = statuses(flood_max=[OUTSIDE, NO_DATA], inland_flood=[NO_DATA, NO_DATA])

    covered, absent = split_by_coverage(frame, ["flood_max", "inland_flood"])

    assert (covered, absent) == (["flood_max"], ["inland_flood"])
    assert list(unobtained_mask(frame, covered)) == [False, True]


def test_split_by_coverage_rejects_a_layer_with_no_status_column() -> None:
    with pytest.raises(HazardDataError, match="tsunami_status"):
        split_by_coverage(statuses(flood_max=[OUTSIDE]), ["tsunami"])


# --- applied_layers: the selection travels with the result ------------------


def test_the_applied_selection_round_trips_through_the_frame() -> None:
    frame = pd.DataFrame({APPLIED_LAYERS_COLUMN: [record_applied_layers(["flood_max", "tsunami"])]})

    assert applied_layers(frame) == ("flood_max", "tsunami")


def test_a_frame_that_never_recorded_the_selection_is_rejected() -> None:
    """Not defaulted to the usual three.

    Which layers to apply is a `--layer` argument. A parquet written
    before the choice was recorded cannot say what it was, and assuming the
    default is how a selection made over other layers gets read as the usual
    one -- the drift hardcoded copies of the default would cause.
    """
    with pytest.raises(HazardDataError, match=APPLIED_LAYERS_COLUMN):
        applied_layers(pd.DataFrame({"mesh_id": ["x"]}))


def test_a_frame_with_two_different_selections_is_rejected() -> None:
    frame = pd.DataFrame({APPLIED_LAYERS_COLUMN: ["flood_max", "flood_max|tsunami"]})

    with pytest.raises(HazardDataError, match="not constant"):
        applied_layers(frame)


def test_recording_an_empty_selection_is_rejected() -> None:
    with pytest.raises(HazardDataError, match="no layers"):
        record_applied_layers([])


def test_status_layers_lists_every_layer_the_frame_carries() -> None:
    frame = statuses(flood_max=[OUTSIDE], tsunami=[NO_DATA])

    assert status_layers(frame) == ["flood_max", "tsunami"]


def test_status_layers_rejects_a_frame_that_carries_none() -> None:
    with pytest.raises(HazardDataError, match=r"no \*_status"):
        status_layers(pd.DataFrame({"mesh_id": ["x"]}))


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(0.0, "clear"), (0.04, "partial"), (0.96, "partial"), (1.0, "covered")],
)
def test_the_three_way_split_follows_the_edges(ratio: float, expected: str) -> None:
    assert coverage_class(ratio) == expected


def test_an_unsampled_cell_is_not_split_into_any_of_the_three() -> None:
    assert coverage_class(float("nan")) == NO_DATA


# --- status ----------------------------------------------------------------


def test_an_unobtainable_layer_is_no_data_not_outside() -> None:
    """Inland flooding has no nationwide dataset; silence is not safety."""
    assert layer_status(0.0, np.nan, available=False, designated=False) == NO_DATA


def test_a_layer_that_does_not_apply_here_is_not_designated() -> None:
    assert layer_status(0.0, np.nan, available=True, designated=False) == NOT_DESIGNATED


def test_a_clear_cell_is_reported_as_outside_the_designated_area() -> None:
    """Not 'safe' -- the polygons only cover designated rivers."""
    assert layer_status(0.0, np.nan, available=True, designated=True) == OUTSIDE


def test_a_hit_cell_carries_its_worst_rank() -> None:
    assert layer_status(0.4, 3.0, available=True, designated=True) == "risk_3.0m"


def test_an_unsampled_cell_is_no_data() -> None:
    assert layer_status(float("nan"), np.nan, available=True, designated=True) == NO_DATA


def test_a_hit_without_a_depth_still_reads_as_risk() -> None:
    """The house-collapse layer has no depth attribute at all."""
    assert layer_status(0.1, np.nan, available=True, designated=True) == "risk"


def test_no_status_is_the_literal_word_safe() -> None:
    """Guard against reintroducing a safety claim the data cannot support."""
    statuses = {
        layer_status(0.0, np.nan, available=False, designated=False),
        layer_status(0.0, np.nan, available=True, designated=False),
        layer_status(0.0, np.nan, available=True, designated=True),
        layer_status(0.5, 5.0, available=True, designated=True),
    }

    assert "safe" not in statuses
