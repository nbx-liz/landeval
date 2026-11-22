"""Tests for direction-specific filter thresholds."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from landeval.calibration import (
    ABOVE_RANGE,
    BELOW_RANGE,
    INSIDE_RANGE,
    MIN_POINTS_PER_DIRECTION,
    Fit,
    classify_cells,
    direction_thresholds,
    fit_directions,
    fit_line,
    fitted_ranges,
    pooled_sigma,
    range_sides,
)

# Neutral values for these tests. The thresholds and sigma margins are the caller's
# choice; nothing here is a value any real screening uses.
SCREENING_GOOD_MAX_MIN = 80.0
SCREENING_EXCLUDE_MIN = 130.0
GOOD_SIGMA_MARGIN = 0.5
EXCLUDE_SIGMA_MARGIN = 1.5

DEST = (35.680000, 139.760000)


def num(value: object) -> float:
    """Narrow a pandas scalar to float; its stub type is a wide union."""
    return float(value)  # type: ignore[arg-type]


def calibration_rows(sector: int, slope: float, intercept: float, n: int = 9) -> pd.DataFrame:
    """Points lying exactly on a line, so the fit is known."""
    freeflow = [10.0 + 10.0 * i for i in range(n)]
    # North-ish coordinates; the sector column is what the fitting uses.
    return pd.DataFrame(
        {
            "sector": [sector] * n,
            "schedule_id": ["arr_0930"] * n,
            "freeflow_min": freeflow,
            "measured_min": [slope * f + intercept for f in freeflow],
            "lat": [36.0 + 0.01 * i for i in range(n)],
            "lon": [139.76] * n,
        }
    )


# --- fitting ---------------------------------------------------------------


def test_fit_recovers_a_known_line() -> None:
    fit = fit_line(calibration_rows(0, 1.2, 20.0))

    assert fit.slope == pytest.approx(1.2)
    assert fit.intercept == pytest.approx(20.0)
    assert fit.sigma == pytest.approx(0.0, abs=1e-9)


def test_too_few_points_yields_an_unusable_fit() -> None:
    fit = fit_line(calibration_rows(0, 1.2, 20.0, n=2))

    assert not fit.is_usable
    assert math.isnan(fit.slope)


def test_pooled_sigma_is_nan_without_degrees_of_freedom() -> None:
    assert math.isnan(pooled_sigma({}, {}))


# --- inversion -------------------------------------------------------------


def test_inversion_is_the_inverse_of_prediction() -> None:
    fit = Fit(slope=1.2, intercept=20.0, sigma=5.0, n=9)

    assert fit.at(fit.osrm_for(90.0)) == pytest.approx(90.0)


def test_a_sigma_margin_moves_the_edge_outward() -> None:
    """Widening must keep more cells, never fewer."""
    fit = Fit(slope=1.2, intercept=20.0, sigma=10.0, n=9)

    assert fit.osrm_for(120.0, sigma_margin=1.0) > fit.osrm_for(120.0)


def test_a_steeper_direction_gets_a_tighter_edge() -> None:
    """The worse the direction, the sooner the filter cuts."""
    steep = Fit(slope=1.33, intercept=28.0, sigma=0.0, n=9)
    shallow = Fit(slope=1.03, intercept=13.0, sigma=0.0, n=9)

    assert steep.osrm_for(120.0) < shallow.osrm_for(120.0)


def test_inverting_an_unusable_fit_raises() -> None:
    with pytest.raises(ValueError, match="unusable"):
        Fit(float("nan"), float("nan"), float("nan"), 0).osrm_for(90.0)


# --- direction model -------------------------------------------------------


def two_sector_calibration() -> pd.DataFrame:
    return pd.concat(
        [calibration_rows(0, 1.33, 28.0), calibration_rows(3, 1.03, 13.0)], ignore_index=True
    )


def test_each_direction_gets_its_own_fit() -> None:
    model = fit_directions(two_sector_calibration(), "arr_0930")

    assert model.fit_for(0).slope == pytest.approx(1.33)
    assert model.fit_for(3).slope == pytest.approx(1.03)


def test_an_unmeasured_direction_falls_back_to_the_pooled_fit() -> None:
    model = fit_directions(two_sector_calibration(), "arr_0930")

    assert model.uses_fallback(7)
    assert model.fit_for(7) is model.overall


def test_a_thin_direction_falls_back() -> None:
    thin = pd.concat(
        [calibration_rows(0, 1.33, 28.0), calibration_rows(3, 1.03, 13.0, n=3)],
        ignore_index=True,
    )
    model = fit_directions(thin, "arr_0930")

    assert MIN_POINTS_PER_DIRECTION > 3
    assert model.uses_fallback(3)


def test_an_unknown_schedule_raises() -> None:
    with pytest.raises(ValueError, match="no calibration rows"):
        fit_directions(two_sector_calibration(), "depart_9999")


# --- thresholds ------------------------------------------------------------


def test_thresholds_are_ordered_within_each_direction() -> None:
    model = fit_directions(two_sector_calibration(), "arr_0930")

    table = direction_thresholds(
        model,
        good_max_min=SCREENING_GOOD_MAX_MIN,
        exclude_min=SCREENING_EXCLUDE_MIN,
        good_sigma_margin=GOOD_SIGMA_MARGIN,
        exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
    )
    good = table["osrm_good_max"].astype(float)
    exclude = table["osrm_exclude"].astype(float)
    assert (good < exclude).all()


def test_a_worse_direction_gets_a_lower_exclusion_edge() -> None:
    model = fit_directions(two_sector_calibration(), "arr_0930")
    table = direction_thresholds(
        model,
        good_max_min=SCREENING_GOOD_MAX_MIN,
        exclude_min=SCREENING_EXCLUDE_MIN,
        good_sigma_margin=GOOD_SIGMA_MARGIN,
        exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
    ).set_index("sector")

    assert num(table.loc[0, "osrm_exclude"]) < num(table.loc[3, "osrm_exclude"])


def test_every_sector_gets_an_edge() -> None:
    """A sector with no measurements must still be covered, via the fallback."""
    model = fit_directions(two_sector_calibration(), "arr_0930")

    assert (
        len(
            direction_thresholds(
                model,
                good_max_min=SCREENING_GOOD_MAX_MIN,
                exclude_min=SCREENING_EXCLUDE_MIN,
                good_sigma_margin=GOOD_SIGMA_MARGIN,
                exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
            )
        )
        == model.sectors
    )


def test_the_exclusion_edge_carries_the_sigma_margin() -> None:
    noisy = calibration_rows(0, 1.2, 20.0).copy()
    noisy.loc[0, "measured_min"] = num(noisy.loc[0, "measured_min"]) + 30.0
    model = fit_directions(noisy, "arr_0930")
    row = (
        direction_thresholds(
            model,
            good_max_min=SCREENING_GOOD_MAX_MIN,
            exclude_min=SCREENING_EXCLUDE_MIN,
            good_sigma_margin=GOOD_SIGMA_MARGIN,
            exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
        )
        .set_index("sector")
        .loc[0]
    )

    # The same target as the call above, so the comparison isolates the margin.
    plain = model.fit_for(0).osrm_for(SCREENING_EXCLUDE_MIN)
    assert num(row["osrm_exclude"]) > plain
    assert EXCLUDE_SIGMA_MARGIN > 0


# --- classification --------------------------------------------------------


def cells(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    """rows: (mesh_id, lat, lon, freeflow_min)."""
    return pd.DataFrame(
        {
            "mesh_id": pd.array([r[0] for r in rows], dtype="string"),
            "lat": [r[1] for r in rows],
            "lon": [r[2] for r in rows],
            "freeflow_min": pd.array([r[3] for r in rows], dtype="Float64"),
        }
    )


def test_classification_uses_the_direction_of_the_cell() -> None:
    """The same OSRM value lands in different classes depending on direction."""
    model = fit_directions(two_sector_calibration(), "arr_0930")
    # sector 0 is north (slope 1.33), sector 3 is east (slope 1.03).
    frame = cells([("n", 36.4, 139.76, 60.0), ("e", 35.680000, 140.4, 60.0)])

    result = classify_cells(
        frame,
        DEST,
        model,
        good_max_min=SCREENING_GOOD_MAX_MIN,
        exclude_min=SCREENING_EXCLUDE_MIN,
        good_sigma_margin=GOOD_SIGMA_MARGIN,
        exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
    )

    north = result[result["mesh_id"] == "n"].iloc[0]
    east = result[result["mesh_id"] == "e"].iloc[0]
    assert num(north["predicted_min"]) > num(east["predicted_min"])
    assert north["screening_class"] != "pass"
    assert east["screening_class"] == "pass"


def test_classes_follow_the_edges() -> None:
    model = fit_directions(two_sector_calibration(), "arr_0930")
    edges = direction_thresholds(
        model,
        good_max_min=SCREENING_GOOD_MAX_MIN,
        exclude_min=SCREENING_EXCLUDE_MIN,
        good_sigma_margin=GOOD_SIGMA_MARGIN,
        exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
    ).set_index("sector")
    good = num(edges.loc[3, "osrm_good_max"])
    exclude = num(edges.loc[3, "osrm_exclude"])

    frame = cells(
        [
            ("pass", 35.680000, 140.4, good - 1),
            ("hold", 35.680000, 140.4, good + 1),
            ("far", 35.680000, 140.4, exclude + 1),
        ]
    )
    result = classify_cells(
        frame,
        DEST,
        model,
        good_max_min=SCREENING_GOOD_MAX_MIN,
        exclude_min=SCREENING_EXCLUDE_MIN,
        good_sigma_margin=GOOD_SIGMA_MARGIN,
        exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
    ).set_index("mesh_id")

    assert result.loc["pass", "screening_class"] == "pass"
    assert result.loc["hold", "screening_class"] == "hold"
    assert result.loc["far", "screening_class"] == "exclude"


def test_prediction_is_reported_alongside_the_class() -> None:
    model = fit_directions(two_sector_calibration(), "arr_0930")
    result = classify_cells(
        cells([("e", 35.680000, 140.4, 60.0)]),
        DEST,
        model,
        good_max_min=SCREENING_GOOD_MAX_MIN,
        exclude_min=SCREENING_EXCLUDE_MIN,
        good_sigma_margin=GOOD_SIGMA_MARGIN,
        exclude_sigma_margin=EXCLUDE_SIGMA_MARGIN,
    )

    assert result.iloc[0]["predicted_min"] == pytest.approx(1.03 * 60 + 13.0)


# --- the range each sector's line was actually fitted from ---------


def calibration_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sector": 3, "freeflow_min": 22.0, "measured_min": 34.0},
            {"sector": 3, "freeflow_min": 74.0, "measured_min": 88.0},
            {"sector": 7, "freeflow_min": 8.0, "measured_min": 15.0},
            {"sector": 7, "freeflow_min": 55.0, "measured_min": 63.0},
        ]
    )


def test_each_sector_reports_both_ends_of_what_it_was_fitted_over() -> None:
    assert fitted_ranges(calibration_points()) == {3: (22.0, 74.0), 7: (8.0, 55.0)}


def test_the_ceiling_is_kept_because_the_domain_has_two_ends() -> None:
    """A check that guarded only the floor while calling itself a range would
    answer a cell above its sector's ceiling as though the line had been
    fitted there."""
    _, ceiling = fitted_ranges(calibration_points())[3]

    assert ceiling == pytest.approx(74.0)


def test_the_ranges_are_per_sector_because_the_sectors_differ() -> None:
    """Fitted floors can differ widely between sectors. One global
    floor would wave through extrapolation in every sector whose own floor is
    higher. This fixture only has to show that two sectors get two answers."""
    ranges = fitted_ranges(calibration_points())

    assert ranges[3] != ranges[7]


def test_no_calibration_points_is_an_error_rather_than_an_empty_range() -> None:
    """An empty mapping would make every sector unknown, and a caller reading
    "not below the floor" from a missing floor lets every cell through."""
    with pytest.raises(ValueError, match="no calibration points"):
        fitted_ranges(calibration_points().iloc[0:0])


def test_a_null_free_flow_is_an_error() -> None:
    frame = calibration_points()
    frame.loc[0, "freeflow_min"] = math.nan

    with pytest.raises(ValueError, match="null"):
        fitted_ranges(frame)


def test_calibration_points_without_a_sector_are_an_error() -> None:
    with pytest.raises(ValueError, match="no sector"):
        fitted_ranges(calibration_points().drop(columns=["sector"]))


# --- one range check, asked the same way by the map and the lookup --


def test_a_cell_inside_its_sector_s_span_gets_an_empty_side() -> None:
    sides = range_sides(pd.Series([3, 3]), pd.Series([30.0, 85.0]), {3: (22.0, 95.0)})

    assert list(sides) == [INSIDE_RANGE, INSIDE_RANGE]


def test_each_end_is_named_because_they_are_opposite_advice() -> None:
    """Below the floor means "too close to estimate"; above the ceiling means
    "further than anything measured"."""
    sides = range_sides(pd.Series([3, 3]), pd.Series([1.0, 200.0]), {3: (22.0, 95.0)})

    assert list(sides) == [BELOW_RANGE, ABOVE_RANGE]


def test_a_sector_is_judged_against_its_own_span() -> None:
    """12 minutes is inside sector 7's span and below sector 3's. One global
    floor would wave through the extrapolation in every higher-floored sector."""
    sides = range_sides(
        pd.Series([3, 7]), pd.Series([12.0, 12.0]), {3: (22.0, 95.0), 7: (8.0, 55.0)}
    )

    assert list(sides) == [BELOW_RANGE, INSIDE_RANGE]


def test_a_sector_with_no_fitted_range_is_an_error_rather_than_inside() -> None:
    """A missing floor read as "not below the floor" lets every cell in that
    sector through -- a check reporting clean because it did not look."""
    with pytest.raises(ValueError, match="no fitted range"):
        range_sides(pd.Series([3, 9]), pd.Series([30.0, 30.0]), {3: (22.0, 95.0)})


@pytest.mark.parametrize(
    ("sectors", "freeflow"), [([3, None], [20.0, 20.0]), ([3, 3], [20.0, None])]
)
def test_a_cell_that_cannot_be_placed_is_an_error(
    sectors: list[object], freeflow: list[object]
) -> None:
    with pytest.raises(ValueError, match="no sector or no free-flow"):
        range_sides(
            pd.Series(sectors, dtype="object"),
            pd.Series(freeflow, dtype="object"),
            {3: (22.0, 95.0)},
        )


def test_the_side_is_truthy_exactly_when_the_cell_is_outside() -> None:
    """Both map builders read the result with `.astype(bool)`, so the empty
    string for "inside" is load-bearing rather than cosmetic."""
    sides = range_sides(pd.Series([3, 3, 3]), pd.Series([1.0, 30.0, 200.0]), {3: (22.0, 95.0)})

    assert list(sides.astype(bool)) == [True, False, True]
