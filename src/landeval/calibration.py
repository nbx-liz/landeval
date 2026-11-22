"""Turning measured travel times into direction-specific filter thresholds.

The same OSRM value can mean very different things by direction: congestion
and corridor layout differ, so one OSRM value can correspond to a measured time
in one direction that is far shorter than in another. A single filter edge
therefore cannot mean one thing, so the edges are derived per direction from
a target expressed in *measured* minutes -- the units the decision is actually
made in.

Inverting a fit is not symmetric with using it. Each edge can be pushed outward
by a margin in sigma multiples, because keeping a cell that later measures badly
costs one API call, while cutting a cell that would have measured well removes
it from consideration with nothing downstream to catch the mistake. How wide
each margin is, is screening policy: the caller passes it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from landeval.sampling import assign_bins, sector_key

# Fits below this are not trusted on their own and fall back to the pooled fit.
MIN_POINTS_PER_DIRECTION = 4


@dataclass(frozen=True)
class Fit:
    """An ordinary least squares fit of measured minutes on free-flow minutes."""

    slope: float
    intercept: float
    sigma: float
    n: int

    @property
    def is_usable(self) -> bool:
        return not math.isnan(self.slope) and self.slope > 0

    def at(self, osrm_min: float) -> float:
        """Predicted measured minutes for a free-flow value."""
        return self.slope * osrm_min + self.intercept

    def osrm_for(self, measured_min: float, *, sigma_margin: float = 0.0) -> float:
        """Free-flow value whose prediction reaches ``measured_min``.

        ``sigma_margin`` shifts the target upward before inverting, which moves
        the resulting edge outward -- the direction that keeps cells rather
        than dropping them.
        """
        if not self.is_usable:
            raise ValueError("cannot invert an unusable fit")
        target = measured_min + sigma_margin * (0.0 if math.isnan(self.sigma) else self.sigma)
        return (target - self.intercept) / self.slope


def fit_line(frame: pd.DataFrame) -> Fit:
    """Fit measured ~ freeflow. Needs three points to leave a residual estimate."""
    x = frame["freeflow_min"].astype(float)
    y = frame["measured_min"].astype(float)
    if len(frame) < 3 or x.nunique() < 2:
        return Fit(float("nan"), float("nan"), float("nan"), len(frame))
    slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    intercept = y.mean() - slope * x.mean()
    residual = y - (slope * x + intercept)
    return Fit(float(slope), float(intercept), float(residual.std(ddof=2)), len(frame))


def fitted_ranges(frame: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """The span of free-flow each sector's fit was actually made from.

    Pass the same rows the fit was made from. A range taken from a different
    frame polices one model's answers with another model's evidence.

    A line fitted from some minutes upward says nothing about a point near
    zero, but it will happily return its intercept for one -- which is how a
    cell at the destination itself can come back as a sizeable drive. The
    range is the model's own domain, read off the fitted points rather than
    chosen, and it is per sector because the sectors' lowest fitted points can
    differ widely: one global floor would miss the extrapolation in every
    sector whose own floor is higher.

    Both ends, because the domain has two of them. A check that guarded only
    the floor while calling itself a range would be easy to trust and half
    absent: a cell above its sector's ceiling would be answered as though the
    line had been fitted there.
    """
    if frame.empty:
        raise ValueError("no calibration points to take a fitted range from")
    for column in ("sector", "freeflow_min"):
        if column not in frame.columns:
            raise ValueError(f"calibration points carry no {column}")
    if frame["freeflow_min"].isna().any():
        raise ValueError("a fitted point has a null free-flow time")
    spans = frame.groupby("sector")["freeflow_min"].agg(["min", "max"])
    if spans.isna().to_numpy().any():
        raise ValueError("a sector's fitted range is null")
    return {
        int(str(sector)): (float(row["min"]), float(row["max"])) for sector, row in spans.iterrows()
    }


BELOW_RANGE = "below"
ABOVE_RANGE = "above"
INSIDE_RANGE = ""


def range_sides(
    sectors: pd.Series, freeflow_min: pd.Series, ranges: Mapping[int, tuple[float, float]]
) -> pd.Series:
    """Which end of its sector's fitted span each cell falls outside, if either.

    One answer, asked the same way by every consumer. With the rule in only one
    of them, the same cell can get a number of minutes in one place and a
    refusal in another.

    A sector with no fitted range is an error rather than an inside verdict: a
    missing floor read as "not below the floor" lets every cell in that sector
    through, which is the shape of a check that reports clean because it did not
    look. A null sector or free-flow time is an error for the same reason.
    """
    if not sectors.index.equals(freeflow_min.index):
        raise ValueError("sectors and free-flow times are not on the same index")
    if sectors.isna().any() or freeflow_min.isna().any():
        raise ValueError("a cell has no sector or no free-flow time to place in a range")
    unknown = sorted({sector_key(s) for s in sectors} - set(ranges))
    if unknown:
        raise ValueError(f"sectors {unknown} have no fitted range")
    sides = []
    for sector, freeflow in zip(sectors, freeflow_min, strict=True):
        floor, ceiling = ranges[sector_key(sector)]
        if float(freeflow) < floor:
            sides.append(BELOW_RANGE)
        elif float(freeflow) > ceiling:
            sides.append(ABOVE_RANGE)
        else:
            sides.append(INSIDE_RANGE)
    return pd.Series(sides, index=sectors.index, dtype="object")


def pooled_sigma(fits: dict[int, Fit], counts: dict[int, int]) -> float:
    """Residual spread after allowing each direction its own line."""
    numerator = sum(
        (counts[key] - 2) * fit.sigma**2
        for key, fit in fits.items()
        if not math.isnan(fit.sigma) and counts[key] > 2
    )
    degrees = sum(counts[key] - 2 for key, fit in fits.items() if not math.isnan(fit.sigma))
    return math.sqrt(numerator / degrees) if degrees > 0 else float("nan")


@dataclass(frozen=True)
class DirectionModel:
    """Per-direction fits plus the pooled fit used where a direction is thin."""

    fits: dict[int, Fit]
    overall: Fit
    sectors: int

    def fit_for(self, sector: int) -> Fit:
        """The fit to use for a sector, falling back to the pooled one.

        A direction with too few points would otherwise get an edge driven by
        noise; falling back keeps the answer conservative and, more
        importantly, makes the fallback visible rather than implicit.
        """
        fit = self.fits.get(sector)
        if fit is None or not fit.is_usable or fit.n < MIN_POINTS_PER_DIRECTION:
            return self.overall
        return fit

    def uses_fallback(self, sector: int) -> bool:
        return self.fit_for(sector) is self.overall


def fit_directions(
    calibration: pd.DataFrame, schedule_id: str, *, sectors: int = 12
) -> DirectionModel:
    """Fit one line per bearing sector from the calibration measurements."""
    subset = calibration[calibration["schedule_id"] == schedule_id]
    if subset.empty:
        raise ValueError(f"no calibration rows for schedule {schedule_id!r}")

    fits = {int(str(sector)): fit_line(group) for sector, group in subset.groupby("sector")}
    return DirectionModel(fits=fits, overall=fit_line(subset), sectors=sectors)


def direction_thresholds(
    model: DirectionModel,
    *,
    good_max_min: float,
    exclude_min: float,
    good_sigma_margin: float,
    exclude_sigma_margin: float,
) -> pd.DataFrame:
    """The OSRM edges that correspond to the measured-minute boundaries.

    ``good_sigma_margin`` and ``exclude_sigma_margin`` are the sigma multiples
    each target is raised by before inverting (`Fit.osrm_for`).
    """
    rows = []
    for sector in range(model.sectors):
        fit = model.fit_for(sector)
        if not fit.is_usable:
            continue
        rows.append(
            {
                "sector": sector,
                "slope": fit.slope,
                "intercept": fit.intercept,
                "sigma": fit.sigma,
                "n": fit.n,
                "fallback": model.uses_fallback(sector),
                "osrm_good_max": fit.osrm_for(good_max_min, sigma_margin=good_sigma_margin),
                "osrm_exclude": fit.osrm_for(exclude_min, sigma_margin=exclude_sigma_margin),
            }
        )
    return pd.DataFrame.from_records(rows)


def classify_cells(
    cells: pd.DataFrame,
    destination: tuple[float, float],
    model: DirectionModel,
    *,
    good_max_min: float,
    exclude_min: float,
    good_sigma_margin: float,
    exclude_sigma_margin: float,
) -> pd.DataFrame:
    """Attach the direction, the predicted measured time, and the screening class.

    ``predicted_min`` is an explanation, not an adopted value: it says
    what the filter believes, and only a Google measurement can confirm it.
    """
    binned = assign_bins(cells, destination, sectors=model.sectors, time_bands=(float("inf"),))
    thresholds = direction_thresholds(
        model,
        good_max_min=good_max_min,
        exclude_min=exclude_min,
        good_sigma_margin=good_sigma_margin,
        exclude_sigma_margin=exclude_sigma_margin,
    ).set_index("sector")

    good_edge = binned["sector"].map(thresholds["osrm_good_max"])
    exclude_edge = binned["sector"].map(thresholds["osrm_exclude"])
    slope = binned["sector"].map(thresholds["slope"])
    intercept = binned["sector"].map(thresholds["intercept"])

    freeflow = binned["freeflow_min"].astype("float64")
    predicted = slope * freeflow + intercept

    screening = pd.Series("hold", index=binned.index, dtype="object")
    screening[freeflow <= good_edge] = "pass"
    screening[freeflow > exclude_edge] = "exclude"
    # A sector with no usable fit yields no edge; that must not read as "pass".
    screening[good_edge.isna()] = "unknown"

    return binned.assign(
        osrm_good_max=good_edge,
        osrm_exclude=exclude_edge,
        predicted_min=predicted,
        screening_class=screening,
    ).drop(columns=["time_band"])
