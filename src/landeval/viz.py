"""Map output for the screening layers.

Mesh cells form a regular grid, so they are rasterised into a single image and
laid over the basemap rather than drawn as one polygon per cell: a
GeoJSON of the working set would be far too heavy for a browser to handle
interactively, which defeats the point of a map meant for discussion.

Bands are discrete on purpose. A continuous ramp reads as precision the OSRM
layer does not have -- it is a filter, not an adopted value -- while
bands tied to the thresholds show the decision being made.

The basemap is GSI tiles, so the page needs network access to the GSI tile
host; a viewer that blocks external tile hosts shows the layers without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from landeval.mesh import _PRIMARY_LAT_DEG, _SUBDIVISIONS, area_km2, mesh_indices

GSI_PALE_TILES = "https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png"
GSI_ATTRIBUTION = '<a href="https://maps.gsi.go.jp/development/ichiran.html">国土地理院</a>'

BAND_ALPHA = 190

# Screening classes get status colours, not series colours: they encode a state,
# and the legend carries the label so the colour never stands alone.
CLASS_COLOURS: dict[str, str] = {
    "pass": "#0ca30c",
    "hold": "#fab219",
    "exclude": "#9a9992",
    "unknown": "#d03b3b",
}
CLASS_LABELS: dict[str, str] = {
    "pass": "原則採用",
    "hold": "条件付き",
    "exclude": "除外",
    "unknown": "判定不能",
}

# Hazard uses a sequential ramp rather than status colours: the value being
# shown is a magnitude (how much of the cell lies outside every designated
# hazard area), so one hue running light to dark carries it and the ordering
# is legible without reading the legend. The grey is not part of the ramp --
# it marks cells the transport filter already removed.
HAZARD_COLOURS: dict[str, str] = {
    "clear": "#0b5f2a",
    "mostly_clear": "#2f8b4d",
    "partly": "#74b878",
    "half": "#b3ddaa",
    "hazardous": "#eaf3dc",
    # Off the sequential ramp on purpose: an unanswerable cell is a state, not
    # a lighter shade of clear. Same purple everywhere it appears.
    "no_data": "#7a4fb0",
    "transport_excluded": "#cfcec8",
}
# The final block classification. `no_data` shares the selection purple so the
# reader learns one colour for "nobody could answer this" across all maps.
BLOCK_COLOURS: dict[str, str] = {
    "block_pass": "#0b5f2a",
    "block_hold": "#e8a317",
    "block_exclude": "#b03030",
    # Recoloured away from the selection purple when `no_data` joined this
    # palette: two classes sharing one fill is a legend that cannot be read.
    "block_unrouted": "#5b7a99",
    "muni_excluded": "#8a5a2a",
    "hazard_excluded": "#5a5954",
    "no_data": "#7a4fb0",
    "transport_excluded": "#cfcec8",
}
# The per-cell estimate. Same purple for `no_data`, same grey for the
# transport cut, so the three commute maps read as one system.
ESTIMATE_COLOURS: dict[str, str] = {
    "est_pass": "#0b5f2a",
    "est_hold": "#e8a317",
    "est_over": "#b03030",
    "model_only": "#6a8caf",
    # Two ways of not knowing, so two colours: the hazard inputs were never
    # obtained here, versus the commute model was never fitted here.
    "outside_fitted_range": "#9aa0a6",
    "no_data": "#7a4fb0",
    "hazard_excluded": "#5a5954",
    "transport_excluded": "#cfcec8",
}
# The three-way split over a chosen set of hazard layers. These are states, not
# magnitudes, so they get status colours and the legend always carries the word.
SELECTION_COLOURS: dict[str, str] = {
    "clear": "#0b7a33",
    "partial": "#e8a317",
    "covered": "#b03030",
    "no_data": "#7a4fb0",
    "transport_excluded": "#cfcec8",
}
SELECTION_LABELS: dict[str, str] = {
    "clear": "指定区域にかからない",
    "partial": "一部がかかる（要確認）",
    "covered": "全面がかかる（除外）",
    "no_data": "判定不能",
    "transport_excluded": "交通で除外済み",
}

# Band labels live beside the rule that produces them, not here: a palette and a
# label table that live apart from their classifier drift apart from it too, and
# a class with no entry renders as a blank legend row.


@dataclass(frozen=True)
class Raster:
    """An RGBA image plus the lat/lon box it covers."""

    image: npt.NDArray[np.uint8]
    south: float
    west: float
    north: float
    east: float

    @property
    def folium_bounds(self) -> list[list[float]]:
        return [[self.south, self.west], [self.north, self.east]]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def band_colour(
    minutes: float, bands: tuple[tuple[float, str], ...]
) -> tuple[int, int, int] | None:
    """Colour for a value, or None when it falls beyond the last band."""
    for edge, colour in bands:
        if minutes <= edge:
            return _hex_to_rgb(colour)
    return None


# The readout grid packs one character per cell. 26 sub-grid outcomes (0..25 of
# 25 sample points) fit in base36 digits, and an unfilled cell is a full stop.
READOUT_ABSENT = "."
READOUT_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

# The three ways a click produces no number. They are kept apart: reporting
# every miss as 「対象範囲外」 would be a statement about scope, and only one of
# them is. On an estimate map a slot of the readout rectangle
# can answer with no number for two reasons: no cell of the working set is
# there, or a cell is there, painted and classified in the legend, that this
# column has no value for. Telling a
# reader that ground the map is showing them is outside the target area is the
# false-clean shape pointed the other way: an answer where there was only
# an absence.
OFF_MAP = "off-map"
NO_VALUE = "no-value"
UNREADABLE = "unreadable"
OFF_MAP_TEXT = "対象範囲外"
NO_VALUE_TEXT = "圏域内・値なし（凡例の分類を参照）"
# The packed string is written from READOUT_ALPHABET alone, so an unknown
# character means the data, not the ground. Left as `indexOf`'s -1 it would have
# been arithmetic: a negative "value" rendered as a confident number.
UNREADABLE_TEXT = "読み出しデータが壊れています"


@dataclass(frozen=True)
class ValueGrid:
    """Per-cell values packed for lookup from a click, plus the grid geometry.

    The map fill is a single raster image, so a cell carries no feature to hang
    a tooltip on. Rather than emitting one polygon per cell, the values
    are packed into the same row/column grid the raster already uses and looked
    up by arithmetic on the clicked coordinate.
    """

    data: str
    row0: int
    row1: int
    col0: int
    col1: int
    lat_step: float
    lon_step: float

    @property
    def width(self) -> int:
        return self.col1 - self.col0 + 1


def _cell_indices(
    frame: pd.DataFrame, mesh_size_m: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    indices = [
        mesh_indices(lat, lon, mesh_size_m)
        for lat, lon in zip(frame["lat"], frame["lon"], strict=True)
    ]
    return (
        np.array([r for r, _ in indices], dtype=np.int64),
        np.array([c for _, c in indices], dtype=np.int64),
    )


def value_grid(
    frame: pd.DataFrame,
    value_column: str,
    *,
    mesh_size_m: int = 500,
    levels: int = 25,
    scale: float = 1.0,
) -> ValueGrid:
    """Pack a numeric column into a lookup grid.

    Values are stored as their position on a ``levels``-step ladder running from
    zero to ``scale``. With the defaults that is a fraction expressed as a count
    of the 25 sample points, which is exactly what the hazard overlay measured,
    so the readout shows the granularity the data actually has rather than a
    precision it does not.
    """
    if frame.empty:
        raise ValueError("cannot build a value grid from an empty frame")
    if value_column not in frame.columns:
        raise ValueError(f"{value_column!r} is not a column of the frame")
    if levels + 1 > len(READOUT_ALPHABET):
        raise ValueError(f"{levels} levels do not fit in the readout alphabet")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    rows, cols = _cell_indices(frame, mesh_size_m)
    row0, row1 = int(rows.min()), int(rows.max())
    col0, col1 = int(cols.min()), int(cols.max())
    width, height = col1 - col0 + 1, row1 - row0 + 1

    packed = [READOUT_ABSENT] * (width * height)
    values = frame[value_column].to_numpy(dtype=float)
    for row, col, value in zip(rows, cols, values, strict=True):
        if not np.isfinite(value):
            continue
        step = round(float(value) / scale * levels)
        packed[(row1 - int(row)) * width + (int(col) - col0)] = READOUT_ALPHABET[
            min(max(step, 0), levels)
        ]

    subdivisions = _SUBDIVISIONS[mesh_size_m]
    return ValueGrid(
        data="".join(packed),
        row0=row0,
        row1=row1,
        col0=col0,
        col1=col1,
        lat_step=_PRIMARY_LAT_DEG / subdivisions,
        lon_step=1.0 / subdivisions,
    )


def rasterize(
    frame: pd.DataFrame,
    value_column: str,
    *,
    mesh_size_m: int = 500,
    bands: tuple[tuple[float, str], ...] | None = None,
    colour_column: str | None = None,
    palette: dict[str, str] | None = None,
) -> Raster:
    """Render one value per mesh cell into an RGBA image covering their extent.

    Cells absent from ``frame`` stay fully transparent, so a gap in the data
    looks like a gap rather than like a value.
    """
    if frame.empty:
        raise ValueError("cannot rasterize an empty frame")
    if value_column not in frame.columns:
        raise ValueError(f"{value_column!r} is not a column of the frame")
    if colour_column is None and bands is None:
        raise ValueError("bands are required to colour by value; pass bands or colour_column")

    indices = [
        mesh_indices(lat, lon, mesh_size_m)
        for lat, lon in zip(frame["lat"], frame["lon"], strict=True)
    ]
    rows = np.array([r for r, _ in indices])
    cols = np.array([c for _, c in indices])
    row0, row1 = int(rows.min()), int(rows.max())
    col0, col1 = int(cols.min()), int(cols.max())

    height, width = row1 - row0 + 1, col1 - col0 + 1
    image = np.zeros((height, width, 4), dtype=np.uint8)

    if colour_column is not None:
        # Categorical fill: the column already holds the class of each cell.
        colours = palette if palette is not None else CLASS_COLOURS
        classes = frame[colour_column].astype("object").to_numpy()
        pixel_rows_c = (row1 - rows).astype(int)
        pixel_cols_c = (cols - col0).astype(int)
        for pixel_row, pixel_col, name in zip(pixel_rows_c, pixel_cols_c, classes, strict=True):
            hex_colour = colours.get(str(name))
            if hex_colour is None:
                continue
            image[pixel_row, pixel_col] = (*_hex_to_rgb(hex_colour), BAND_ALPHA)
        subdivisions = _SUBDIVISIONS[mesh_size_m]
        lat_step = _PRIMARY_LAT_DEG / subdivisions
        lon_step = 1.0 / subdivisions
        return Raster(
            image=image,
            south=row0 * lat_step,
            west=col0 * lon_step + 100.0,
            north=(row1 + 1) * lat_step,
            east=(col1 + 1) * lon_step + 100.0,
        )

    values = frame[value_column].astype("float64").to_numpy()
    # Image rows run north to south, the opposite of the grid index.
    pixel_rows = (row1 - rows).astype(int)
    pixel_cols = (cols - col0).astype(int)
    for pixel_row, pixel_col, value in zip(pixel_rows, pixel_cols, values, strict=True):
        if np.isnan(value):
            continue
        if bands is None:  # unreachable: refused above; narrows the type
            raise ValueError("bands are required to colour by value")
        colour = band_colour(float(value), bands)
        if colour is None:
            continue
        image[pixel_row, pixel_col] = (*colour, BAND_ALPHA)

    subdivisions = _SUBDIVISIONS[mesh_size_m]
    lat_step = _PRIMARY_LAT_DEG / subdivisions
    lon_step = 1.0 / subdivisions
    return Raster(
        image=image,
        south=row0 * lat_step,
        west=col0 * lon_step + 100.0,
        north=(row1 + 1) * lat_step,
        east=(col1 + 1) * lon_step + 100.0,
    )


def _png_data_uri(image: npt.NDArray[np.uint8]) -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _class_legend_html(
    counts: dict[str, int],
    areas: dict[str, float],
    generated_at: str,
    subtitle: str,
    palette: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    caveat: str | None = None,
) -> str:
    colours = palette if palette is not None else CLASS_COLOURS
    names = labels if labels is not None else CLASS_LABELS
    rows = []
    for name, colour in colours.items():
        if name not in counts:
            continue
        # From the cells themselves, not from the count: a mesh cell is 15" by
        # 22.5", so its area depends on its latitude and is not 0.25 km2
        # anywhere in Japan (see `mesh.cell_area_km2`).
        area = areas[name]
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
            f'<span style="width:16px;height:12px;background:{colour};'
            f'border:1px solid #444;display:inline-block"></span>'
            f"<span>{names[name]} {counts[name]:,} セル / {area:,.0f} km²</span></div>"
        )
    return (
        '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;'
        "background:rgba(255,255,255,0.94);padding:10px 12px;border:1px solid #999;"
        'border-radius:4px;font:12px/1.5 sans-serif;color:#111">'
        f'<div style="font-weight:600;margin-bottom:4px">{subtitle}</div>'
        + "".join(rows)
        + (
            # A limit of the whole map has no colour to carry it, so it is
            # written into the legend. Printing it to the terminal that built
            # the file leaves the reader of the file with no way to learn it.
            f'<div style="margin-top:6px;padding-top:6px;border-top:1px solid #ccc;'
            f'color:#7a4fb0;max-width:22em">{caveat}</div>'
            if caveat
            else ""
        )
        + f'<div style="margin-top:6px;color:#555">生成: {generated_at}</div>'
        "</div>"
    )


def _readout_html(
    grid: ValueGrid,
    title: str,
    levels: int = 25,
    scale: float = 1.0,
    unit: str = "%",
    style: str = "coverage",
) -> str:
    """A click handler that reports the packed value under the clicked point.

    The lookup is the inverse of the rasterisation: floor the coordinate into
    the same row/column the cell was drawn at, then index the packed string.
    Keeping both sides on the mesh indices means the readout can never point at
    a different cell than the one the reader is looking at.
    """
    return f"""
<div id="landeval-readout" style="position:fixed;top:12px;right:12px;z-index:9999;
 background:rgba(255,255,255,0.94);padding:10px 12px;border:1px solid #999;
 border-radius:4px;font:12px/1.5 sans-serif;color:#111;min-width:190px">
<div style="font-weight:600;margin-bottom:4px">{title}</div>
<div id="landeval-readout-body" style="color:#555">地図をクリックすると値を表示</div>
</div>
<script>
(function () {{
  var DATA = "{grid.data}";
  var ROW0 = {grid.row0}, ROW1 = {grid.row1}, COL0 = {grid.col0}, COL1 = {grid.col1};
  var LAT_STEP = {grid.lat_step!r}, LON_STEP = {grid.lon_step!r};
  var WIDTH = {grid.width}, LEVELS = {levels}, SCALE = {scale!r};
  var UNIT = "{unit}", STYLE = "{style}";
  var ALPHABET = "{READOUT_ALPHABET}";

  function lookup(lat, lon) {{
    var row = Math.floor(lat / LAT_STEP);
    var col = Math.floor((lon - 100.0) / LON_STEP);
    if (row < ROW0 || row > ROW1 || col < COL0 || col > COL1) return "{OFF_MAP}";
    var ch = DATA.charAt((ROW1 - row) * WIDTH + (col - COL0));
    if (ch === "{READOUT_ABSENT}") return "{NO_VALUE}";
    var hits = ALPHABET.indexOf(ch);
    return hits < 0 ? "{UNREADABLE}" : hits;
  }}

  function describe(hits) {{
    if (hits === "{OFF_MAP}") return "{OFF_MAP_TEXT}";
    if (hits === "{NO_VALUE}") return "{NO_VALUE_TEXT}";
    if (hits === "{UNREADABLE}") return "{UNREADABLE_TEXT}";
    if (STYLE === "coverage") {{
      var percent = Math.round((hits / LEVELS) * 100);
      var label = hits === 0 ? "かからない"
                : hits >= LEVELS ? "全面がかかる（除外）"
                : "一部がかかる（要確認）";
      return label + "<br><b>" + percent + "%</b>（" + hits + " / " + LEVELS + " 点）";
    }}
    var value = (hits / LEVELS) * SCALE;
    var stepText = (SCALE / LEVELS).toFixed(0);
    return "<b>" + Math.round(value) + " " + UNIT + "</b><br>"
         + "（" + stepText + " " + UNIT + "刻み）";
  }}

  function attach() {{
    var found = null;
    for (var key in window) {{
      if (key.indexOf("map_") === 0 && window[key] && window[key].on) {{
        found = window[key];
        break;
      }}
    }}
    if (!found) {{ window.setTimeout(attach, 200); return; }}
    found.on("click", function (event) {{
      var body = document.getElementById("landeval-readout-body");
      body.innerHTML = describe(lookup(event.latlng.lat, event.latlng.lng));
    }});
  }}
  attach();
}})();
</script>
"""


def _legend_html(bands: tuple[tuple[float, str], ...], generated_at: str, subtitle: str) -> str:
    rows = []
    previous = 0.0
    for edge, colour in bands:
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0">'
            f'<span style="width:16px;height:12px;background:{colour};'
            f'border:1px solid #444;display:inline-block"></span>'
            f"<span>{previous:.0f}–{edge:.0f} 分</span></div>"
        )
        previous = edge
    return (
        '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;'
        "background:rgba(255,255,255,0.94);padding:10px 12px;border:1px solid #999;"
        'border-radius:4px;font:12px/1.5 sans-serif;color:#111">'
        f'<div style="font-weight:600;margin-bottom:4px">{subtitle}</div>'
        + "".join(rows)
        + f'<div style="margin-top:6px;color:#555">生成: {generated_at}</div>'
        "</div>"
    )


def build_map(
    freeflow: pd.DataFrame,
    destination: tuple[float, float],
    destination_name: str,
    output: Path,
    *,
    calibration_points: pd.DataFrame | None = None,
    measured: pd.DataFrame | None = None,
    mesh_size_m: int = 500,
    bands: tuple[tuple[float, str], ...] | None = None,
    class_column: str | None = None,
    palette: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    layer_title: str = "方位別 足切り結果",
    caveat: str | None = None,
    readout_column: str | None = None,
    readout_title: str = "クリックした地点",
    readout_levels: int = 25,
    readout_scale: float = 1.0,
    readout_unit: str = "%",
    readout_style: str = "coverage",
    generated_at: datetime | None = None,
) -> Path:
    """Write a standalone HTML map of the OSRM filter layer and the sampled points.

    With ``class_column`` the fill shows the screening class of each cell rather
    than its free-flow band, which is the view that answers "what did the filter
    keep".
    """
    import folium

    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
    raster = rasterize(
        freeflow,
        "freeflow_min",
        mesh_size_m=mesh_size_m,
        bands=bands,
        colour_column=class_column,
        palette=palette,
    )

    fmap = folium.Map(
        location=list(destination),
        zoom_start=9,
        tiles=GSI_PALE_TILES,
        attr=GSI_ATTRIBUTION,
        control_scale=True,
    )

    folium.raster_layers.ImageOverlay(
        image=_png_data_uri(raster.image),
        bounds=raster.folium_bounds,
        opacity=1.0,
        name=(layer_title if class_column else "OSRM 自由流時間（足切り用・採用値ではない）"),
        interactive=False,
    ).add_to(fmap)

    folium.Marker(
        location=list(destination),
        tooltip=destination_name,
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(fmap)

    if calibration_points is not None and not calibration_points.empty:
        layer = folium.FeatureGroup(name=f"校正点 ({len(calibration_points)})", show=True)
        for _, point in calibration_points.iterrows():
            folium.CircleMarker(
                location=[float(point["lat"]), float(point["lon"])],
                radius=4,
                color="#1f3d99",
                fill=True,
                fill_opacity=0.9,
                tooltip=f"{point['mesh_id']} 自由流 {float(point['freeflow_min']):.0f} 分",
            ).add_to(layer)
        layer.add_to(fmap)

    if measured is not None and not measured.empty:
        layer = folium.FeatureGroup(name=f"Google 実測 ({len(measured)} 行)", show=True)
        for mesh_id, rows in measured.groupby("mesh_id"):
            head = rows.iloc[0]
            lines = "<br>".join(
                f"{row['schedule_id']}: <b>{row['measured_min']:.0f} 分</b>"
                for _, row in rows.iterrows()
            )
            popup = (
                f"<b>{mesh_id}</b><br>OSRM 自由流 {float(head['freeflow_min']):.1f} 分<br>{lines}"
            )
            folium.CircleMarker(
                location=[float(head["lat"]), float(head["lon"])],
                radius=8,
                color="#111111",
                weight=2,
                fill=True,
                fill_color="#ffd166",
                fill_opacity=1.0,
                popup=folium.Popup(popup, max_width=260),
            ).add_to(layer)
        layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    # get_root() is annotated as returning Element, but a Map's root is a Figure
    # and only a Figure carries the html slot the legend has to go into.
    root = fmap.get_root()
    if class_column:
        counts = freeflow[class_column].value_counts().to_dict()
        areas = freeflow.groupby(class_column)["mesh_id"].apply(area_km2).to_dict()
        legend = _class_legend_html(
            {str(k): int(v) for k, v in counts.items()},
            {str(k): float(v) for k, v in areas.items()},
            stamp,
            layer_title,
            palette,
            labels,
            caveat,
        )
    else:
        if caveat:
            # Rather than dropping it: a caveat that does not reach the file is
            # the failure this parameter exists to prevent, and losing it quietly
            # on one branch would reintroduce exactly that.
            raise ValueError("caveat is only rendered on the class legend; pass class_column")
        if bands is None:
            raise ValueError("bands are required to colour by value; pass bands or class_column")
        legend = _legend_html(bands, stamp, "OSRM 自由流時間")
    root.html.add_child(folium.Element(legend))  # type: ignore[attr-defined]
    if readout_column:
        grid = value_grid(
            freeflow,
            readout_column,
            mesh_size_m=mesh_size_m,
            levels=readout_levels,
            scale=readout_scale,
        )
        root.html.add_child(  # type: ignore[attr-defined]
            folium.Element(
                _readout_html(
                    grid, readout_title, readout_levels, readout_scale, readout_unit, readout_style
                )
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output))
    return output
