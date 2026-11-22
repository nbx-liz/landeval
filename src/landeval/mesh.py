"""JIS X 0410 standard grid squares (地域メッシュ).

Mesh codes are handled through a normalised integer index rather than by
string surgery on the digits. Writing ``lat * 1.5`` puts the primary mesh row
on a unit scale, and every subdivision level is then a plain integer division,
which makes encode/decode exact inverses and makes bbox enumeration a simple
range over indices.

Supported levels:

===========  ======  ============  =====================
Nominal size Digits  Subdivisions  Cell size (lat, lon)
===========  ======  ============  =====================
1 km         8       80            30", 45"
500 m        9       160           15", 22.5"
250 m        10      320           7.5", 11.25"
===========  ======  ============  =====================
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from functools import cache

from pyproj import Geod

# The ellipsoid every area in this project is measured on. Named rather than
# spelled at the call site so the reference frame is one decision.
WGS84 = "WGS84"

# Bounding box of the Japanese territory. Mesh codes are only defined for the
# Japanese grid, so a coordinate outside it is a caller error, not a cell.
JAPAN_LAT_RANGE = (20.0, 46.5)
JAPAN_LON_RANGE = (122.0, 154.0)

# Subdivisions of a primary mesh along each axis, per supported cell size.
_SUBDIVISIONS: dict[int, int] = {1000: 80, 500: 160, 250: 320}
_CODE_LENGTH: dict[int, int] = {1000: 8, 500: 9, 250: 10}
_LENGTH_TO_SIZE: dict[int, int] = {v: k for k, v in _CODE_LENGTH.items()}

# Degrees of latitude spanned by one primary mesh row (40 minutes).
_PRIMARY_LAT_DEG = 2.0 / 3.0

# Cell edges are not exactly representable in binary floating point, so an
# index computed on an edge can land just below the integer and floor to the
# neighbouring cell. Snapping within this tolerance (~0.1 mm in degrees) keeps
# encode/decode exact inverses and stops bbox enumeration from emitting a
# leading column that does not touch the box.
_EPS = 1e-9


class MeshError(Exception):
    """Raised for an unsupported mesh size, an out-of-range point, or a malformed code."""


def _check_size(size_m: int) -> int:
    if size_m not in _SUBDIVISIONS:
        raise MeshError(
            f"unsupported mesh size {size_m} m; supported: {sorted(_SUBDIVISIONS, reverse=True)}"
        )
    return _SUBDIVISIONS[size_m]


def _check_point(lat: float, lon: float) -> None:
    lat_ok = JAPAN_LAT_RANGE[0] <= lat <= JAPAN_LAT_RANGE[1]
    lon_ok = JAPAN_LON_RANGE[0] <= lon <= JAPAN_LON_RANGE[1]
    if not (lat_ok and lon_ok):
        raise MeshError(
            f"coordinate ({lat}, {lon}) is outside Japan "
            f"(lat {JAPAN_LAT_RANGE}, lon {JAPAN_LON_RANGE})"
        )


def _floor_index(value: float) -> int:
    """Floor to an integer index, snapping values sitting on a cell edge."""
    nearest = round(value)
    if abs(value - nearest) < _EPS:
        return int(nearest)
    return math.floor(value)


def _indices(lat: float, lon: float, subdivisions: int) -> tuple[int, int]:
    """Return the (row, column) index of the containing cell at the given level."""
    return (
        _floor_index(lat / _PRIMARY_LAT_DEG * subdivisions),
        _floor_index((lon - 100.0) * subdivisions),
    )


def mesh_step_deg(size_m: int) -> tuple[float, float]:
    """Return (latitude, longitude) degrees spanned by one cell of this size.

    The grid is regular in degrees, so a caller working on many cells at once
    can offset from each centre instead of decoding every code.
    """
    subdivisions = _check_size(size_m)
    return _PRIMARY_LAT_DEG / subdivisions, 1.0 / subdivisions


def mesh_size_of(code: str) -> int:
    """Return the nominal cell size in metres implied by the code length."""
    size = _LENGTH_TO_SIZE.get(len(code))
    if size is None:
        raise MeshError(
            f"mesh code {code!r} has length {len(code)}; expected one of {sorted(_LENGTH_TO_SIZE)}"
        )
    return size


def mesh_code(lat: float, lon: float, size_m: int) -> str:
    """Encode a coordinate as a mesh code of the requested cell size."""
    subdivisions = _check_size(size_m)
    _check_point(lat, lon)
    row, col = _indices(lat, lon, subdivisions)
    return _encode(row, col, subdivisions)


def mesh_indices(lat: float, lon: float, size_m: int) -> tuple[int, int]:
    """Return the global (row, column) grid index of the cell containing a point.

    Row increases northward and column eastward, both without gaps, which is
    what turns a set of cells into an array for rasterising.
    """
    subdivisions = _check_size(size_m)
    _check_point(lat, lon)
    return _indices(lat, lon, subdivisions)


def _encode(row: int, col: int, subdivisions: int) -> str:
    primary_row, r = divmod(row, subdivisions)
    primary_col, c = divmod(col, subdivisions)

    step = subdivisions // 8  # secondary mesh
    secondary_row, r = divmod(r, step)
    secondary_col, c = divmod(c, step)

    step //= 10  # standard (1 km) mesh
    tertiary_row, r = divmod(r, step)
    tertiary_col, c = divmod(c, step)

    code = f"{primary_row}{primary_col}{secondary_row}{secondary_col}{tertiary_row}{tertiary_col}"

    while step > 1:  # each remaining level halves the cell
        step //= 2
        half_row, r = divmod(r, step)
        half_col, c = divmod(c, step)
        code += str(2 * half_row + half_col + 1)
    return code


def _decode(code: str) -> tuple[int, int, int]:
    """Return (row, column, subdivisions) for a mesh code."""
    size = mesh_size_of(code)
    subdivisions = _SUBDIVISIONS[size]
    if not code.isdigit():
        raise MeshError(f"mesh code {code!r} contains non-digit characters")

    primary_row, primary_col = int(code[0:2]), int(code[2:4])
    secondary_row, secondary_col = int(code[4]), int(code[5])
    tertiary_row, tertiary_col = int(code[6]), int(code[7])

    if secondary_row > 7 or secondary_col > 7:
        raise MeshError(f"mesh code {code!r} has a secondary index outside 0-7")

    step = subdivisions // 80
    row = ((primary_row * 8 + secondary_row) * 10 + tertiary_row) * step
    col = ((primary_col * 8 + secondary_col) * 10 + tertiary_col) * step

    for digit in code[8:]:
        value = int(digit)
        if not 1 <= value <= 4:
            raise MeshError(f"mesh code {code!r} has subdivision digit {digit!r} outside 1-4")
        step //= 2
        row += ((value - 1) // 2) * step
        col += ((value - 1) % 2) * step

    return row, col, subdivisions


def mesh_bounds(code: str) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) in degrees for a mesh code."""
    row, col, subdivisions = _decode(code)
    lat_step = _PRIMARY_LAT_DEG / subdivisions
    lon_step = 1.0 / subdivisions
    south = row * lat_step
    west = col * lon_step + 100.0
    return south, west, south + lat_step, west + lon_step


def mesh_center(code: str) -> tuple[float, float]:
    """Return the (lat, lon) centre of a mesh cell."""
    south, west, north, east = mesh_bounds(code)
    return (south + north) / 2, (west + east) / 2


def primary_mesh_of(code: str) -> str:
    """The four-digit primary mesh (第1次地域区画) a cell lies in.

    The code's first two digits are the primary row and the next two the
    primary column, for every supported cell size; the code is checked first so
    a malformed one raises instead of yielding a wrong prefix.
    """
    mesh_bounds(code)
    return code[:4]


# Every area is a whole number of these. pyproj's geodesic area
# differs in its last bit between machines -- aarch64 libproj contracts to FMA
# -- and that bit is enough to turn a round sum into a value such as
# 0.7999999999999999.
# A multiple of a power of two is exact in binary floating point, and so is any
# sum of them below 2**23 km2, so once each 250 m row is rounded to this unit,
# every cell, every total and every share is the same on every machine. About
# 0.001 m2, far below anything this project prints.
AREA_UNIT_KM2 = 2.0**-30


def _supported_rows() -> range:
    """The 250 m rows under every 1 km row that touches JAPAN_LAT_RANGE.

    North edge inclusive: `mesh_code(46.5, ...)` opens 1 km row 5580, so its
    four 250 m rows are supported too. Coarser cells are sums of these rows,
    which is why the range is closed under the 1 km grid.
    """
    coarse = _SUBDIVISIONS[1000]
    per_km_row = _SUBDIVISIONS[250] // coarse
    south = _floor_index(JAPAN_LAT_RANGE[0] / _PRIMARY_LAT_DEG * coarse)
    north = _floor_index(JAPAN_LAT_RANGE[1] / _PRIMARY_LAT_DEG * coarse)
    return range(south * per_km_row, (north + 1) * per_km_row)


SUPPORTED_250M_ROWS = _supported_rows()


def _raw_row_area_km2(row: int) -> float:
    """pyproj's area in km2 of any 250 m cell in this row, before rounding."""
    lat_step, lon_step = mesh_step_deg(250)
    south = row * lat_step
    north = south + lat_step
    area, _ = Geod(ellps=WGS84).polygon_area_perimeter(
        [0.0, lon_step, lon_step, 0.0],
        [south, south, north, north],
    )
    return abs(area) / 1e6


@cache
def _row_units(row: int) -> int:
    """Cached on the row rather than the code: every cell at one latitude has
    the same area. Unbounded because its keys are: `area_units_of_row` admits
    only the 12,724 supported rows."""
    return round(_raw_row_area_km2(row) / AREA_UNIT_KM2)


def area_units_of_row(row: int) -> int:
    """Area of any 250 m cell in this row, in whole AREA_UNIT_KM2."""
    if row not in SUPPORTED_250M_ROWS:
        raise MeshError(
            f"250 m row {row} is outside the supported rows "
            f"{SUPPORTED_250M_ROWS.start}..{SUPPORTED_250M_ROWS.stop - 1} (JAPAN_LAT_RANGE)"
        )
    return _row_units(row)


def cell_area_units(code: str) -> int:
    """Area of one cell in whole AREA_UNIT_KM2: the exact sum of its 250 m cells.

    Summing rather than measuring each level keeps a parent exactly equal to
    its children at every level, which rounding each level separately would not.
    """
    row, _, subdivisions = _decode(code)
    per_side = _SUBDIVISIONS[250] // subdivisions
    first = row * per_side
    if first not in SUPPORTED_250M_ROWS or first + per_side - 1 not in SUPPORTED_250M_ROWS:
        raise MeshError(f"mesh code {code!r} lies outside JAPAN_LAT_RANGE {JAPAN_LAT_RANGE}")
    return per_side * sum(_row_units(first + offset) for offset in range(per_side))


def cell_area_km2(code: str) -> float:
    """Area of one cell, on the WGS84 ellipsoid.

    Not a constant, and specifically not 0.25. The nominal sizes in this
    module's table are labels: JIS X 0410 defines the cells in arc seconds, so a
    "500 m" cell is 15" by 22.5" -- about 463 m by 565 m at the latitude of
    Tokyo, and smaller further north. Multiplying a cell count by 0.25 is
    therefore several percent low at that latitude.

    A malformed code raises rather than returning zero: an area of zero for an
    unplaceable cell would shrink whatever total it lands in without saying so.
    So does a well-formed code outside Japan, whose area no test has pinned.

    The value is a whole number of AREA_UNIT_KM2, the same on every
    machine, and a cell is exactly the sum of its children.
    """
    return cell_area_units(code) * AREA_UNIT_KM2


def area_km2(codes: Iterable[str]) -> float:
    """Total area of the given cells.

    Zero for an empty selection: "no cells qualified" is an answer a caller
    prints, and it should not have to distinguish it from a failure.
    """
    return sum(cell_area_km2(code) for code in codes)


def meshes_in_bbox(
    south: float, west: float, north: float, east: float, size_m: int
) -> Iterator[str]:
    """Yield every mesh code whose cell intersects the bounding box.

    The bbox edges are half-open in the same sense as :func:`mesh_code`: a cell
    is included when its interior overlaps the box.
    """
    subdivisions = _check_size(size_m)
    if north <= south or east <= west:
        raise MeshError(
            f"invalid bbox: south={south} west={west} north={north} east={east} "
            "(north must exceed south and east must exceed west)"
        )
    _check_point(south, west)
    _check_point(north, east)

    row0, col0 = _indices(south, west, subdivisions)
    lat_step = _PRIMARY_LAT_DEG / subdivisions
    lon_step = 1.0 / subdivisions

    row = row0
    while row * lat_step < north - _EPS:
        col = col0
        while col * lon_step + 100.0 < east - _EPS:
            yield _encode(row, col, subdivisions)
            col += 1
        row += 1
