# landeval

Building blocks for regional screening in property search. `landeval` is a Python package of small,
independently testable pieces:

- **Mesh grids**: JIS X 0410 standard grid squares (1 km, 500 m and 250 m), cell bounds,
  centres and areas.
- **Hazard overlays**: published hazard polygons (flood, landslide, tsunami, storm surge) and
  inland-flood raster tiles sampled onto mesh cells as a fraction of the cell, plus a
  point-by-point earthquake layer.
  The vocabulary never says "safe": ground outside a designated zone is reported as outside it.
- **Routing and travel-time measurement**: a free-flow layer from a local OSRM server, a Google
  Routes API client for adopted travel times, direction-aware schedules, and calibration of the
  free-flow filter against measured times.
- **Tolls**: the toll request shape and a strict parser for the price the API returns.
- **Provenance**: when each measurement was fetched and which batch it belongs to.

The package is a library plus a set of command-line scripts that fetch public data and build the
tables the library reads. It contains no data and no personal configuration; you supply both.

## Installation

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are required.

```sh
git clone https://github.com/nbx-liz/landeval.git
cd landeval
uv sync
```

## The data-root contract

The library never decides where data lives. Every function that reads or writes a file takes the
path from its caller, and no module holds a path as a module-level default.

The scripts under `scripts/` take a data root from `--root <dir>` or, when that is absent, from the
environment variable `LANDEVAL_ROOT`. With neither, a script stops instead of looking in the current
directory. Default data paths (for example `data/raw/admin`) are joined to that root; a path you
pass explicitly keeps its usual meaning, relative to the current directory.

## Usage

`config/example.yaml` is a complete example configuration. It uses only synthetic group names, major
railway stations as destinations and neutral thresholds. Copy it and edit it; `landeval.config`
validates it and rejects unknown keys, missing directions and out-of-range coordinates.

```sh
cp config/example.yaml config/mine.yaml
export LANDEVAL_ROOT="$PWD"

# Fetch the administrative boundaries of the prefectures in the config's `region` section.
uv run python scripts/fetch_admin.py --config config/mine.yaml

# Look up what a free-text address resolves to, and how precisely.
uv run python scripts/geocode.py "<an address>"
```

```python
from pathlib import Path

from landeval.config import load_config
from landeval.mesh import mesh_code, mesh_center

config = load_config(Path("config/example.yaml"))
name, group = config.screening_group()
lat, lon = group.points[0].lat, group.points[0].lon
code = mesh_code(lat, lon, 500)
print(name, code, mesh_center(code))
```

The hazard scripts read a manifest of source files that you write (see `landeval.hazard_sources`
for its grammar): every prefecture of a prefecture-split layer must either list a file or declare
that none is designated. A forgotten entry is a load error, never a silent gap.

## Data sources

Nothing is bundled. Each script downloads from a public source, and you are responsible for
following that source's terms of use and for attributing it where it requires. Check the provider's
current terms before redistributing anything derived from its data.

| Script or module | Source | Notes |
| --- | --- | --- |
| `fetch_admin.py` | Ministry of Land, Infrastructure, Transport and Tourism (MLIT) National Land Numerical Information, administrative boundaries (N03) | Per-prefecture archives. |
| `fetch_hazard.py`, `build_hazard.py` | MLIT National Land Numerical Information hazard datasets (flood A31b, landslide A33, tsunami A40, storm surge A49) | The files to fetch, and the base URL, come from your manifest. |
| `fetch_estat.py` | e-Stat (Statistics Bureau of Japan) small-area boundaries from the census | Archives are in CP932; `landeval.smallarea` states the encoding. |
| `fetch_inland_flood.py` | Inland-flood raster tiles served by the Geospatial Information Authority of Japan (GSI) Disaster Prevention Map portal | Published per municipality; a roster of publishing municipalities is required, because a missing tile alone is ambiguous. |
| `geocode.py`, `landeval.geocode` | GSI address search | |
| `landeval.viz` | GSI map tiles as a basemap | Output maps carry the GSI attribution. |
| `landeval.jshis` | J-SHIS (NIED) web API, earthquake hazard by point | Fetched on demand, one point per request. |
| `fetch_osm.py`, `build_ic_table.py`, `compute_freeflow.py` | OpenStreetMap data from a Geofabrik extract, under the Open Database License (ODbL) | The extract is checksum-verified. `compute_freeflow.py` expects an OSRM server built from the extract, by default at `http://127.0.0.1:5000`; building that server is not part of this repository. |
| `landeval.routes`, `landeval.tolls` | Google Routes API | Requires your own API key; `landeval.routes.api_key_from_env` reads it from the environment variable `GOOGLE_MAPS_API_KEY`. Billed by Google; review the Google Maps Platform terms, including those on caching, before storing responses. |

## Development

```sh
uv sync --locked --extra dev
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

`BLUEPRINT.md` describes the structure, the data contracts and the invariants the public tests
check.

## License

Apache License 2.0. See `LICENSE`.
