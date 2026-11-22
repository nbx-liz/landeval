# landeval blueprint

This document states the structure, responsibilities and invariants of the `landeval` package. It
covers the 28 modules under `src/landeval/`, the command-line scripts under `scripts/`, and what
the tests in `tests/` check. Where this document and the code disagree, the disagreement is a bug
in one of them; the tests are the executable form of the invariants below.

## 1. Purpose and scope

`landeval` provides building blocks for regional screening in property search. It does not
ship a screening run or an address lookup: it ships the pieces such a workflow is assembled
from, each usable and tested on its own:

1. **Cells.** JIS X 0410 grid squares (the mesh), with exact areas.
2. **Free-flow times.** Drive times from cell centres to a destination from a local OSRM server
   (`scripts/compute_freeflow.py` writes them). A cheap filter, not an adopted value.
3. **Hazard overlay.** Published hazard polygons sampled onto cells as a fraction, and an
   earthquake layer read point by point.
4. **Calibration helpers.** Choosing a calibration sample (`sampling`) and turning measured
   times into direction-specific thresholds (`calibration`). The measuring itself is the
   caller's.
5. **Measurement.** A measurement loop over points, destinations and schedules against the
   Google Routes API (`measure`), with provenance, and adopted values for one address bought on
   request (`adopted`).

The library writes no table on its own initiative; the callers, including the scripts in
section 7, decide what to run and where results go.

The package holds no data and no personal configuration. Callers supply both.

## 2. Package boundary and path contract

### 2.1 Boundary rule

The package imports only from an allowlist of top-level modules: the standard library modules it
uses, its declared dependencies, and itself. The allowlist is the measured set, not a guess. In
addition:

- Dynamic import and introspection forms are refused: `importlib`, `__import__`, `builtins`,
  `getattr`, `setattr`, `vars`, `globals`, `modules` and similar names may not appear, and
  neither may `eval`, `exec` or `compile` as bare names.
- Dunder names are refused except `__name__`, `__setattr__` and `__version__`, which closes
  introspection such as `os.__dict__` and keeps a module from deriving a path from its own
  `__file__`.
- A file that cannot be parsed fails the check; it is never skipped.

The check scans source text. It catches a dependency written in the ordinary way; deliberate
evasion is out of its scope. What holds regardless of how a dependency is written is running the
tests in an environment where only the declared dependencies are installed.

### 2.2 Path contract

Callers decide every data path.

- No module holds a `Path` as a module-level value, and no public function or constructor gives a
  path-like parameter a built-in default.
- A module does not compute a location from `__file__` or from the current directory.
- Scripts take a data root from `--root`, or from `LANDEVAL_ROOT` when `--root` is absent. With
  neither they stop. They do not search the current directory, because a decoy `data/` tree there
  would be read without a word. Default data paths are declared relative to the root and joined to
  it after argument parsing (`scripts/_root.py`); a path given on the command line is left alone.

### 2.3 No default encodes one user's policy

Every module-level constant and every non-trivial parameter default is listed in
`publish/constants.toml` and classed as general: a unit, a standard, an API value, a vocabulary or
a method choice, each with a reason. A value that comes from the user's own measurements or
judgement is not a default; the caller passes it. A test compares the inventory with the source in
both directions. Its bound: dataclass `default_factory` values, enum members and literals inside
function bodies are not inventoried.

## 3. Modules

### 3.1 Grid and geography

| Module | Responsibility |
| --- | --- |
| `mesh` | JIS X 0410 grid squares at 1000, 500 and 250 m: code to bounds and centre, point to code, areas as exact integer units so that results do not depend on floating-point noise across machines. Unsupported sizes and out-of-range points raise `MeshError`. |
| `admin` | Attaches a municipality to each cell from MLIT N03 boundaries by a left spatial join: a cell outside every boundary polygon is kept, with a null municipality code and name, never assigned to the nearest one. |
| `smallarea` | Reads town-level (chome) boundaries from e-Stat archives. States the CP932 encoding explicitly; merges polygon parts before counting places; drops non-place rows and reports how many. |
| `geocode` | Calls the GSI address search and separately judges how much of the query the answer accounted for, so a coarse answer cannot pass as a precise one. |

### 3.2 Hazard

| Module | Responsibility |
| --- | --- |
| `hazard_sources` | The reviewed manifest of hazard source files. Every prefecture of a prefecture-split layer is accounted for: a file is expected, or its absence is declared. A prefecture that is listed nowhere is a load error. A layer is either fetchable or declared unavailable, never both and never neither. `HazardSources.prefecture_files` maps each prefecture of a prefecture-split layer to the files expected for it. The archive pin (`HazardPin`, read by `load_hazard_pin`) fixes what the archives hold: every archive the manifest expects, with its sha256, and for every (archive, member fragment) pair a layer reads either the exact `.shp` members or the declaration that there are none. `load_hazard_pin` refuses a missing file, a malformed pin (an entry with both or neither of `members` and `absent`, an empty member list, a member listed under two fragments of one layer, an unknown key or schema), and a pin that does not fit the manifest or the current fragments. It never regenerates the pin and never falls back to the manifest; `scripts/pin_hazard_archives.py` generates it from the archives and checks the archives against it with `--check`. The pin names the archives of one region, so a deployment keeps it beside its own manifest, not in the package. |
| `hazard_overlay` | Drives the overlay across downloaded archives, layer by layer. `run_layer` takes the pin and reads only the members it lists, by exact name; the partial match on a fragment (`shapefile_members`) is used only to generate the pin. A missing archive, an archive whose sha256 differs from the pin, a listed member that is missing or unreadable, a null or empty geometry, and an attribute value outside the published code list all raise `HazardDataError`. A pin entry declared `absent` means the published archive holds no area for that fragment, and reads nothing. `LayerRun.covered` is the set of coverage units -- prefectures for a prefecture-split layer, primary meshes for a mesh-split one -- in which every listed member returned rows; rows that the attribute filter removes in full still count as read, a listed member with no rows leaves its unit out, and a unit whose entries are all `absent` is covered. A layer in which no unit holds any area is not refused: it reads outside wherever it is covered. A layer that filters on an attribute, such as 土砂災害特別警戒区域 on `A33_002`, declares the attribute's complete code list in `OverlaySpec.filter_domain`. Every value is checked against that list before the kept subset is selected, so an unknown or fractional code raises and is never dropped or truncated. A filtering `OverlaySpec` without a code list that contains its kept codes is refused when it is constructed. |
| `hazard` | Samples polygons onto a 5 by 5 sub-grid per 500 m cell and returns a fraction. Rank codes and depth vocabularies are closed grammars: an unrecognised rank code raises `HazardDataError` instead of being mapped to a depth, to zero or to "outside". A layer's status is `outside`, `risk` or `risk_<depth>m`, `not_designated` or `no_data`, and a cell's coverage is `clear`, `partial`, `covered` or `no_data`; the word "safe" appears in neither. `scripts/build_hazard.py` gives each cell a coverage state per layer: covered (its unit is in `LayerRun.covered`), not covered (its prefecture is unknown or has no expected archive, its primary mesh has no expected archive, or its unit is not covered), not designated (decided, and so covered), or the layer is unavailable. A polygon hit keeps its `risk` or `risk_<depth>m` in every state; without a hit a covered cell reads `outside`, a not-designated one `not_designated`, and any other `no_data`. The state is stored as the boolean `<layer>_covered` beside each `<layer>_status`, and in `hazard_points.npz` as `coverage`, one word per cell with the bit position of `membership`; the two agree. `unobtained_mask` and `split_by_coverage` read `<layer>_covered`, and refuse a table without it or with a null in it. `union_ratio` returns NaN for a cell with an unread selected layer, except where every sample point is already inside a zone of a read layer: that cell is 1.0, wholly covered. `hazard_free_ratio` is the share of a cell's sample points inside no zone of any layer the index lists as `available`; unread ground holds no hit, so the ratio overstates the free share wherever one of those layers is not covered, and whether it may be used for a cell is decided by `unobtained_mask` over the layers the reader judges by. A table or index without coverage is refused, never read as unknown. |
| `hazard_points` | The 100 m sample point nearest one address. `SamplePoints.load` is the one reader of `hazard_points.npz`, and refuses an index without `coverage` or with a `coverage` that does not match `mesh_id`. At the point, a set bit is inside the zone whether or not the layer was read for the cell; a clear bit is outside only where the layer was read, and unknown elsewhere. |
| `inland_flood` | Inland-flood raster tiles are published per municipality, so a 404 is ambiguous. A roster of publishing municipalities comes first: inside it a 404 means outside the zone; outside it the cell stays `no_data`; with no roster nothing is ingested. An unreadable roster row is an error, because dropping it would move a publisher into the set whose 404s read as clear. |
| `jshis` | The earthquake layer from the J-SHIS API, one point at a time: ground amplification and the probability of strong shaking, plus the microlandform name. It reports numbers, never an inside/outside designation, and no liquefaction verdict. |

### 3.3 Routing and travel time

| Module | Responsibility |
| --- | --- |
| `osrm` | Client for a local OSRM server. Unreachable pairs stay `None` end to end and are never turned into a large number. The snap distance is reported so that cells snapping to a far road can be rejected. |
| `freeflow` | Free-flow travel times from every cell centre in the candidate area to the screening destination. The candidate area is derived from the destination and the configured horizon, so there is one source of truth for it. Nothing is filtered in place; every row keeps its reason. |
| `interchange` | The nearest expressway interchange read off a route. Connectivity, decided when the interchange table is built, is the rule, not tags alone. The interchange table and the node set carry the same build id and are read as a pair; a mismatch is refused. |
| `basedate` | Chooses the reference date a routing query is asked about: far enough ahead that live traffic is not blended in, and outside every exclusion window (national holidays and the days next to them, the year-end and New Year period, Golden Week and Obon). |
| `routes` | Google Routes API client. The caller passes the key (`api_key_from_env` reads it from `GOOGLE_MAPS_API_KEY`); it is kept out of URLs, cache bodies and error messages, and cache paths are digests of the request, which does not include the key. Arrival-anchored schedules are solved by fixed-point iteration over the departure time; a solution that does not settle says so. |
| `tolls` | The toll request shape (field mask, passes, emission type) and a strict parser of the returned price. Anything outside the supported grammar is an error rather than a zero. |
| `measure` | Runs a measurement over points, destinations and schedules. It lives in the library so it can be tested against a mock transport before any billed run. |
| `measurements` | Merges measurement vintages. The configuration's declared direction is the authority; a stored value that disagrees is an error. |
| `provenance` | Records when each measurement was fetched and which batch it belongs to. A measured row whose fetch time cannot be resolved stops the run. |
| `adopted` | Adopted values for one address, bought from the Routes API on request, and returned as values rather than rows. |
| `adoptedreport` | Formats adopted values in a block separate from, and labelled differently than, estimates. |

### 3.4 Sampling and calibration

| Module | Responsibility |
| --- | --- |
| `sampling` | Chooses calibration points from a bearing by free-flow-time grid, so the sample spans the relationship being fitted; within a grid cell it prefers the point nearest a road. |
| `calibration` | Turns measured times into direction-specific filter thresholds. Each edge can be widened by a margin the caller passes. |
| `stability` | Tests the premise that the API's future-time answer does not vary from day to day, by re-asking cached queries. A schedule with fewer rows than strata is an error rather than a shorter answer. |
| `representatives` | Chooses one measurable point per municipality. Ground outside every selected hazard layer (a `selected_hazard_ratio` of zero) comes first, and a cell with no ratio sorts last; a fallback is recorded in `basis`. The order is fully determined, down to a tie-break on `mesh_id`. |

### 3.5 Configuration, storage and output

| Module | Responsibility |
| --- | --- |
| `config` | The configuration grammar (section 4). |
| `schema` | The storage-table contract (section 5). |
| `viz` | Map output. Cells are rasterised into one image over the basemap, and bands are discrete on purpose. The GSI basemap needs network access. |

## 4. Configuration contract

`landeval.config.load_config` reads human-edited YAML, which is an open grammar, so every off-contract
input is rejected loudly:

- Every model forbids extra keys.
- Each schedule states `id`, `type` (`arrive` or `depart`), `time` (`HH:MM`), `daytype` and
  `direction` (`outbound` or `inbound`). `type` says which end of the trip the clock is pinned to;
  `direction` says which end is the origin. Neither is inferred from the other, and `direction` has
  no default.
- A schedule id is unique across the whole configuration, because storage tables are keyed by
  `schedule_id` without the destination group.
- Exactly one destination group sets `used_for_screening: true`, and
  `thresholds.commute.target_schedule` names a schedule of that group.
- Aggregation rules: `single` needs exactly one point and cannot be combined; `min` needs at least
  two points.
- Thresholds are strictly increasing: `good_max_min < hard_max_min < exclude_min`.
- Coordinates must fall inside Japan, which catches transposed latitude and longitude.
- `vehicle` and `region` are optional as sections, but each field inside them is required with no
  default. A script that needs a region stops when none is stated.

`config/example.yaml` is a valid configuration and is checked by a test.

## 5. Storage contract

`landeval.schema.TABLE_KEYS` declares the tables of the storage contract under a
processed-data directory and the columns that identify one row. Of these, the public scripts write
`freeflow_all`, `freeflow_horizon` and `mesh_hazard`; the others are the contract a caller's own
pipeline writes to. `scripts/fetch_inland_flood.py` reads cells that carry municipality columns
(`final_cells` by default), which only such a pipeline writes:

| Group | Tables | Key |
| --- | --- | --- |
| Cell tables | `combined_cells`, `final_cells`, `freeflow_all`, `freeflow_horizon`, `mesh_hazard`, `screened_cells`, `selected_cells`, `cell_municipality`, `blocks`, `representatives`, `commute_sheet` | `mesh_id` |
| Measurements | `block_measurements`, `calibration`, `municipality_measured`, `municipality_rep_target_times`, `municipality_rep_schedule_times`, `block_rep_target_times`, `block_rep_afternoon_outbound_times`, `block_rep_schedule_times` | `mesh_id`, `schedule_id` |
| Destination measurements | `group_destination_times` | `mesh_id`, `schedule_id`, `destination_name` |
| Municipality summary | `areas_sheet` | `muni_code` |
| Provenance | `fetch_provenance` | `cache_digest` |
| Stability evidence | `stability_check`, `stability_control` | `mesh_id`, `schedule_id`, `quantile` |

The `municipality_rep_*` and `block_rep_*` tables hold raw measured travel times, named for what was
measured: the representative point of a municipality or of a block, and the departure times of the
measurement. `municipality_measured` is a different table, the municipality classification built from
the raw ones.

Keys are declared, not discovered. A table that carries a `schedule_id` always names it in its key,
even while it holds a single value, so that adding a second schedule changes the data and not the
meaning of the key.

The intended column order and dtypes of every table are written out as `EXPECTED_SCHEMA` in
`tests/test_schema_contract_synthetic.py`. The tests there check that the contract's own functions
(`column_types`, `sample_of`, `contract_path`) and a parquet round trip preserve them, on synthetic
frames; they do not check that each writer produces them. `sample_of` sorts by the key and takes
every `len // rows`-th row, at most `rows`, so it is deterministic; a table shorter than twice
`rows` yields its first `rows` rows in key order.

## 6. Invariants

Each invariant is checked by a public test.

- **INV-1, no false clean.** An input that is missing, unparseable or from an unrecognised
  vocabulary fails loudly. It is never rendered as "outside the zone" or "no hazard". A hazard
  layer's status is only `outside`, `risk` or `risk_<depth>m`, `not_designated` or `no_data`.
- **INV-2, unreachable is not slow.** A pair with no route stays `None`; it is never coerced to a
  number or dropped without a record.
- **INV-3, adopted values are labelled as such.** An adopted value comes only from the Routes
  API, on request, and `adoptedreport` prints it in a block of its own that says where its
  numbers came from. A prediction from the calibrated filter (`predicted_min`) is a different
  column, from a different function; keeping a caller's own estimates apart from this block is the
  caller's part.
- **INV-4, direction is explicit.** Every schedule states its direction, and merged vintages never
  assume one.
- **INV-5, thresholds are the caller's.** Screening and calibration thresholds are parameters
  the caller passes; no module holds one as a default (section 2.3). Tables written after a
  threshold was applied, such as `freeflow_horizon` or a `screening_class` column, record that
  choice and are not threshold-free.
- **INV-6, determinism.** Mesh areas are integer units, representative choice and calibration
  sampling are fully ordered, so the same input gives the same output.
- **INV-7, the key never persists.** The Routes API key, passed by the caller, appears in no
  request URL, cache body or error message.
- **INV-8, a pair is read as a pair.** The interchange table and its node set, written as two
  files by `scripts/build_ic_table.py`, share a build id, and `landeval.interchange` refuses a
  mismatched pair.
- **INV-9, callers decide paths.** See section 2.2.
- **INV-10, no policy defaults.** See section 2.3.
- **INV-11, the boundary holds.** See section 2.1.
- **INV-12, the tolled shape is exact.** Toll passes and emission types are validated against known
  spellings; a response outside the supported price grammar raises instead of returning a number.

## 7. Scripts

| Script | Role |
| --- | --- |
| `_root.py` | Data-root resolution shared by the scripts (section 2.2). |
| `fetch_admin.py`, `fetch_estat.py`, `fetch_hazard.py`, `fetch_inland_flood.py`, `fetch_osm.py` | Download public data. A failed download is reported and makes the script exit non-zero; it is never read as data. For inland-flood tiles a 404 is read through the roster (section 3.2). |
| `build_hazard.py`, `build_ic_table.py`, `compute_freeflow.py` | Build the tables the library reads. `build_hazard.py` checks the archive pin and every pinned archive before any layer is overlaid, and on any failure exits non-zero without writing. |
| `pin_hazard_archives.py` | Generates the archive pin from the downloaded archives (or from an inventory report written by the same reader), and with `--check` counts the differences between the archives and a pin. It never overwrites a file and writes nothing when anything could not be read. |
| `build_q250_units.py` | Regenerates the 250 m area fixture that the mesh tests compare against. |
| `geocode.py` | Terminal front for `landeval.geocode`. |
| `inspect_hazard_schema.py`, `probe_gml_read.py`, `probe_hazard_sources.py`, `survey_depth_vocabulary.py` | One-off checks of source data and manifests. They are run to close a vocabulary against real data rather than to widen a parser one rejected string at a time. |

## 8. Tests

The public tests run on the exported tree alone, with synthetic or checked-in fixtures: the
package boundary, the path contract, the constants inventory, the configuration grammar and the
example configuration, the storage contract, the interchange build-id contract, and each module's
behaviour against mock transports. No test needs network access, a data tree, or an API key.
