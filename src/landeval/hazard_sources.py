"""The reviewed inventory of hazard source files.

Hazard data is the canonical false-clean surface: a file that fails to arrive must not
end up rendered as "this area is safe". The manifest closes that grammar. Every
prefecture of a prefecture-split layer is accounted for exactly once -- either a
file is expected, or its absence is declared with a reason -- so a forgotten
prefecture is a load error rather than a quiet hole in the map.

The tsunami layer is why this exists rather than a URL template: each
prefecture issues its 津波浸水想定 on its own schedule, so the year suffix
differs per prefecture and a uniform guess 404s for any prefecture whose
year differs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MESH_CODE_PATTERN = r"^\d{4}$"
PREF_CODE_PATTERN = r"^\d{2}$"

Split = Literal["primary_mesh", "prefecture"]


class HazardSourceError(Exception):
    """Raised for a manifest that cannot be read or does not satisfy the contract."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Layer(_Strict):
    """One hazard layer and where its files come from.

    A layer is either fetchable (``split`` plus ``patterns`` or ``files``) or
    declared unavailable. Never both, and never neither -- an unstated layer
    would have no defined value for the cells it covers.
    """

    label: str = Field(min_length=1)
    split: Split | None = None
    patterns: tuple[str, ...] = ()
    files: dict[str, str] = Field(default_factory=dict)
    not_designated: dict[str, str] = Field(default_factory=dict)
    unavailable: str | None = None

    @model_validator(mode="after")
    def _fetchable_or_declared_unavailable(self) -> Self:
        if self.unavailable is not None:
            if self.split is not None or self.patterns or self.files or self.not_designated:
                raise ValueError(
                    f"layer {self.label!r} is marked unavailable but also declares sources"
                )
            return self
        if self.split is None:
            raise ValueError(f"layer {self.label!r} needs either `split` or `unavailable`")
        if bool(self.patterns) == bool(self.files):
            raise ValueError(
                f"layer {self.label!r} must declare exactly one of `patterns` or `files`"
            )
        if self.split == "primary_mesh":
            if self.files or self.not_designated:
                raise ValueError(
                    f"layer {self.label!r} is mesh-split; per-prefecture keys do not apply"
                )
            if not all("{mesh}" in pattern for pattern in self.patterns):
                raise ValueError(f"layer {self.label!r} patterns must contain '{{mesh}}'")
        elif self.patterns and not all("{pref}" in pattern for pattern in self.patterns):
            raise ValueError(f"layer {self.label!r} patterns must contain '{{pref}}'")
        overlap = set(self.files) & set(self.not_designated)
        if overlap:
            raise ValueError(
                f"layer {self.label!r} lists {sorted(overlap)} as both present and not designated"
            )
        return self

    @property
    def is_available(self) -> bool:
        return self.unavailable is None


class HazardSources(_Strict):
    """The whole manifest."""

    base_url: str = Field(pattern=r"^https://")
    primary_meshes: tuple[str, ...] = Field(min_length=1)
    prefectures: dict[str, str] = Field(min_length=1)
    layers: dict[str, Layer] = Field(min_length=1)

    @model_validator(mode="after")
    def _codes_are_well_formed(self) -> Self:
        for mesh in self.primary_meshes:
            if not re.match(MESH_CODE_PATTERN, mesh):
                raise ValueError(f"primary mesh code {mesh!r} is not four digits")
        for code in self.prefectures:
            if not re.match(PREF_CODE_PATTERN, code):
                raise ValueError(f"prefecture code {code!r} is not two digits")
        return self

    @model_validator(mode="after")
    def _every_prefecture_is_accounted_for(self) -> Self:
        """The false-clean closure: no prefecture may be silently missing from a layer."""
        for name, layer in self.layers.items():
            if not layer.is_available or layer.split != "prefecture":
                continue
            covered = set(layer.files) | set(layer.not_designated)
            if layer.patterns:
                covered |= set(self.prefectures)
            missing = set(self.prefectures) - covered
            if missing:
                raise ValueError(
                    f"layer {name!r} does not say what happens for prefecture(s) "
                    f"{sorted(missing)}: list them under `files` or `not_designated`"
                )
            unknown = covered - set(self.prefectures)
            if unknown:
                raise ValueError(
                    f"layer {name!r} references unknown prefecture(s) {sorted(unknown)}"
                )
        return self

    def expected_files(self, layer_name: str) -> dict[str, str]:
        """Map of key -> path for the files a layer expects to be able to fetch."""
        layer = self.layers[layer_name]
        if not layer.is_available:
            return {}
        if layer.split == "primary_mesh":
            return {
                f"{mesh}_{index}": pattern.format(mesh=mesh)
                for index, pattern in enumerate(layer.patterns)
                for mesh in self.primary_meshes
            }
        if layer.patterns:
            return {
                f"{code}_{index}": pattern.format(pref=code)
                for index, pattern in enumerate(layer.patterns)
                for code in self.prefectures
            }
        return dict(layer.files)

    def prefecture_files(self, layer_name: str) -> dict[str, tuple[str, ...]]:
        """Map of prefecture code -> every file a prefecture-split layer expects for it.

        Prefectures declared not designated have no entry: nothing is expected
        for them. The code comes from the manifest structure, never from a key
        or file name, so it cannot be misread from a path.
        """
        layer = self.layers[layer_name]
        if not layer.is_available or layer.split != "prefecture":
            raise HazardSourceError(f"layer {layer_name!r} is not an available prefecture split")
        if layer.patterns:
            return {
                code: tuple(pattern.format(pref=code) for pattern in layer.patterns)
                for code in self.prefectures
            }
        return {code: (path,) for code, path in layer.files.items()}

    def mesh_files(self, layer_name: str) -> dict[str, tuple[str, ...]]:
        """Map of primary mesh code -> every file a mesh-split layer expects for it.

        The code comes from the manifest structure, as in :meth:`prefecture_files`.
        """
        layer = self.layers[layer_name]
        if not layer.is_available or layer.split != "primary_mesh":
            raise HazardSourceError(f"layer {layer_name!r} is not an available mesh split")
        return {
            mesh: tuple(pattern.format(mesh=mesh) for pattern in layer.patterns)
            for mesh in self.primary_meshes
        }

    def unit_files(self, layer_name: str) -> dict[str, tuple[str, ...]]:
        """Map of coverage unit -> the files a layer expects for it.

        The unit is a prefecture for a prefecture-split layer and a primary
        mesh for a mesh-split one. An unavailable layer has no units.
        """
        layer = self.layers[layer_name]
        if not layer.is_available:
            return {}
        if layer.split == "primary_mesh":
            return self.mesh_files(layer_name)
        return self.prefecture_files(layer_name)

    def archive_keys(self, layer_name: str) -> set[str]:
        """Every archive a layer expects, as ``<layer>/<file name>`` under the raw root.

        That is where ``scripts/fetch_hazard.py`` puts a file and where the
        overlay opens it, so it is also the key the archive pin uses.
        """
        return {
            archive_key(layer_name, path)
            for paths in self.unit_files(layer_name).values()
            for path in paths
        }

    def url_for(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path}"


def archive_key(layer_name: str, path: str) -> str:
    """The raw-root relative path of one manifest file: ``<layer>/<file name>``."""
    return f"{layer_name}/{Path(path).name}"


def load_hazard_sources(path: Path) -> HazardSources:
    """Read and validate the manifest, raising ``HazardSourceError`` on any problem."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HazardSourceError(f"hazard source manifest not found: {path}") from error
    except yaml.YAMLError as error:
        raise HazardSourceError(f"hazard source manifest is not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise HazardSourceError(
            f"hazard source manifest must be a mapping, got {type(raw).__name__}"
        )
    try:
        return HazardSources.model_validate(raw)
    except ValidationError as error:
        raise HazardSourceError(
            f"hazard source manifest does not satisfy the contract: {error}"
        ) from error


# --- the archive pin ----------------------------------------------------------
#
# The manifest says which archives a layer expects. It cannot say what is inside
# them, and "listed" is not "read". The pin closes that: for every (archive,
# member fragment) a layer reads, it declares either the exact `.shp` members
# that match the fragment or that none does. It is generated from the archives
# on disk by `scripts/pin_hazard_archives.py` and never by the build. Each
# archive is fixed by its sha256, so the member lists cannot drift from the
# bytes they were read from.


class HazardPinError(Exception):
    """Raised for an archive pin that is missing, malformed, or does not fit the manifest."""


class PinEntry(_Strict):
    """What one (archive, fragment) pair holds: the exact members, or none."""

    members: tuple[str, ...] | None = None
    absent: Literal[True] | None = None

    @model_validator(mode="before")
    @classmethod
    def _exactly_one_key_and_not_null(cls, data: Any) -> Any:
        # Decided on the keys the entry carries, not on the parsed values: an
        # explicit null is still a key, so `{"members": null, "absent": true}`
        # holds both and `{"members": null}` holds one that says nothing. Each
        # would otherwise read as the other key alone.
        if not isinstance(data, Mapping):
            return data
        present = [key for key in ("members", "absent") if key in data]
        if len(present) != 1:
            raise ValueError("an entry must hold exactly one of `members` or `absent`")
        if data[present[0]] is None:
            raise ValueError(f"`{present[0]}` must not be null")
        return data

    @model_validator(mode="after")
    def _exactly_one_of_members_or_absent(self) -> Self:
        if (self.members is None) == (self.absent is None):
            raise ValueError("an entry must hold exactly one of `members` or `absent`")
        if self.members is not None:
            if not self.members:
                raise ValueError("`members` must list at least one member")
            if len(set(self.members)) != len(self.members):
                raise ValueError(f"`members` lists a member twice: {list(self.members)}")
        return self


class PinArchive(_Strict):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PinLayer(_Strict):
    """One output layer: the fragments it was pinned with, and one entry per pair."""

    fragments: tuple[str, ...] = Field(min_length=1)
    entries: dict[str, dict[str, PinEntry]]

    @model_validator(mode="after")
    def _entries_name_only_known_fragments_and_each_member_once(self) -> Self:
        known = set(self.fragments)
        if len(known) != len(self.fragments):
            raise ValueError(f"fragments are listed twice: {list(self.fragments)}")
        for archive, by_fragment in self.entries.items():
            unknown = set(by_fragment) - known
            if unknown:
                raise ValueError(f"{archive} has entries for unknown fragment(s) {sorted(unknown)}")
            seen: dict[str, str] = {}
            for fragment, entry in by_fragment.items():
                for member in entry.members or ():
                    if member in seen:
                        raise ValueError(
                            f"{archive}: member {member!r} is listed under both "
                            f"{seen[member]!r} and {fragment!r}"
                        )
                    seen[member] = fragment
        return self


class HazardPin(_Strict):
    """The whole pin file."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    pin_schema: Literal["hazard-archive-pin/v1"] = Field(alias="schema")
    archives: dict[str, PinArchive]
    layers: dict[str, PinLayer]

    @model_validator(mode="after")
    def _entries_name_pinned_archives(self) -> Self:
        for name, layer in self.layers.items():
            unpinned = set(layer.entries) - set(self.archives)
            if unpinned:
                raise ValueError(
                    f"layer {name!r} has entries for archives without a sha256: {sorted(unpinned)}"
                )
        return self

    def entry(self, layer: str, archive: str, fragment: str) -> PinEntry:
        return self.layers[layer].entries[archive][fragment]

    def check_against(
        self, sources: HazardSources, fragments: Mapping[str, tuple[str, tuple[str, ...]]]
    ) -> None:
        """Refuse a pin that does not fit the manifest or the overlay's fragments.

        ``fragments`` maps each output layer that reads archives to its source
        layer and its member fragments (``OverlaySpec.member_contains``).
        """
        expected = {
            key
            for source, layer in sources.layers.items()
            if layer.is_available
            for key in sources.archive_keys(source)
        }
        pinned = set(self.archives)
        if pinned != expected:
            raise HazardPinError(
                "the pin's archives are not the manifest's: "
                f"unpinned {sorted(expected - pinned)}, not expected {sorted(pinned - expected)}"
            )
        if set(self.layers) != set(fragments):
            raise HazardPinError(
                f"the pin covers layers {sorted(self.layers)}, the overlay reads "
                f"{sorted(fragments)}"
            )
        for name, (source, wanted) in fragments.items():
            layer = self.layers[name]
            if tuple(layer.fragments) != tuple(wanted):
                raise HazardPinError(
                    f"layer {name!r} was pinned with fragments {list(layer.fragments)}, "
                    f"the overlay now reads {list(wanted)}; regenerate the pin"
                )
            archives = sources.archive_keys(source)
            stray = set(layer.entries) - archives
            if stray:
                raise HazardPinError(
                    f"layer {name!r} has entries for archives of another layer: {sorted(stray)}"
                )
            for archive in sorted(archives):
                missing = [f for f in wanted if f not in layer.entries.get(archive, {})]
                if missing:
                    raise HazardPinError(
                        f"layer {name!r} has no pin entry for {archive} and fragment(s) {missing}"
                    )


def load_hazard_pin(
    path: Path,
    *,
    sources: HazardSources,
    fragments: Mapping[str, tuple[str, tuple[str, ...]]],
) -> HazardPin:
    """Read the archive pin and check it against the manifest and the fragments.

    Raises ``HazardPinError`` when the file is missing, is not a valid pin, or
    does not fit ``sources`` and ``fragments``. It never regenerates the pin
    and never falls back to the manifest alone.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise HazardPinError(
            f"hazard archive pin not found: {path}; generate it with scripts/pin_hazard_archives.py"
        ) from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise HazardPinError(f"hazard archive pin is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise HazardPinError(f"hazard archive pin must be an object, got {type(raw).__name__}")
    try:
        pin = HazardPin.model_validate(raw)
    except ValidationError as error:
        raise HazardPinError(
            f"hazard archive pin does not satisfy the contract: {error}"
        ) from error
    pin.check_against(sources, fragments)
    return pin


def sha256_of(path: Path) -> str:
    """The sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
