"""Callers decide every data path.

Two halves, both behavioural:

* **Scripts.** Each script that reads or writes data takes its root from
  ``--root`` or ``LANDEVAL_ROOT`` and stops when it gets neither. A decoy ``data/``
  tree sits in the working directory, so a script that fell back to the current
  directory would find plausible files instead of failing.
* **Library.** No public module holds a ``Path`` at module level, and no public
  function or constructor gives a path-like parameter a built-in default.

The library check has a stated bound: it inspects module constants and parameter
defaults, not every path a function body might construct. The body of a function
that builds ``Path("data/...")`` itself is covered by a grep for
``Path("data`` under ``src`` (which returns nothing) and by the scripts' behaviour, not
by this file.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import os
import pkgutil
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path, PurePath
from types import ModuleType

import pytest

import landeval

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

NO_DATA = "NO_DATA"

# For each script: the smallest argv that gets past argparse to root resolution, or
# NO_DATA for a script that touches no data. Positional and required arguments are
# here so that an argparse exit 2 cannot stand in for the refusal under test.
SCRIPT_ARGV: dict[str, list[str] | str] = {
    "build_hazard.py": [],
    "pin_hazard_archives.py": [],
    "build_ic_table.py": [],
    "build_q250_units.py": [],
    "compute_freeflow.py": [],
    "fetch_admin.py": [],
    "fetch_estat.py": [],
    "fetch_hazard.py": [],
    "fetch_inland_flood.py": [],
    "fetch_osm.py": [],
    "geocode.py": ["Tokyo"],
    "inspect_hazard_schema.py": ["--layer", "flood"],
    "probe_gml_read.py": ["absent.zip"],
    "probe_hazard_sources.py": [],
    "survey_depth_vocabulary.py": [],
}

DATA_SCRIPTS = sorted(name for name, argv in SCRIPT_ARGV.items() if argv != NO_DATA)

# Names that look like a data file in the decoy tree, so that a read from the
# working directory would succeed in finding something rather than fail early.
DECOY_FILES = (
    "data/processed/final_cells.parquet",
    "data/processed/mesh_hazard.parquet",
    "data/processed/freeflow_horizon.parquet",
    "data/raw/admin/N03.shp",
    "data/raw/japan-latest.osm.pbf",
    "data/cache/routes/index.json",
    "config/destinations.yaml",
    "config/hazard_sources.yaml",
)


def _load_root_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_root_under_test", SCRIPTS / "_root.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT_HELPER = _load_root_helper()


@pytest.fixture
def decoy_cwd(tmp_path: Path) -> Path:
    for relative in DECOY_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"decoy")
    return tmp_path


@pytest.fixture
def docker_stub(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ``docker`` first on ``PATH`` that records a call. Returns the record's path.

    The shell scripts must stop before docker. If one regressed, the stub keeps it
    from touching the real daemon and the record makes the call visible.
    """
    bin_dir = tmp_path_factory.mktemp("stub-bin")
    sentinel = bin_dir / "docker-was-called"
    stub = bin_dir / "docker"
    stub.write_text(f'#!/bin/sh\ntouch "{sentinel}"\nexit 99\n')
    stub.chmod(0o755)
    return sentinel


def _stubbed_path(docker_stub: Path) -> str:
    return f"{docker_stub.parent}{os.pathsep}{os.environ.get('PATH', '')}"


def _env_without_root(**extra: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != ROOT_HELPER.ROOT_ENV}
    env.update(extra)
    return env


def _interpreter(script: str) -> str:
    if script.endswith(".sh"):
        bash = shutil.which("bash")
        assert bash is not None, "bash is needed to run the shell scripts"
        return bash
    return sys.executable


def _run(
    script: str, argv: list[str], cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [_interpreter(script), str(SCRIPTS / script), *argv],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=120,
        check=False,
    )


# --- the table covers the scripts on disk -----------------------------------


PUBLIC_PROJECT = "landeval"


def project_name(root: Path) -> str:
    """The ``[project].name`` of the tree's ``pyproject.toml``, which tells the layouts apart."""
    name = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
    assert isinstance(name, str)
    return name


def scripts_on_disk() -> set[str]:
    return {
        path.name
        for pattern in ("*.py", "*.sh")
        for path in SCRIPTS.glob(pattern)
        if path.name != "_root.py"
    }


def test_the_script_table_names_exactly_the_scripts_on_disk() -> None:
    """In the public tree, this file's table is every script there is.

    A tree whose project is not ``landeval`` holds further scripts that this table does not
    list; there, the equality over both tables is checked by the tests that list the others.
    """
    if project_name(REPO_ROOT) != PUBLIC_PROJECT:
        pytest.skip(
            "not the public tree: the table-versus-disk equality of this tree is checked "
            "by its own tests, which list the scripts this table does not"
        )

    assert set(SCRIPT_ARGV) == scripts_on_disk()


# --- scripts ----------------------------------------------------------------


@pytest.mark.parametrize("script", DATA_SCRIPTS)
def test_a_script_given_no_root_fails_even_with_data_in_the_working_directory(
    script: str, decoy_cwd: Path, docker_stub: Path
) -> None:
    argv = SCRIPT_ARGV[script]
    assert isinstance(argv, list)

    result = _run(script, argv, decoy_cwd, _env_without_root(PATH=_stubbed_path(docker_stub)))

    assert result.returncode != 0
    assert ROOT_HELPER.MISSING_ROOT_MESSAGE in result.stderr, result.stderr[-800:]
    if script.endswith(".sh"):
        # A shell script carries its own copy of the message; this pins the copy.
        assert result.stderr.strip() == ROOT_HELPER.MISSING_ROOT_MESSAGE
    assert not docker_stub.exists()
    assert not (decoy_cwd / "data" / "osrm").exists()


def test_an_empty_landeval_root_counts_as_unset(decoy_cwd: Path) -> None:
    result = _run("build_hazard.py", [], decoy_cwd, _env_without_root(LANDEVAL_ROOT=""))

    assert result.returncode != 0
    assert ROOT_HELPER.MISSING_ROOT_MESSAGE in result.stderr


def test_an_empty_root_option_counts_as_unset(decoy_cwd: Path) -> None:
    """``--root ""`` must not become the current directory through ``Path("")``."""
    result = _run("build_hazard.py", ["--root", ""], decoy_cwd, _env_without_root())

    assert result.returncode != 0
    assert ROOT_HELPER.MISSING_ROOT_MESSAGE in result.stderr


@pytest.mark.parametrize("source", ["option", "environment"])
def test_a_root_that_is_not_a_directory_fails(
    source: str, decoy_cwd: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    missing = tmp_path_factory.mktemp("elsewhere") / "absent"
    if source == "option":
        result = _run("build_hazard.py", ["--root", str(missing)], decoy_cwd, _env_without_root())
    else:
        result = _run(
            "build_hazard.py", [], decoy_cwd, _env_without_root(LANDEVAL_ROOT=str(missing))
        )

    assert result.returncode != 0
    assert ROOT_HELPER.NOT_A_DIRECTORY_MESSAGE in result.stderr


def test_the_root_option_wins_over_the_environment(
    decoy_cwd: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    good = tmp_path_factory.mktemp("good")
    missing = good / "absent"
    result = _run(
        "build_hazard.py",
        ["--root", str(missing)],
        decoy_cwd,
        _env_without_root(LANDEVAL_ROOT=str(good)),
    )

    assert ROOT_HELPER.NOT_A_DIRECTORY_MESSAGE in result.stderr


# --- the helper itself ------------------------------------------------------


def _namespace(**values: object) -> argparse.Namespace:
    return argparse.Namespace(**values)


def test_the_environment_root_is_used_when_the_option_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ROOT_HELPER.ROOT_ENV, str(tmp_path))
    args = _namespace(root=None)

    assert ROOT_HELPER.resolve_root(args) == tmp_path.resolve()


def test_defaults_are_joined_to_the_root_and_a_given_path_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ROOT_HELPER.ROOT_ENV, raising=False)
    given = Path("relative/to/the/cwd.parquet")
    args = _namespace(
        root=str(tmp_path),
        cells=ROOT_HELPER.under_root("data/processed/final_cells.parquet"),
        out=given,
    )

    root = ROOT_HELPER.resolve_root(args)

    assert args.cells == root / "data/processed/final_cells.parquet"
    assert args.out == given


def test_no_root_raises_system_exit_with_the_fixed_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ROOT_HELPER.ROOT_ENV, raising=False)

    with pytest.raises(SystemExit) as raised:
        ROOT_HELPER.resolve_root(_namespace(root=None))

    assert raised.value.code == ROOT_HELPER.MISSING_ROOT_MESSAGE


# --- library ----------------------------------------------------------------


def _public_modules() -> list[ModuleType]:
    modules = []
    for package in (landeval,):
        for info in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
            if info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            modules.append(importlib.import_module(info.name))
    return modules


PUBLIC_MODULES = _public_modules()
MODULE_IDS = [module.__name__ for module in PUBLIC_MODULES]

PATH_SUFFIXES = ("dir", "path", "root", "file")


@pytest.mark.parametrize("module", PUBLIC_MODULES, ids=MODULE_IDS)
def test_no_module_level_attribute_is_a_path(module: ModuleType) -> None:
    held = sorted(name for name, value in vars(module).items() if isinstance(value, PurePath))

    assert held == []


def _callables_of(module: ModuleType) -> list[tuple[str, inspect.Signature]]:
    found = []
    for name, value in vars(module).items():
        if name.startswith("_") or getattr(value, "__module__", None) != module.__name__:
            continue
        target = value.__init__ if inspect.isclass(value) else value
        if not (inspect.isfunction(target) or inspect.ismethod(target)):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        found.append((name, signature))
    return found


def _is_path_like(parameter: inspect.Parameter) -> bool:
    name = parameter.name.lower()
    annotation = (
        "" if parameter.annotation is inspect.Parameter.empty else str(parameter.annotation)
    )
    return name.endswith(PATH_SUFFIXES) or "Path" in annotation


@pytest.mark.parametrize("module", PUBLIC_MODULES, ids=MODULE_IDS)
def test_no_path_like_parameter_has_a_default_other_than_none(module: ModuleType) -> None:
    offenders = [
        f"{name}({parameter.name}={parameter.default!r})"
        for name, signature in _callables_of(module)
        for parameter in signature.parameters.values()
        if _is_path_like(parameter)
        and parameter.default is not inspect.Parameter.empty
        and parameter.default is not None
    ]

    assert offenders == []


def test_the_library_check_sees_the_constructors_that_took_a_default_cache_dir() -> None:
    """Guards the guard: if the walk found nothing it would pass on every module."""
    seen = {
        f"{module.__name__}.{name}"
        for module in PUBLIC_MODULES
        for name, _ in _callables_of(module)
    }

    assert {"landeval.routes.RoutesClient", "landeval.jshis.JshisClient"} <= seen
    assert "landeval.inland_flood.TileClient" in seen
