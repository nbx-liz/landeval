"""Package boundary: ``landeval`` depends on nothing outside its import allowlist."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import landeval

PUBLIC_MODULES = frozenset(
    {
        "admin",
        "adopted",
        "adoptedreport",
        "basedate",
        "calibration",
        "config",
        "freeflow",
        "geocode",
        "hazard",
        "hazard_overlay",
        "hazard_points",
        "hazard_sources",
        "inland_flood",
        "interchange",
        "jshis",
        "measure",
        "measurements",
        "mesh",
        "osrm",
        "provenance",
        "representatives",
        "routes",
        "sampling",
        "schema",
        "smallarea",
        "stability",
        "tolls",
        "viz",
    }
)
# What this guard claims, and what it does not.
#
# A computed import target ("pkg" + "_name") cannot be resolved by reading the
# source, so the scan does not try. Instead the grammar is closed in three ways:
#
# 1. ALLOWED_IMPORTS: the package may import only these top-level modules,
#    measured as exactly the set its modules use. builtins, importlib, sys,
#    pkgutil and runpy are not on it.
# 2. DYNAMIC_NAMES: these names may not appear at all, whether as a name, an
#    attribute or an imported alias. No module uses one.
# 3. ALLOWED_DUNDERS: a dunder name or attribute (__dict__, __class__,
#    __globals__, ...) is refused unless it is one of the three the modules
#    use. This closes introspection such as os.__dict__, and it keeps a module
#    from deriving a path from its own __file__.
#
# Adding to any set is a reviewed change to this test.
#
# The bound: the guard catches a dependency an author writes in the ordinary
# way -- an import, an importlib or builtins call. Deliberate evasion is out of
# scope and is left to review: a static scan of Python cannot close it. What
# holds however a dependency is written is testing the package in an
# environment where nothing outside its declared dependencies is installed.
ALLOWED_IMPORTS = frozenset(
    {
        "PIL",
        "__future__",
        "base64",
        "collections",
        "csv",
        "dataclasses",
        "datetime",
        "enum",
        "folium",
        "functools",
        "geopandas",
        "hashlib",
        "httpx",
        "io",
        "jpholiday",
        "json",
        "landeval",
        "math",
        "numpy",
        "os",
        "pandas",
        "pathlib",
        "pydantic",
        "pyproj",
        "re",
        "shapely",
        "string",
        "tempfile",
        "time",
        "typing",
        "unicodedata",
        "urllib",
        "yaml",
        "zipfile",
    }
)
DYNAMIC_NAMES = frozenset(
    {
        "__builtins__",
        "__import__",
        "__loader__",
        "__spec__",
        "builtins",
        "compile",
        "eval",
        "exec",
        "find_spec",
        "getattr",
        "globals",
        "import_module",
        "importlib",
        "modules",
        "setattr",
        "vars",
    }
)
# Checked as a name or an alias, not as an attribute: re.compile is ordinary
# code. As an attribute they can only be reached through builtins or
# __builtins__, and both of those are already refused.
BARE_BUILTINS = frozenset({"compile", "eval", "exec"})
ALLOWED_DUNDERS = frozenset({"__name__", "__setattr__", "__version__"})


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _modules_on_disk(package_file: str | None) -> set[str]:
    """Dotted module names under the package, recursively, relative to it."""
    assert package_file is not None
    root = Path(package_file).parent
    names: set[str] = set()
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.add(".".join(parts))
    return names


def _is_forbidden_name(name: str, forbidden: frozenset[str]) -> bool:
    return any(name == f or name.startswith(f + ".") for f in forbidden)


def _import_violations(
    node: ast.Import | ast.ImportFrom, forbidden: frozenset[str] = frozenset()
) -> list[str]:
    if isinstance(node, ast.Import):
        targets = [a.name for a in node.names]
    elif node.level > 0:
        # A relative import stays inside landeval by construction; its names
        # are still checked below.
        targets = []
    else:
        targets = [node.module or ""]
    found = [
        t
        for t in targets
        if _is_forbidden_name(t, forbidden) or t.split(".")[0] not in ALLOWED_IMPORTS
    ]
    for alias in node.names:
        found += [n for n in (alias.name, alias.asname) if n in DYNAMIC_NAMES]
    return found


def _violations(tree: ast.AST, forbidden: frozenset[str] = frozenset()) -> list[str]:
    """Every refused form in ``tree``. ``forbidden`` names packages that are also
    refused when a string constant merely mentions them."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            found += _import_violations(node, forbidden)
        elif isinstance(node, ast.Name) and (
            node.id in DYNAMIC_NAMES or (_is_dunder(node.id) and node.id not in ALLOWED_DUNDERS)
        ):
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and (
            node.attr in DYNAMIC_NAMES - BARE_BUILTINS
            or (_is_dunder(node.attr) and node.attr not in ALLOWED_DUNDERS)
        ):
            found.append(node.attr)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(f in node.value for f in forbidden)
        ):
            found.append(node.value)
    return found


def test_public_package_never_references_local_package() -> None:
    """Every module stays inside the allowlist and refuses the dynamic forms."""
    root = Path(landeval.__file__ or "").parent
    files = sorted(root.rglob("*.py"))
    assert files, "scanned zero files under src/landeval"
    bad: dict[str, list[str]] = {}
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))  # fail closed
        hits = _violations(tree)
        if hits:
            bad[str(f)] = hits
    assert not bad, f"landeval must stay inside its import allowlist: {bad}"


def test_scanner_detects_forbidden_forms() -> None:
    for src in (
        "import other_pkg",
        "import other_pkg.sub",
        "from other_pkg import sub",
        "from other_pkg.sub import x",
        "import importlib\nimportlib.import_module('other_pkg.sub')",
        "import importlib\nimportlib.import_module('other_pkg')",
        # Computed targets: rejected for using the machinery, not by resolving it.
        "import importlib\nm = importlib.import_module(__package__ + '_x')",
        "from importlib import import_module\nimport_module('other' + '_pkg')",
        "m = __import__('other' + '_pkg')",
        "import sys\nm = sys.modules['other' + '_pkg']",
        "exec('import other' + '_pkg')",
        "import importlib.util\nimportlib.util.find_spec('x')",
        "import runpy",
        "import pkgutil",
        # Execution builtins reached as an attribute or an alias.
        "import builtins\nbuiltins.exec('import other' + '_pkg', {})",
        "from builtins import exec as execute\nexecute('x')",
        "from os import path as exec",
        "f = getattr(__builtins__, 'ex' + 'ec')",
        "g = globals()['__builti' + 'ns__']",
        "import sys",
        "import subprocess",
        # The importer reached by introspection through an allowed module.
        "import os\nos.__dict__['__builti' + 'ns__']['__im' + 'port__']('x')",
        "x = ().__class__.__base__.__subclasses__()",
        "def f():\n    return f.__globals__",
        # The library may not derive a path from its own location.
        "from pathlib import Path\nROOT = Path(__file__).parents[2]",
    ):
        assert _violations(ast.parse(src)), src
    # A forbidden name is refused even when a string merely mentions it.
    assert _violations(ast.parse("x = 'see other_pkg here'"), forbidden=frozenset({"other_pkg"}))
    for clean in (
        "from landeval import mesh",
        "import json\nx = json.loads('{}')",
        "from pathlib import Path\nfrom . import mesh",
        "import numpy as np\nimport pandas as pd",
        "import re\npattern = re.compile('x')",
        "log = __name__\nobject.__setattr__(self, 'a', 1)",
    ):
        assert not _violations(ast.parse(clean)), clean


def test_allowed_imports_equal_what_the_public_package_uses() -> None:
    """The allowlist is the measured set, not a guess: an entry no module uses
    any more is dead weight that widens the bound for nothing."""
    used: set[str] = set()
    for f in Path(landeval.__file__ or "").parent.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                used.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                used.add(node.module.split(".")[0])
    assert used == ALLOWED_IMPORTS


def test_allowed_dunders_equal_what_the_public_package_uses() -> None:
    used: set[str] = set()
    for f in Path(landeval.__file__ or "").parent.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Name) and _is_dunder(node.id):
                used.add(node.id)
            elif isinstance(node, ast.Attribute) and _is_dunder(node.attr):
                used.add(node.attr)
    assert used == ALLOWED_DUNDERS


def test_unparseable_file_fails_closed() -> None:
    with pytest.raises(SyntaxError):
        ast.parse("def (:", filename="broken.py")


@pytest.mark.parametrize("name", sorted(PUBLIC_MODULES))
def test_public_module_lives_in_public_package(name: str) -> None:
    assert importlib.import_module(f"landeval.{name}") is not None


def test_module_inventory_reads_subpackages(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    for rel in ("__init__.py", "a.py", "sub/__init__.py", "sub/b.py"):
        (pkg / rel).write_text("", encoding="utf-8")
    assert _modules_on_disk(str(pkg / "__init__.py")) == {"a", "sub", "sub.b"}


def test_module_lists_match_disk() -> None:
    assert _modules_on_disk(landeval.__file__) == PUBLIC_MODULES
