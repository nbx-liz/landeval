"""The package keeps no default that encodes one user's own policy.

``publish/constants.toml`` lists every module-level constant and every
parameter default whose value is not ``None``, ``True``, ``False`` or an empty
container, for each module of ``landeval``. A constant is a name matching
``^_*[A-Z][A-Z0-9_]*$``, so ``_PRIVATE`` constants count too. Each entry is
classed ``"general"`` (a unit, a standard, an API value, a vocabulary, a method
choice) and says why. A value set from a user's own measurements or judgement
would be ``"personal"``: the caller has to pass it, so this test refuses it.

Bound of this test: it sees

* module-level bindings: every name bound by ``Assign`` (any target shape:
  names, tuples, lists, starred, nested), ``AnnAssign`` with a value,
  ``AugAssign``, ``for`` targets and ``with ... as`` targets, both at the top
  level and inside top-level ``if``/``try``/``with``/``for``/``while``/
  ``match`` blocks;
* class-level constant attributes of the module's classes;
* the signature defaults of functions, methods and classes.

A second, run-time check closes the grammar from the other side: every
constant-named attribute of the imported module must be one of those bindings,
a ``def``/``class`` statement, or an imported name. An attribute that is none of
them (made by ``globals()``, ``setattr`` or a binding form the walk does not
read) fails, so a form the walk misses cannot pass silently.

It does not see literals written inside a function body, enum members, or
dataclass ``default_factory`` values. A personal number spelled inline in a
function is outside what it can find, so the review of a new function still has
to look for one.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import importlib.util
import inspect
import pkgutil
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

import landeval

INVENTORY = Path(__file__).resolve().parents[1] / "publish" / "constants.toml"
CONSTANT_NAME = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
GENERAL = "general"


def _trivial(value: object) -> bool:
    if value is None or value is True or value is False:
        return True
    return isinstance(value, (tuple, list, dict, set, frozenset)) and len(value) == 0


_ABSENT = object()

# Compound statements whose bodies run at module level when the module does.
_BLOCKS = (ast.If, ast.Try, ast.TryStar, ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While)

# Statements that bind through a single ``target``.
_SINGLE_TARGET = (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)


def _target_names(target: ast.expr) -> list[str]:
    """Every name an assignment target binds, whatever its shape."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_names(element)]
    return []  # attribute and subscript targets bind no module name


def _module_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """The statements of ``body`` and of every block nested in it at module level."""
    statements: list[ast.stmt] = []
    for node in body:
        statements.append(node)
        if isinstance(node, _BLOCKS):
            nested = list(node.body) + list(getattr(node, "orelse", []))
            nested += list(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                nested += handler.body
            statements += _module_statements(nested)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                statements += _module_statements(case.body)
    return statements


@dataclass(frozen=True)
class Bindings:
    """What a module's source binds at module level, by kind."""

    assigned: frozenset[str]
    defined: frozenset[str]
    imported: frozenset[str]


def module_bindings(tree: ast.Module) -> Bindings:
    assigned: set[str] = set()
    defined: set[str] = set()
    imported: set[str] = set()
    for node in _module_statements(tree.body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assigned.update(_target_names(target))
        elif isinstance(node, _SINGLE_TARGET) and getattr(node, "value", True) is not None:
            # ``X: int`` with no value binds nothing; every other form here does.
            assigned.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    assigned.update(_target_names(item.optional_vars))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names):
                raise AssertionError(f"star import in a public module: {ast.dump(node)}")
            imported.update(a.asname or a.name for a in node.names)
    return Bindings(frozenset(assigned), frozenset(defined), frozenset(imported))


def unexplained_attributes(module: ModuleType) -> set[str]:
    """Constant-named attributes no binding the scanner reads accounts for."""
    bindings = module_bindings(ast.parse(inspect.getsource(module)))
    known = bindings.assigned | bindings.defined | bindings.imported
    return {name for name in vars(module) if CONSTANT_NAME.match(name) and name not in known}


def inspect_module(module: ModuleType) -> set[str]:
    """Keys ``module:NAME`` and ``module:Qualified.name(param)`` found at runtime.

    A constant-named attribute that no binding explains is reported too, so it
    shows up as missing from the inventory as well as failing the run-time
    check.
    """
    found: set[str] = set()
    bindings = module_bindings(ast.parse(inspect.getsource(module)))
    for name in bindings.assigned:
        if not CONSTANT_NAME.match(name):
            continue
        # A name bound in a branch not taken is absent at run time; it is still
        # reported rather than read as trivial.
        value = getattr(module, name, _ABSENT)
        if value is _ABSENT or not _trivial(value):
            found.add(f"{module.__name__}:{name}")
    found |= {f"{module.__name__}:{name}" for name in unexplained_attributes(module)}

    def add_defaults(owner: object, qualname: str) -> None:
        try:
            signature = inspect.signature(owner)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        for param in signature.parameters.values():
            default = param.default
            if default is inspect.Parameter.empty or default is dataclasses.MISSING:
                continue
            if isinstance(default, dataclasses.Field) or "<factory>" in repr(default):
                continue
            if not _trivial(default):
                found.add(f"{module.__name__}:{qualname}({param.name})")

    for name, obj in vars(module).items():
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            add_defaults(obj, name)
        elif inspect.isclass(obj) and not issubclass(obj, enum.Enum):
            add_defaults(obj, name)
            for attr, value in vars(obj).items():
                if attr == "__init__":
                    continue  # already seen through the class signature
                if inspect.isfunction(value):
                    add_defaults(value, f"{name}.{attr}")
                elif isinstance(value, (staticmethod, classmethod)):
                    add_defaults(value.__func__, f"{name}.{attr}")
                elif CONSTANT_NAME.match(attr) and not _trivial(value):
                    found.add(f"{module.__name__}:{name}.{attr}")
    return found


def public_modules() -> list[ModuleType]:
    names = ["landeval"] + [m.name for m in pkgutil.iter_modules(landeval.__path__, "landeval.")]
    return [importlib.import_module(name) for name in names]


def found_keys() -> set[str]:
    keys: set[str] = set()
    for module in public_modules():
        keys |= inspect_module(module)
    return keys


def load_inventory() -> dict[str, dict[str, str]]:
    with INVENTORY.open("rb") as handle:
        return tomllib.load(handle)


def not_general(inventory: dict[str, dict[str, str]]) -> list[str]:
    return sorted(key for key, entry in inventory.items() if entry.get("class") != GENERAL)


def test_the_inventory_names_exactly_what_the_modules_hold() -> None:
    """Both directions: a new constant must be listed, a removed one must go."""
    listed = set(load_inventory())
    found = found_keys()

    assert sorted(found - listed) == [], "constants or defaults missing from the inventory"
    assert sorted(listed - found) == [], "inventory entries that no module holds any more"


def test_no_entry_is_personal() -> None:
    """A personal value must be gone from the public side, not merely listed."""
    inventory = load_inventory()

    assert not_general(inventory) == [], "personal or unclassed values remain in the package"


def test_every_entry_says_why() -> None:
    inventory = load_inventory()

    assert sorted(k for k, e in inventory.items() if not str(e.get("reason", "")).strip()) == []
    assert sorted(k for k, e in inventory.items() if set(e) != {"class", "reason"}) == []


def test_the_policy_values_are_not_defined_by_the_package() -> None:
    """These are policy values a caller passes; the package must not define them."""
    gone = {
        "landeval.stability:BAND_S",
        "landeval.stability:NOISE_TOLERANCE_S",
        "landeval.stability:SYSTEMATIC_TOLERANCE_S",
        "landeval.calibration:DEFAULT_GOOD_MAX_MIN",
        "landeval.calibration:DEFAULT_EXCLUDE_MIN",
        "landeval.viz:SCREENING_BANDS",
        "landeval.viz:DEFAULT_BANDS",
        "landeval.sampling:DEFAULT_TIME_BANDS",
        "landeval.calibration:EXCLUDE_SIGMA_MARGIN",
        "landeval.calibration:GOOD_SIGMA_MARGIN",
        "landeval.measure:DEFAULT_MAX_UNROUTED",
        "landeval.measure:measure_points(max_unrouted)",
    }

    assert gone & found_keys() == set()
    assert gone & set(load_inventory()) == set()


def load_synthetic(source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("synthetic_module", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_scanner_finds_what_it_claims_to_find(tmp_path: Path) -> None:
    """A scanner that finds nothing would make the equality above vacuous."""
    source = tmp_path / "synthetic_module.py"
    source.write_text(
        "from dataclasses import dataclass, field\n"
        "LIMIT = 7\n"
        "FLAG = True\n"
        "EMPTY = ()\n"
        "lower_case = 3\n"
        "def f(a, b=2, c=None, d=(), e=False):\n"
        "    return a\n"
        "class K:\n"
        "    CEILING = 9\n"
        "    def m(self, x=1.5):\n"
        "        return x\n"
        "@dataclass\n"
        "class D:\n"
        "    n: int = 4\n"
        "    xs: list[int] = field(default_factory=list)\n",
        encoding="utf-8",
    )
    module = load_synthetic(source)

    assert inspect_module(module) == {
        "synthetic_module:LIMIT",
        "synthetic_module:f(b)",
        "synthetic_module:K.CEILING",
        "synthetic_module:K.m(x)",
        "synthetic_module:D(n)",
    }


# Every binding form a naive scanner would miss, plus the forms the closed walk
# reads, and one attribute no source binding makes.
EVERY_BINDING_FORM = (
    "import contextlib\n"
    "from re import IGNORECASE\n"
    "_HIDDEN = 5\n"
    "LEFT, RIGHT = 1, 2\n"
    "FIRST, *REST = (3, 4, 5)\n"
    "[NESTED_A, (NESTED_B, NESTED_C)] = [6, (7, 8)]\n"
    "ANNOTATED: int = 9\n"
    "BARE: int\n"
    "COUNTER = 0\n"
    "COUNTER += 1\n"
    "if IGNORECASE:\n"
    "    IN_IF = 10\n"
    "else:\n"
    "    NOT_TAKEN = 11\n"
    "try:\n"
    "    IN_TRY = 12\n"
    "except ImportError:\n"
    "    pass\n"
    "with contextlib.nullcontext(13) as IN_WITH:\n"
    "    pass\n"
    'globals()["RUNTIME_ONLY"] = 14\n'
)

EVERY_BINDING_FORM_KEYS = {
    f"synthetic_module:{name}"
    for name in (
        "_HIDDEN",
        "LEFT",
        "RIGHT",
        "FIRST",
        "REST",
        "NESTED_A",
        "NESTED_B",
        "NESTED_C",
        "ANNOTATED",
        "COUNTER",
        "IN_IF",
        "NOT_TAKEN",
        "IN_TRY",
        "IN_WITH",
        "RUNTIME_ONLY",
    )
}


def test_every_binding_form_is_reported_as_uninventoried(tmp_path: Path) -> None:
    """``_`` names and unpacked targets must not pass unseen.

    An imported name belongs to the module that defines it, and a bare
    annotation binds nothing, so neither is reported. A name bound only in a
    branch not taken is still reported rather than read as trivial.
    """
    source = tmp_path / "synthetic_module.py"
    source.write_text(EVERY_BINDING_FORM, encoding="utf-8")
    module = load_synthetic(source)

    found = inspect_module(module)
    uninventoried = found - set(load_inventory())

    assert found == EVERY_BINDING_FORM_KEYS
    assert uninventoried >= EVERY_BINDING_FORM_KEYS


def test_the_runtime_check_names_an_attribute_no_binding_explains(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_module.py"
    source.write_text(EVERY_BINDING_FORM, encoding="utf-8")
    module = load_synthetic(source)

    assert unexplained_attributes(module) == {"RUNTIME_ONLY"}


def test_every_runtime_constant_of_the_package_has_a_binding_the_scanner_reads() -> None:
    """The run-time side of the closed grammar: inventorying a name is not enough."""
    unexplained = {
        module.__name__: sorted(names)
        for module in public_modules()
        if (names := unexplained_attributes(module))
    }

    assert unexplained == {}


def test_a_star_import_is_refused() -> None:
    """``from x import *`` binds names the walk cannot list, so it fails loudly."""
    with pytest.raises(AssertionError, match="star import"):
        module_bindings(ast.parse("from os.path import *\n"))


def test_the_scanner_sees_every_public_module() -> None:
    names = {m.__name__ for m in public_modules()}

    assert len(names) >= 28
    assert {"landeval.stability", "landeval.calibration", "landeval.viz"} <= names
    assert "landeval.stability:QUANTILES" in found_keys()


def test_the_class_check_refuses_a_personal_or_unclassed_entry() -> None:
    """The check above would pass on any inventory; prove it can fail."""
    inventory = {
        "m:A": {"class": "general", "reason": "x"},
        "m:B": {"class": "personal", "reason": "x"},
        "m:C": {"class": "gneral", "reason": "x"},
        "m:D": {"reason": "x"},
    }

    assert not_general(inventory) == ["m:B", "m:C", "m:D"]
