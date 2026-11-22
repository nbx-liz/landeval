"""The data root of a script run: the caller decides it, the library never does.

A script gets its root from ``--root`` or from ``LANDEVAL_ROOT``, in that order.
An empty value counts as unset. With neither, the script stops: it does not look
in the current directory or beside the package, because that reads a decoy
``data/`` tree without a word when a run is started from the wrong place.

A data path default is written ``under_root("data/processed/x.parquet")``. It is
joined to the root after parsing, by ``resolve_root``. A path the user gives on
the command line is left alone and keeps its usual meaning, relative to the
current directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from landeval.config import Region

ROOT_ENV = "LANDEVAL_ROOT"

# Tests assert on these two strings, so they are constants and not literals at
# the raise sites.
MISSING_ROOT_MESSAGE = (
    "error: no data root. Pass --root <dir> or set LANDEVAL_ROOT; "
    "the current directory is never used as a default."
)
NOT_A_DIRECTORY_MESSAGE = "error: the data root is not an existing directory"


class RootRelative:
    """A default that is a path under the data root, joined once the root is known."""

    def __init__(self, relative: str) -> None:
        self.relative = relative

    def __str__(self) -> str:
        return f"<root>/{self.relative}"

    __repr__ = __str__

    def under(self, root: Path) -> Path:
        return root / self.relative


def under_root(relative: str) -> RootRelative:
    """Mark ``relative`` as a path below the data root, for use as an argparse default."""
    return RootRelative(relative)


def add_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=f"data root (the repository checkout); falls back to ${ROOT_ENV}, no other default",
    )


def resolve_root(args: argparse.Namespace) -> Path:
    """Return the root, and join every ``under_root`` default on ``args`` to it.

    Raises ``SystemExit`` with ``MISSING_ROOT_MESSAGE`` when neither ``--root`` nor
    ``LANDEVAL_ROOT`` is set, and with ``NOT_A_DIRECTORY_MESSAGE`` when the root
    given is not an existing directory.
    """
    given: str | None = getattr(args, "root", None)
    if given is not None and given.strip() != "":
        root = Path(given)
    else:
        from_env = os.environ.get(ROOT_ENV, "").strip()
        if not from_env:
            raise SystemExit(MISSING_ROOT_MESSAGE)
        root = Path(from_env)
    if not root.is_dir():
        raise SystemExit(f"{NOT_A_DIRECTORY_MESSAGE}: {root}")
    root = root.resolve()
    args.root = root
    for name, value in vars(args).items():
        if isinstance(value, RootRelative):
            setattr(args, name, value.under(root))
        elif isinstance(value, list) and any(isinstance(v, RootRelative) for v in value):
            setattr(
                args, name, [v.under(root) if isinstance(v, RootRelative) else v for v in value]
            )
    return root


DEFAULT_CONFIG = "config/destinations.yaml"


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=under_root(DEFAULT_CONFIG),
        help="configuration file whose `region` section selects the area",
    )


def load_region(config_path: Path) -> Region:
    """Return the configuration's ``region``, or stop: no script assumes an area."""
    from landeval.config import ConfigError, load_config

    try:
        region = load_config(config_path).region
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if region is None:
        raise SystemExit(f"error: {config_path} has no `region` section; this script needs one.")
    return region
