"""The `kwan` command.

Subcommands come from a registry of modules. Each module in `REGISTRY` exposes
`register(subparsers)`, which adds its own top-level commands and sets two defaults on each parser:

    func   the handler, called with the parsed arguments; it returns an exit code or None
    extra  (optional) the extra (pip/uv) that brings the packages the command needs, e.g. "eval"

A registry module that does not exist yet, or that cannot be imported because an optional package is
missing, is skipped without a word. The hint is printed only when one of its commands is asked for.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from kwan._version import __version__

REGISTRY = (
    "kwan.curriculum._cli",
    "kwan.graph._cli",
    "kwan.eval._cli",
    "kwan.eval.fit",
    "kwan.context._cli",
    "kwan._bundle_cli",
)

# command -> (registry module that provides it, extra it needs or None); used only for hints
COMMANDS = {
    "curriculum": ("kwan.curriculum._cli", "curriculum"),
    "graph": ("kwan.graph._cli", "graph-ai"),
    "eval": ("kwan.eval._cli", "eval"),
    "sim": ("kwan.eval._cli", None),
    "bench": ("kwan.eval._cli", None),
    "fit": ("kwan.eval.fit", "eval"),
    "context": ("kwan.context._cli", None),
    "bundle": ("kwan._bundle_cli", None),
}


def _install_hint(extra: str | None) -> str:
    return (
        f"Install what it needs with: uv add 'kwan[{extra}]' (in a source checkout: uv sync --extra {extra})"
        if extra
        else "Check that kwan is fully installed."
    )


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, Exception]]:
    """The parser with every available subcommand, and {registry module: why it was skipped}."""
    parser = argparse.ArgumentParser(
        prog="kwan",
        description=(
            "Kwan: an adaptive learning engine for Ghanaian pupils, on the NaCCA curriculum. "
            "This release covers Mathematics B1 to B6."
        ),
    )
    parser.add_argument("--version", action="version", version=f"kwan {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    skipped: dict[str, Exception] = {}
    for name in REGISTRY:
        try:
            importlib.import_module(name).register(subparsers)
        except ImportError as exc:  # the module is not there yet, or one of its optional packages is missing
            skipped[name] = exc
    return parser, skipped


def _first_word(argv: Sequence[str]) -> str | None:
    return next((a for a in argv if not a.startswith("-")), None)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser, skipped = build_parser()
    word = _first_word(argv)
    if word in COMMANDS and COMMANDS[word][0] in skipped:
        module, extra = COMMANDS[word]
        exc = skipped[module]
        missing = getattr(exc, "name", None)
        if missing == module:
            print(f"kwan {word}: this command is not part of this install yet.", file=sys.stderr)
        else:
            print(f"kwan {word}: cannot run ({exc}). {_install_hint(extra)}", file=sys.stderr)
        return 2
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    try:
        code = func(args)
    except ModuleNotFoundError as exc:
        hint = _install_hint(getattr(args, "extra", None))
        print(f"kwan {args.command}: the package {exc.name!r} is not installed. {hint}", file=sys.stderr)
        return 1
    return code if isinstance(code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
