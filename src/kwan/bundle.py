"""The bundle: one JSON file per subject with everything Kwan needs to run (PRD 6.1, decision D2).

    {"schema_version": 2, "bundle_version": "math-1.0.0rc1", "subject": "math", "text_mode": "labels",
     "created_at": "...", "source_commit": "...",
     "graph": {"nodes": [...], "edges": [...]},
     "params": {...},                      EloParams.to_dict()
     "item_bank": {...} or null,            ItemBank.to_dict()
     "contexts": {id: [{"text", "lang", "reviewed"}]}, "context_bank": {"1:lower": [...]},
     "i18n": {"en": {key: template}},
     "provenance": {"params": {"datasets": [...]}, "item_bank": {...}, "graph": {...}, "contexts": {...}}}

`load_bundle` reads and checks a bundle (by default the one shipped inside the package).
`build_bundle` makes one from the committed data files. It refuses parameters or item banks whose
provenance names a dataset that is evaluation only (PRD 5.3: AMMORE, Junyi), and it refuses the
`full` text mode unless the full-text graph exists under `build/` and `data/curriculum/permissions.json`
records NaCCA's written permission.

Loading needs only the standard library and networkx. Building reads the repository's `data/`.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx

from kwan import ids
from kwan._paths import data_dir, repo_root
from kwan._version import __version__
from kwan.graph import schema
from kwan.learner.item_bank import ItemBank
from kwan.learner.params import EloParams

SCHEMA_VERSION = 2
TEXT_MODES = ("labels", "full")
MAX_BYTES = 1_500_000  # PRD 7: the maths bundle stays under 1.5 MB uncompressed
REQUIRED_KEYS = (
    "schema_version",
    "bundle_version",
    "subject",
    "text_mode",
    "created_at",
    "source_commit",
    "graph",
    "params",
    "contexts",
    "context_bank",
    "i18n",
    "provenance",
)

__all__ = [
    "MAX_BYTES",
    "SCHEMA_VERSION",
    "TEXT_MODES",
    "Bundle",
    "BundleError",
    "LicenceError",
    "build_bundle",
    "bundle_file_name",
    "bundle_version_for",
    "check_provenance",
    "default_bundle_path",
    "dumps_bundle",
    "load_bundle",
    "load_params_file",
    "packaged_bundle_path",
    "source_commit",
    "validate_bundle",
    "write_bundle",
]


class BundleError(ValueError):
    """The file is not a bundle this Kwan can use, or the bundle cannot be built as asked."""


class LicenceError(BundleError):
    """Parameters or an item bank were fitted on a dataset whose licence forbids shipping them."""


# ----------------------------------------------------------------------------- names and paths


def bundle_file_name(subject: str = ids.DEFAULT_SUBJECT) -> str:
    """`kwan-bundle.<subject>.json`."""
    return f"kwan-bundle.{subject}.json"


def bundle_version_for(subject: str = ids.DEFAULT_SUBJECT, version: str = __version__) -> str:
    """The bundle version of a subject at a package version: `math-1.0.0rc1`."""
    return f"{subject}-{version}"


def packaged_bundle_path(subject: str = ids.DEFAULT_SUBJECT) -> Path:
    """Where the bundle shipped inside the package lives (`kwan/bundles/kwan-bundle.<subject>.json`)."""
    return Path(__file__).resolve().parent / "bundles" / bundle_file_name(subject)


def default_bundle_path(subject: str = ids.DEFAULT_SUBJECT) -> Path:
    """The bundle `load_bundle()` opens with no path: the one shipped inside the package.

    `KWAN_BUNDLE` in the environment names another file instead."""
    env = os.environ.get("KWAN_BUNDLE")
    if env:
        return Path(env).expanduser().resolve()
    return packaged_bundle_path(subject)


# ----------------------------------------------------------------------------- the loaded bundle


@dataclass(frozen=True)
class Bundle:
    """A checked bundle. The fields are the file's top-level keys; `path` is where it was read from."""

    schema_version: int
    bundle_version: str
    subject: str
    text_mode: str
    created_at: str
    source_commit: str
    graph: dict[str, Any]
    params: dict[str, Any]
    contexts: dict[str, list[dict[str, Any]]]
    context_bank: dict[str, list[str]]
    i18n: dict[str, dict[str, str]]
    provenance: dict[str, Any]
    item_bank: dict[str, Any] | None = None
    path: Path | None = field(default=None, compare=False)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], path: Path | None = None) -> Bundle:
        """Check a parsed bundle and wrap it. Raises `BundleError` when something is wrong."""
        validate_bundle(d)
        return cls(
            schema_version=int(d["schema_version"]),
            bundle_version=str(d["bundle_version"]),
            subject=str(d["subject"]),
            text_mode=str(d["text_mode"]),
            created_at=str(d["created_at"]),
            source_commit=str(d["source_commit"]),
            graph={"nodes": list(d["graph"]["nodes"]), "edges": list(d["graph"]["edges"])},
            params=dict(d["params"]),
            contexts={str(k): [dict(e) for e in v] for k, v in d["contexts"].items()},
            context_bank={str(k): [str(t) for t in v] for k, v in d["context_bank"].items()},
            i18n={str(lang): {str(k): str(t) for k, t in table.items()} for lang, table in d["i18n"].items()},
            provenance=dict(d["provenance"]),
            item_bank=dict(d["item_bank"]) if d.get("item_bank") else None,
            path=path,
        )

    def to_dict(self) -> dict[str, Any]:
        """The bundle as it is written to disk (keys in file order)."""
        return {
            "schema_version": self.schema_version,
            "bundle_version": self.bundle_version,
            "subject": self.subject,
            "text_mode": self.text_mode,
            "created_at": self.created_at,
            "source_commit": self.source_commit,
            "graph": self.graph,
            "params": self.params,
            "item_bank": self.item_bank,
            "contexts": self.contexts,
            "context_bank": self.context_bank,
            "i18n": self.i18n,
            "provenance": self.provenance,
        }

    def load_graph(self) -> nx.DiGraph:
        """The curriculum graph as a networkx graph (node keys are canonical ids)."""
        d = {
            "schema_version": schema.SCHEMA_VERSION,
            "subject": self.subject,
            "text_mode": self.text_mode,
            **self.graph,
        }
        return schema.graph_from_dict(d, strict=True)

    def elo_params(self) -> EloParams:
        """The learner-model parameters."""
        return EloParams.from_dict(self.params)

    def load_item_bank(self) -> ItemBank | None:
        """The item-difficulty bank, or None when the bundle has none."""
        return ItemBank.from_dict(self.item_bank) if self.item_bank else None

    @property
    def node_count(self) -> int:
        """Number of indicators in the bundle's graph."""
        return len(self.graph["nodes"])

    @property
    def edge_count(self) -> int:
        """Number of prerequisite links in the bundle's graph."""
        return len(self.graph["edges"])


def validate_bundle(d: Mapping[str, Any]) -> None:
    """Raise `BundleError` unless `d` has the shape of a schema 2 bundle."""
    if not isinstance(d, Mapping):
        raise BundleError("a bundle must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise BundleError(f"bundle is missing {', '.join(missing)}")
    if d["schema_version"] != SCHEMA_VERSION:
        raise BundleError(f"bundle schema_version {d['schema_version']!r} is not supported (expected {SCHEMA_VERSION})")
    if d["text_mode"] not in TEXT_MODES:
        raise BundleError(f"unknown text_mode {d['text_mode']!r} (expected one of {', '.join(TEXT_MODES)})")
    for key in ("bundle_version", "subject", "created_at", "source_commit"):
        if not isinstance(d[key], str) or not d[key]:
            raise BundleError(f"bundle {key} must be non-empty text")
    graph = d["graph"]
    if (
        not isinstance(graph, Mapping)
        or not isinstance(graph.get("nodes"), list)
        or not isinstance(graph.get("edges"), list)
    ):
        raise BundleError("bundle graph must hold a 'nodes' list and an 'edges' list")
    if not graph["nodes"]:
        raise BundleError("bundle graph has no nodes")
    for key in ("params", "contexts", "context_bank", "i18n", "provenance"):
        if not isinstance(d[key], Mapping):
            raise BundleError(f"bundle {key} must be an object")
    if d.get("item_bank") is not None and not isinstance(d["item_bank"], Mapping):
        raise BundleError("bundle item_bank must be an object or null")
    for lang, table in d["i18n"].items():
        if not isinstance(table, Mapping):
            raise BundleError(f"bundle i18n[{lang!r}] must be an object of templates")
    for indicator, entries in d["contexts"].items():
        if not isinstance(entries, list) or not all(isinstance(e, Mapping) and e.get("text") for e in entries):
            raise BundleError(f"bundle contexts[{indicator!r}] must be a list of settings with text")
    try:
        EloParams.from_dict(d["params"])
    except (TypeError, ValueError) as err:
        raise BundleError(f"bundle params: {err}") from err


def load_bundle(path: str | os.PathLike[str] | None = None) -> Bundle:
    """Read and check a bundle file. With no path, the bundle shipped inside the package.

    Raises `FileNotFoundError` when the file is not there and `BundleError` when it is not a bundle."""
    file = Path(path) if path is not None else default_bundle_path()
    if not file.is_file():
        hint = " (build it with `kwan bundle`)" if path is None else ""
        raise FileNotFoundError(f"no bundle at {file}{hint}")
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
    except ValueError as err:
        raise BundleError(f"{file}: not valid JSON ({err})") from err
    return Bundle.from_dict(data, path=file)


# ----------------------------------------------------------------------------- licence checks


def _datasets_of(provenance: Mapping[str, Any] | None) -> list[str]:
    datasets = (provenance or {}).get("datasets") or []
    if isinstance(datasets, str):
        datasets = [datasets]
    return [str(x) for x in datasets]


def check_provenance(params_provenance: Mapping[str, Any] | None, item_bank: ItemBank | None = None) -> None:
    """Raise `LicenceError` when the parameters or the item bank were fitted on evaluation-only data.

    A dataset the registry does not know also raises, so a typo can never pass as permission."""
    from kwan.data.registry import commercial_ok

    for what, datasets in (
        ("params", _datasets_of(params_provenance)),
        ("item_bank", _datasets_of(item_bank.provenance) if item_bank else []),
    ):
        for name in datasets:
            try:
                ok = commercial_ok(name)
            except KeyError as err:
                raise LicenceError(f"{what}: {err.args[0]}; add it to data/datasets.yaml first") from None
            if not ok:
                raise LicenceError(
                    f"{what} were fitted on {name}, whose licence allows evaluation only; "
                    "a shipped bundle may hold only parameters fitted on commercial-OK data (PRD 5.3)"
                )


# ----------------------------------------------------------------------------- building


def source_commit(root: Path | None = None) -> str:
    """The git commit of the checkout (`git rev-parse HEAD`), or `"unknown"` without git or a repository."""
    root = root if root is not None else repo_root()
    if root is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = out.stdout.strip()
    return text if out.returncode == 0 and text else "unknown"


def load_params_file(path: str | os.PathLike[str]) -> tuple[EloParams, dict[str, Any]]:
    """Read a fitted-parameters file and return `(params, provenance)`.

    Two layouts are read: `{"params": {...}, "provenance": {...}}`, or the flat `EloParams.to_dict()`
    keys with an optional `provenance` (or `datasets`) key beside them."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, Mapping):
        raise BundleError(f"{path}: a parameters file must hold a JSON object")
    inner = d.get("params") if isinstance(d.get("params"), Mapping) else d
    prov: dict[str, Any] = dict(d.get("provenance") or {})
    if "datasets" not in prov and "datasets" in d:
        prov["datasets"] = d["datasets"]
    prov["datasets"] = _datasets_of(prov)
    prov.setdefault("source", str(path))
    try:
        return EloParams.from_dict(inner), prov
    except (TypeError, ValueError) as err:
        raise BundleError(f"{path}: {err}") from err


def _default_params(subject: str) -> tuple[EloParams, dict[str, Any]]:
    """The fitted parameters file `data/params/<subject>.params.json` when it exists, else the defaults."""
    path = data_dir() / "params" / f"{subject}.params.json"
    if path.is_file():
        return load_params_file(path)
    return EloParams(), {"datasets": [], "source": "defaults", "note": "EloParams defaults (v1 values)"}


def _permission_granted(permissions_path: Path | None = None) -> tuple[bool, Path]:
    path = permissions_path if permissions_path is not None else data_dir() / "curriculum" / "permissions.json"
    if not path.is_file():
        return False, path
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return bool(isinstance(doc, Mapping) and doc.get("nacca_text_commit_allowed") is True), path


def _full_graph_path(subject: str) -> Path:
    root = repo_root()
    base = root if root is not None else Path.cwd()
    return base / "build" / "graph" / f"{subject}.graph.full.json"


def _load_graph_for(subject: str, text_mode: str, graph: nx.DiGraph | None) -> tuple[nx.DiGraph, str]:
    """The graph to bundle and where it came from, checking the text-mode rules."""
    if text_mode not in TEXT_MODES:
        raise BundleError(f"unknown text mode {text_mode!r} (expected one of {', '.join(TEXT_MODES)})")
    if text_mode == "full":
        allowed, perm = _permission_granted()
        full = _full_graph_path(subject)
        if not full.is_file():
            raise BundleError(
                f"cannot build a full-text bundle: {full} is not there "
                "(run `kwan curriculum fetch`, `kwan curriculum parse` and `kwan graph build` first)"
            )
        if not allowed:
            raise BundleError(
                f"cannot build a full-text bundle: {perm} does not record NaCCA's written permission "
                "(nacca_text_commit_allowed is not true). Build with --text labels instead."
            )
        if graph is None:
            graph = schema.load_graph(full)
        if graph.graph.get("text_mode") != "full":
            raise BundleError("the graph given is not a full-text graph")
        return graph, str(full)
    if graph is None:
        path = schema.default_graph_path(subject)
        if not path.is_file():
            raise BundleError(f"no committed graph at {path}; build it with `kwan graph build`")
        return schema.load_graph(path), str(path)
    source = "given"
    if graph.graph.get("text_mode") == "full":
        graph = schema.strip_text(graph)
        source = "given (curriculum text stripped)"
    return graph, source


def _contexts_blocks(
    subject: str, contexts: Any, include_drafts: bool
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], dict[str, int]]:
    """`(contexts, context_bank, counts)` from a ContextStore, a bundle-shaped mapping, or the data files."""
    from kwan.context.settings import ContextStore

    if contexts is None:
        store = (
            ContextStore.from_file(subject=subject)
            if (data_dir() / "contexts" / f"{subject}.json").is_file()
            else ContextStore(None, None, subject)
        )
    elif isinstance(contexts, ContextStore):
        store = contexts
    elif isinstance(contexts, Mapping):
        store = ContextStore.from_bundle({"subject": subject, **contexts})
    else:
        raise TypeError("contexts must be a ContextStore, a mapping with 'contexts' and 'context_bank', or None")
    block = store.bundle_contexts(include_drafts=include_drafts)
    reviewed = sum(1 for entries in block.values() for e in entries if e.get("reviewed"))
    drafts = sum(1 for entries in block.values() for e in entries if not e.get("reviewed"))
    return block, store.bundle_bank(), {"indicators": len(block), "reviewed": reviewed, "drafts": drafts}


def _i18n_block(i18n: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """`{lang: {key: template}}`: the given tables, else every `data/i18n/<lang>.json`, else built-in English."""
    from kwan.recommend import explain

    if i18n is not None:
        return explain.resolve_templates(i18n)
    folder = data_dir() / "i18n"
    tables: dict[str, dict[str, str]] = {}
    if folder.is_dir():
        for file in sorted(folder.glob("*.json")):
            tables[file.stem] = explain.load_templates(file)
    return explain.resolve_templates(tables or None)


def build_bundle(
    subject: str = ids.DEFAULT_SUBJECT,
    text_mode: str = "labels",
    params: EloParams | Mapping[str, Any] | None = None,
    item_bank: ItemBank | Mapping[str, Any] | None = None,
    contexts: Any = None,
    out: str | os.PathLike[str] | None = None,
    *,
    params_provenance: Mapping[str, Any] | None = None,
    graph: nx.DiGraph | None = None,
    i18n: Mapping[str, Any] | None = None,
    include_drafts: bool = True,
    bundle_version: str | None = None,
    commit: str | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the bundle of a subject and return it as a dictionary; with `out`, also write it there.

    subject: the subject whose committed files are read (`data/graph/<subject>.graph.json`,
        `data/contexts/<subject>.json`, `data/i18n/*.json`, `data/params/<subject>.params.json`).
    text_mode: `"labels"` (Kwan's short labels, what the project ships) or `"full"` (the NaCCA text
        from `build/graph/<subject>.graph.full.json`, allowed only with recorded written permission).
    params: the learner-model parameters (an `EloParams` or its dictionary). None reads the fitted
        parameters file when there is one, else the defaults.
    item_bank: an optional item-difficulty bank (an `ItemBank` or its dictionary).
    contexts: a `ContextStore`, a mapping with `contexts` and `context_bank`, or None for the data files.
    out: a directory (the file is named `kwan-bundle.<subject>.json`) or a file path.
    params_provenance: `{"datasets": [...], ...}` for `params`; None takes it from the file or marks defaults.
    graph: a graph to bundle instead of the committed one (tests).
    i18n: templates per language instead of the `data/i18n` files.
    include_drafts: keep unreviewed practice settings in the bundle (marked `reviewed: false`; the
        engine uses them only when a caller asks with `include_drafts=True`).
    bundle_version, commit, created_at: override the version, the git commit and the build time.

    Raises `LicenceError` when the parameters or the item bank were fitted on evaluation-only data, and
    `BundleError` for a text mode that cannot be built or a result larger than `MAX_BYTES`.
    """
    if params is None:
        elo, prov = _default_params(subject)
    else:
        elo = params if isinstance(params, EloParams) else EloParams.from_dict(params)
        prov = {"datasets": [], "source": "given"}
    if params_provenance is not None:
        prov = {**prov, **dict(params_provenance)}
    prov["datasets"] = _datasets_of(prov)
    bank = item_bank if isinstance(item_bank, ItemBank) or item_bank is None else ItemBank.from_dict(item_bank)
    check_provenance(prov, bank)

    G, graph_source = _load_graph_for(subject, text_mode, graph)
    dumped = schema.dump_graph(G)
    if dumped["text_mode"] != text_mode:
        raise BundleError(f"the graph is in {dumped['text_mode']!r} mode, not {text_mode!r}")
    ctx, bank_block, ctx_counts = _contexts_blocks(subject, contexts, include_drafts)
    when = created_at if created_at is not None else datetime.now(UTC)
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_version": bundle_version or bundle_version_for(subject),
        "subject": subject,
        "text_mode": text_mode,
        "created_at": when.isoformat(timespec="seconds"),
        "source_commit": commit if commit is not None else source_commit(),
        "graph": {"nodes": dumped["nodes"], "edges": dumped["edges"]},
        "params": elo.to_dict(),
        "item_bank": bank.to_dict() if bank is not None else None,
        "contexts": ctx,
        "context_bank": bank_block,
        "i18n": _i18n_block(i18n),
        "provenance": {
            "kwan_version": __version__,
            "params": prov,
            "item_bank": dict(bank.provenance) if bank is not None else None,
            "graph": {"source": graph_source, "nodes": len(dumped["nodes"]), "edges": len(dumped["edges"])},
            "contexts": ctx_counts,
        },
    }
    validate_bundle(doc)
    size = len(dumps_bundle(doc).encode("utf-8"))
    if size > MAX_BYTES:
        raise BundleError(f"the bundle is {size} bytes, over the {MAX_BYTES} byte limit (PRD 7)")
    if out is not None:
        write_bundle(doc, out)
    return doc


def dumps_bundle(doc: Mapping[str, Any]) -> str:
    """The file text of a bundle: compact JSON, one top-level key per line, UTF-8 text kept as it is."""
    lines = [
        f'"{k}": {json.dumps(v, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}' for k, v in doc.items()
    ]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def write_bundle(doc: Mapping[str, Any], out: str | os.PathLike[str]) -> Path:
    """Write a bundle to `out` (a directory, or a file path) in one step and return the file path."""
    target = Path(out)
    if target.is_dir() or not target.suffix:
        target = target / bundle_file_name(str(doc.get("subject", ids.DEFAULT_SUBJECT)))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(dumps_bundle(doc))
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)
    return target
