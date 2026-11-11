"""Namespaced indicator ids.

The canonical id of an indicator is `"<subject>:<code>"`, for example `math:B4.1.1.1.2`. The bare
code (`B4.1.1.1.2`) is accepted wherever an id comes in. `Resolver` turns anything a caller may send
(a canonical id, a bare code, or an alias such as a code misprinted in the curriculum PDF) into the
key the graph uses for that node.

Only the standard library is used here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

SEP = ":"
DEFAULT_SUBJECT = "math"

__all__ = ["DEFAULT_SUBJECT", "SEP", "Resolver", "code_of", "is_namespaced", "make", "split", "subject_of"]


def is_namespaced(id: str) -> bool:
    """True for `math:B4.1.1.1.2`, False for the bare code `B4.1.1.1.2`."""
    if not isinstance(id, str):
        return False
    subject, sep, code = id.partition(SEP)
    return bool(sep) and bool(subject) and bool(code)


def make(subject: str, code: str) -> str:
    """The canonical id for a subject and a code: `make("math", "B4.1.1.1.2") -> "math:B4.1.1.1.2"`.

    A code that already carries the same subject is returned unchanged, so `make` is safe to call twice.
    """
    if not isinstance(subject, str) or not subject or SEP in subject:
        raise ValueError(f"invalid subject {subject!r}")
    if not isinstance(code, str) or not code:
        raise ValueError(f"invalid indicator code {code!r}")
    if SEP in code:
        have, bare = split(code)
        if have != subject or not bare:
            raise ValueError(f"indicator {code!r} does not belong to subject {subject!r}")
        return code
    return f"{subject}{SEP}{code}"


def split(id: str, default_subject: str | None = None) -> tuple[str | None, str]:
    """`(subject, code)` of an id. A bare code gives `(default_subject, code)`."""
    if not isinstance(id, str) or not id:
        raise ValueError(f"invalid indicator id {id!r}")
    subject, sep, code = id.partition(SEP)
    if not sep:
        return default_subject, id
    if not subject or not code:
        raise ValueError(f"invalid indicator id {id!r}")
    return subject, code


def code_of(id: str) -> str:
    """The bare code of an id: `code_of("math:B4.1.1.1.2") -> "B4.1.1.1.2"`. A bare code is returned as it is."""
    return split(id)[1]


def subject_of(id: str, default: str | None = None) -> str | None:
    """The subject of an id, or `default` for a bare code."""
    return split(id, default)[0]


class Resolver:
    """Maps any accepted spelling of an indicator id to the node key of one graph.

    Accepted: the node key itself, the canonical id, the bare code, and every entry of a node's
    `aliases` (bare or namespaced). In a v2 graph the node key is the canonical id, so `resolve`
    returns canonical ids. In a v1 graph the node keys are bare codes and `resolve` returns those,
    which lets the same caller code run on both.

    A node's own id always wins over another node's alias with the same spelling.
    """

    def __init__(self, G: Any, subject: str | None = None) -> None:
        graph_attrs = getattr(G, "graph", None)
        if subject is None and isinstance(graph_attrs, Mapping):
            subject = graph_attrs.get("subject")
        self.subject: str = subject or DEFAULT_SUBJECT
        self._index: dict[str, str] = {}
        nodes = list(self._nodes(G))
        for key, data in nodes:  # own ids first, so they beat aliases
            for spelling in self._spellings(key, data.get("subject")):
                self._index.setdefault(spelling, key)
        for key, data in nodes:
            for alias in data.get("aliases") or ():
                if isinstance(alias, str) and alias:
                    for spelling in self._spellings(alias, data.get("subject")):
                        self._index.setdefault(spelling, key)

    @staticmethod
    def _nodes(G: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
        if hasattr(G, "nodes") and callable(getattr(G.nodes, "data", None)):
            return ((str(n), d) for n, d in G.nodes(data=True))
        if isinstance(G, Mapping):  # {id: record}
            return ((str(n), d if isinstance(d, Mapping) else {}) for n, d in G.items())
        return ((str(r["id"]), r) for r in G)  # an iterable of indicator records

    def _spellings(self, id: str, node_subject: str | None) -> tuple[str, str]:
        subject, code = split(id, node_subject or self.subject)
        return code, f"{subject}{SEP}{code}"

    def _lookup(self, id: Any) -> str | None:
        if not isinstance(id, str):
            return None
        hit = self._index.get(id)
        if hit is not None:
            return hit
        text = id.strip()
        if not text:
            return None
        subject, sep, code = text.partition(SEP)
        relaxed = f"{subject.lower()}{SEP}{code.upper()}" if sep else text.upper()
        return self._index.get(text) or self._index.get(relaxed)

    def resolve(self, id: str) -> str:
        """The node key for `id`. Raises `KeyError("unknown indicator '...'")` when nothing matches."""
        hit = self._lookup(id)
        if hit is None:
            raise KeyError(f"unknown indicator '{id}'")
        return hit

    def get(self, id: str, default: str | None = None) -> str | None:
        """Like `resolve`, but returns `default` for an unknown id."""
        hit = self._lookup(id)
        return default if hit is None else hit

    def __contains__(self, id: object) -> bool:
        return self._lookup(id) is not None

    def __len__(self) -> int:
        return len(set(self._index.values()))
