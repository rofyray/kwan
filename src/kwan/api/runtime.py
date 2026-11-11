"""What the API runs on: the engine, the learner store, the API keys and the JSON logger.

Settings come from the environment:

    KWAN_API_KEYS   comma-separated keys accepted in the `X-API-Key` header. Unset: every request is
                    allowed and a warning is logged once (fine for the demo and local work).
    KWAN_STORE      `memory`, or `sqlite:///<path>` (default `sqlite:///kwan.sqlite3`, a file in the
                    working directory; `sqlite:///:memory:` keeps it in memory).
    KWAN_BUNDLE     path of the bundle to serve (default: the bundle shipped with the package).

Logs are one JSON object per line on the `kwan.api` logger, so a log collector can read them.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kwan.kwan import Kwan
from kwan.store import LearnerStore, MemoryStore, SQLiteStore

ENV_KEYS = "KWAN_API_KEYS"
ENV_STORE = "KWAN_STORE"
ENV_BUNDLE = "KWAN_BUNDLE"
DEFAULT_STORE = "sqlite:///kwan.sqlite3"
LOGGER = "kwan.api"

__all__ = [
    "DEFAULT_STORE",
    "ENV_BUNDLE",
    "ENV_KEYS",
    "ENV_STORE",
    "LOGGER",
    "JsonFormatter",
    "Runtime",
    "from_env",
    "logger",
    "open_store",
    "parse_keys",
]


# ----------------------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    """One JSON object per log line: time, level, logger, message and any `extra` fields."""

    _SKIP = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        doc.update({k: v for k, v in record.__dict__.items() if k not in self._SKIP and not k.startswith("_")})
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False, default=str)


def logger() -> logging.Logger:
    """The `kwan.api` logger, given a JSON handler on first use (unless one was configured already)."""
    log = logging.getLogger(LOGGER)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


# ----------------------------------------------------------------------------- settings


def parse_keys(text: str | None) -> frozenset[str]:
    """The API keys in a comma-separated string (blank entries are ignored)."""
    return frozenset(k.strip() for k in (text or "").split(",") if k.strip())


def open_store(url: str | None) -> tuple[LearnerStore, str]:
    """Open the learner store named by a `KWAN_STORE` value; returns `(store, kind)`.

    `memory` gives a `MemoryStore`; `sqlite:///<path>` a `SQLiteStore` (`sqlite:///:memory:` in memory).
    Raises `ValueError` for anything else."""
    url = (url or DEFAULT_STORE).strip()
    if url in ("memory", "memory://"):
        return MemoryStore(), "memory"
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///") :] or ":memory:"
        return SQLiteStore(path), "sqlite"
    raise ValueError(f"{ENV_STORE} must be 'memory' or 'sqlite:///<path>', got {url!r}")


# ----------------------------------------------------------------------------- the runtime


@dataclass
class Runtime:
    """Everything a request handler needs. One per process; safe to share between threads."""

    engine: Kwan
    store: LearnerStore
    api_keys: frozenset[str] = frozenset()
    store_kind: str = "memory"
    log: logging.Logger = field(default_factory=logger)
    _warned: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def auth_enabled(self) -> bool:
        """True when at least one API key is configured."""
        return bool(self.api_keys)

    def key_ok(self, key: str | None) -> bool:
        """Whether a request with this `X-API-Key` may proceed. Without configured keys, always, with
        a warning logged the first time."""
        if not self.api_keys:
            with self._lock:
                if not self._warned:
                    self._warned = True
                    self.log.warning(
                        f"{ENV_KEYS} is not set: every request is allowed. Set it before exposing the API.",
                        extra={"event": "auth_disabled"},
                    )
            return True
        return key is not None and key in self.api_keys

    def close(self) -> None:
        """Close the store."""
        self.store.close()


def from_env(
    environ: dict[str, str] | None = None, bundle_loader: Callable[[str | None], Kwan] | None = None
) -> Runtime:
    """Build the runtime from environment variables (see the module notes)."""
    env = os.environ if environ is None else environ
    load = bundle_loader or Kwan.from_bundle
    engine = load(env.get(ENV_BUNDLE) or None)
    store, kind = open_store(env.get(ENV_STORE))
    rt = Runtime(engine=engine, store=store, api_keys=parse_keys(env.get(ENV_KEYS)), store_kind=kind)
    rt.log.info(
        "kwan api ready",
        extra={
            "event": "ready",
            "bundle_version": engine.bundle_version,
            "store": kind,
            "auth": rt.auth_enabled,
            "skills": len(engine),
        },
    )
    return rt
