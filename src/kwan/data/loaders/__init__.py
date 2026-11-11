"""Loaders: every learner dataset in one interaction schema (`kwan.data.schema`).

    from kwan.data import loaders
    df = loaders.load("ammore_2024:ghana", "raw/ammore")

`LOADERS` maps a results-file key (or `key:subset`) to (module, function). Modules are imported when
a loader is called, so importing this package needs neither pandas nor pyarrow.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

LOADERS: dict[str, tuple[str, str]] = {
    "assistments_2009": ("assistments", "load_2009"),
    "assistments_2012": ("assistments", "load_2012"),
    "assistments_2015": ("assistments", "load_2015"),
    "assistments_2017": ("assistments", "load_2017"),
    "eedi_2020_task34": ("eedi", "load_task34"),
    "eedi_2020_task12": ("eedi", "load_task12"),
    "ammore_2024": ("ammore", "load"),
    "ammore_2024:all": ("ammore", "load"),
    "ammore_2024:ghana": ("ammore", "load_ghana"),
    "xes3g5m": ("xes3g5m", "load"),
    "junyi_2015": ("junyi", "load"),
}

# where each dataset sits under the raw-data directory after `make data`
RAW_PATHS: dict[str, str] = {
    "assistments_2009": "skill_builder_data_corrected.csv",
    "assistments_2012": "assist2012",
    "assistments_2015": "assist2015",
    "assistments_2017": "assist2017",
    "eedi_2020_task34": "eedi/data",
    "eedi_2020_task12": "eedi/data",
    "ammore_2024": "ammore",
    "xes3g5m": "xes3g5m",
    "junyi_2015": "junyi",
}


def loader(name: str) -> Any:
    """The loader function for a results-file key such as `eedi_2020_task12` or `ammore_2024:ghana`."""
    try:
        module, func = LOADERS[name]
    except KeyError:
        raise KeyError(f"no loader for '{name}' (known: {', '.join(LOADERS)})") from None
    return getattr(importlib.import_module(f"kwan.data.loaders.{module}"), func)


def raw_path(name: str, raw_dir: str | Path = "raw") -> Path:
    """Default place of a dataset under the raw-data directory."""
    return Path(raw_dir) / RAW_PATHS[name.split(":", 1)[0]]


def load(name: str, path: str | Path | None = None, raw_dir: str | Path = "raw", **kwargs: Any) -> Any:
    """Load a dataset in the interaction schema, from `path` or from its default place under `raw_dir`."""
    return loader(name)(path if path is not None else raw_path(name, raw_dir), **kwargs)
