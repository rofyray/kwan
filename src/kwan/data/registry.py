"""The dataset registry: `data/datasets.yaml` and the licence facts the core package needs.

`load_registry()` reads the YAML file (PyYAML comes with the `eval` extra and is imported inside the
function). `commercial_ok()` and `licence()` answer from the small table below, with no optional
package, so the bundle builder can refuse parameters fitted on non-commercial data in any install.
A test proves that the table and the YAML file say the same thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kwan._paths import data_path

FILE_NAME = "datasets.yaml"

# registry key -> (licence, parameters fitted on it may ship in a bundle)
FACTS: dict[str, tuple[str, bool]] = {
    "assistments_2009": ("research use; cite Feng, Heffernan & Koedinger (2009)", True),
    "eedi_2020": ("research use; cite Wang et al. (2020) arXiv:2007.12061", True),
    "ammore_2024": ("CC BY-NC 4.0", False),
    "assistments_2012": ("research use", True),
    "assistments_2015": ("research use", True),
    "assistments_2017": ("research use", True),
    "xes3g5m": ("MIT", True),
    "junyi_2015": ("non-commercial; cite Chang, Hsu & Chen (EDM 2015)", False),
}

# other names a dataset goes by (results-file keys, subsets) -> registry key
ALIASES: dict[str, str] = {
    "eedi_2020_task34": "eedi_2020",
    "eedi_2020_task12": "eedi_2020",
    "ammore_2024_all": "ammore_2024",
    "ammore_2024_ghana": "ammore_2024",
    "ammore": "ammore_2024",
    "junyi": "junyi_2015",
}


def dataset_key(name: str) -> str:
    """The registry key for a dataset name, a results-file key or a `dataset:subset` name.

    Raises KeyError("unknown dataset '...'") for anything the registry does not list.
    """
    key = str(name).strip()
    for candidate in (key, key.replace("-", "_"), key.split(":", 1)[0], key.split(":", 1)[0].replace("-", "_")):
        if candidate in FACTS:
            return candidate
        if candidate in ALIASES:
            return ALIASES[candidate]
    raise KeyError(f"unknown dataset '{name}'")


def commercial_ok(dataset: str) -> bool:
    """True when parameters fitted on this dataset may ship in a Kwan bundle (PRD 5.3).

    False for AMMORE (CC BY-NC) and Junyi (non-commercial): evaluation only. Unknown datasets raise
    KeyError, so a typo can never pass as permission.
    """
    return FACTS[dataset_key(dataset)][1]


def licence(dataset: str) -> str:
    """The licence text of a dataset, as written in the registry."""
    return FACTS[dataset_key(dataset)][0]


def non_commercial() -> tuple[str, ...]:
    """Registry keys of the datasets that are evaluation only."""
    return tuple(key for key, (_, ok) in FACTS.items() if not ok)


def registry_path() -> Path:
    """Where `datasets.yaml` lives (the data directory of a source checkout, or `KWAN_DATA_DIR`)."""
    return data_path(FILE_NAME)


def load_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the dataset registry: {registry key: {name, url, size, licence, citation, role, ...}}."""
    import yaml

    path = Path(path) if path is not None else registry_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping of dataset keys")
    return data


def get(dataset: str, path: str | Path | None = None) -> dict[str, Any]:
    """One registry entry, looked up by registry key, results-file key or alias."""
    return load_registry(path)[dataset_key(dataset)]


def raw_path(dataset: str, raw_dir: str | Path = "raw", path: str | Path | None = None) -> Path:
    """Where a dataset lives under the raw-data directory after `make data`."""
    return Path(raw_dir) / get(dataset, path)["raw"]
