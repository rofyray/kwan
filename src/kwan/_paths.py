"""One place that finds the repository `data/` directory.

Order: the `KWAN_DATA_DIR` environment variable, else the `data/` folder next to the nearest
`pyproject.toml` above this file (a source checkout or an editable install), else `data/` under the
current working directory (an installed wheel, where the caller brings its own data).
"""

import os
from pathlib import Path

ENV_VAR = "KWAN_DATA_DIR"


def repo_root() -> Path | None:
    """The directory holding `pyproject.toml` above this file, or None outside a source checkout."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def data_dir() -> Path:
    """Absolute path of the directory that holds `graph/`, `curriculum/`, `contexts/` and eval_results.json."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    root = repo_root()
    return (root if root is not None else Path.cwd().resolve()) / "data"


def data_path(name: str) -> Path:
    """Absolute path of one file in the data directory, for example `data_path("eval_results.json")`."""
    return data_dir() / name
