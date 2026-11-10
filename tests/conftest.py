"""Shared fixtures. Tests run from any working directory: data paths are resolved from the repo root."""
import json
from pathlib import Path

import pytest

import graph

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def G():
    """The committed NaCCA B1 to B6 curriculum graph (read-only: do not mutate in tests)."""
    return graph.load(DATA / "graph.json")


@pytest.fixture(scope="session")
def indicators():
    return json.loads((DATA / "indicators.json").read_text(encoding="utf-8"))
