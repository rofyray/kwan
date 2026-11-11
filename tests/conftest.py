"""Shared fixtures. Tests run from any working directory: data paths are resolved from the repo root.

`G` is the committed v2 graph (`data/graph/math.graph.json`, namespaced ids, short labels). `G_v1` and
`indicators_v1` are the frozen v1 files under `tests/fixtures/`, with every NaCCA description replaced
by the Kwan short label: the v1-equivalence tests pin v1 behaviour on them.
"""

import json
from pathlib import Path

import pytest

from kwan.graph import build, schema

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GRAPH_V1 = FIXTURES / "graph_v1.json"
INDICATORS_V1 = FIXTURES / "indicators_v1.json"


@pytest.fixture(scope="session")
def G():
    """The committed NaCCA B1 to B6 curriculum graph, schema v2 (read-only: do not mutate in tests)."""
    return schema.load_graph()


@pytest.fixture(scope="session")
def G_v1():
    """The frozen v1 graph (bare ids, 266 nodes, 287 edges), descriptions replaced by short labels."""
    return build.load(GRAPH_V1)


@pytest.fixture(scope="session")
def indicators_v1():
    """The frozen v1 indicator records, descriptions replaced by short labels."""
    return json.loads(INDICATORS_V1.read_text(encoding="utf-8"))
