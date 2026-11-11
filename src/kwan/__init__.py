"""Kwan: an open-source learning path engine for Ghanaian primary maths (NaCCA B1 to B6).

Kwan keeps track of what a pupil has mastered, picks the next skill to learn, gives the reason and
sets the practice in everyday Ghanaian life. The core needs only the standard library and networkx.

    from kwan import Kwan

    engine = Kwan.from_bundle()
    state = engine.new_state("pupil-0042")
    engine.update(state, "B1.1.1.1.1", correct=True)
    for rec in engine.recommend(state, k=3):
        print(rec.indicator, rec.rationale)
"""

from typing import Any

from kwan._version import __version__

__all__ = ["Kwan", "LearnerState", "Recommendation", "__version__"]

_PUBLIC = {"Kwan": "kwan.kwan", "LearnerState": "kwan.learner.elo", "Recommendation": "kwan.kwan"}


def __getattr__(name: str) -> Any:
    """Load the public classes on first use, so `import kwan` stays cheap."""
    module = _PUBLIC.get(name)
    if module is None:
        raise AttributeError(f"module 'kwan' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC))
