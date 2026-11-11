"""Decision-model providers for the AI-augmented graph (PRD Phase 7, task 7.2).

| name            | module            | what it is                                                        |
|-----------------|-------------------|-------------------------------------------------------------------|
| `fake`          | `fake.py`         | fixed outputs; tests and `kwan graph augment --dry-run`           |
| `local-encoder` | `local_encoder.py`| open-weight sentence encoder plus a small calibrated classifier   |
| `llm-json`      | `llm_json.py`     | any chat model that answers `{"probability": float}` over HTTP     |
| `typesafe-jev`  | `typesafe_jev.py` | hosted decision model, through a gateway's documented endpoint    |

Nothing heavy is imported here. `get_provider` imports the module it needs when it is asked for it.
Only `fake` runs in CI.
"""

from __future__ import annotations

from typing import Any

PROVIDERS = ("fake", "local-encoder", "llm-json", "typesafe-jev")

__all__ = ["PROVIDERS", "get_provider"]


def get_provider(name: str, **kwargs: Any) -> Any:
    """Build a provider by name. Keyword arguments go to its constructor."""
    if name == "fake":
        from kwan.graph.providers.fake import FakeDecisionModel

        return FakeDecisionModel(**kwargs)
    if name == "local-encoder":
        from kwan.graph.providers.local_encoder import LocalEncoderDecisionModel

        return LocalEncoderDecisionModel(**kwargs)
    if name == "llm-json":
        from kwan.graph.providers.llm_json import LLMJsonDecisionModel

        return LLMJsonDecisionModel(**kwargs)
    if name == "typesafe-jev":
        from kwan.graph.providers.typesafe_jev import TypesafeJevDecisionModel

        return TypesafeJevDecisionModel(**kwargs)
    raise ValueError(f"unknown provider {name!r}: choose from {', '.join(PROVIDERS)}")
