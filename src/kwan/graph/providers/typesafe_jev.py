"""`typesafe-jev`: the hosted Jev decision model, through a gateway's documented decisions endpoint.

NOT RUN AGAINST A LIVE SERVICE, AND READ THIS FIRST. When this was written (2026-11-10):

* TypeSafe's own API reference could not be found, so its native API is NOT implemented here and
  nothing about it is assumed.
* What could be read is the public reference of one gateway that serves the model, AI/ML API:
  https://docs.aimlapi.com/api-references/decision-models/typesafe/jev . This provider follows that
  page and nothing else: `POST <endpoint>` with a bearer key and the body
  `{"model": "typesafe/jev", "state": <text>, "questions": {"prerequisite": {"type": "noul",
  "instructions": <the yes/no question>}}}`; the answer is read from
  `answers.prerequisite.noul`, the probability of "yes".
* No key was available, so it has been unit-tested with a mock HTTP transport only. It has no Junyi
  benchmark. Run `kwan graph augment --junyi raw/junyi --provider typesafe-jev` before trusting it.

Because the endpoint depends on which service you use, there is no default: set

    KWAN_JEV_ENDPOINT   full URL of the decisions endpoint, e.g. the gateway's `/v1/decisions`
    KWAN_JEV_API_KEY    bearer key for that service
    KWAN_JEV_MODEL      model id (optional, default `typesafe/jev`; pin a version for a stable run)

Without them the constructor raises a clear error. If you use TypeSafe directly or another gateway
whose request or response differs, change `_body` and `_read` below; they are the only two places
that know the contract. It never runs in CI.
"""

from __future__ import annotations

import os
from typing import Any

from kwan.graph.augment import PairQuestion, render_prompt

ENV_ENDPOINT = "KWAN_JEV_ENDPOINT"
ENV_KEY = "KWAN_JEV_API_KEY"
ENV_MODEL = "KWAN_JEV_MODEL"
DEFAULT_MODEL = "typesafe/jev"
CONTRACT_SOURCE = "https://docs.aimlapi.com/api-references/decision-models/typesafe/jev"
QUESTION_KEY = "prerequisite"

__all__ = ["TypesafeJevDecisionModel"]


class TypesafeJevDecisionModel:
    """Ask Jev a yes/no ("noul") question through a gateway's decisions endpoint. Not run live here."""

    name = "typesafe-jev"
    licence = "hosted commercial service; see the provider's terms"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = endpoint or os.environ.get(ENV_ENDPOINT, "")
        self.api_key = api_key or os.environ.get(ENV_KEY, "")
        self.model = model or os.environ.get(ENV_MODEL, "") or DEFAULT_MODEL
        if not self.endpoint or not self.api_key:
            raise RuntimeError(
                "typesafe-jev is not set up. TypeSafe's own API reference was not available when this "
                f"provider was written, so there is no default endpoint. Set {ENV_ENDPOINT} and {ENV_KEY} "
                f"for a service that follows the decisions contract documented at {CONTRACT_SOURCE}. "
                "For any other contract, edit `_body` and `_read` in src/kwan/graph/providers/typesafe_jev.py."
            )
        self.last_cost_usd = 0.0
        self._timeout = timeout
        self._client = client

    def _http(self) -> Any:
        if self._client is None:
            import httpx  # optional dependency: graph-ai extra

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def prompt(self, q: PairQuestion) -> str:
        """The `state` text: both skills, without the answer-format line a chat model needs."""
        return render_prompt(q).rsplit("\nQuestion:", 1)[0]

    def _body(self, q: PairQuestion) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": self.prompt(q),
            "questions": {QUESTION_KEY: {"type": "noul", "instructions": q.question}},
        }

    def _read(self, payload: Any) -> float:
        try:
            value = payload["answers"][QUESTION_KEY]["noul"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"typesafe-jev: no 'answers.{QUESTION_KEY}.noul' in the response") from exc
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"typesafe-jev: {value!r} is not a probability")
        return float(value)

    def prerequisite_probability(self, q: PairQuestion) -> float:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = self._http().post(self.endpoint, headers=headers, json=self._body(q))
        if response.status_code >= 400:
            raise RuntimeError(f"typesafe-jev: the service answered HTTP {response.status_code}")
        payload = response.json()
        p = self._read(payload)
        meta = payload.get("meta") if isinstance(payload, dict) else None
        spent = meta.get("usd_spent") if isinstance(meta, dict) else None  # not in the documented schema
        self.last_cost_usd = float(spent) if isinstance(spent, int | float) and not isinstance(spent, bool) else 0.0
        return p
