"""`llm-json`: any chat model that answers `{"probability": float}` over HTTP.

NOT RUN AGAINST A LIVE SERVICE. This provider was written and unit-tested with a mock HTTP transport
only. No API key was available when it was written, so no real model has answered through it, and
it has no Junyi benchmark of its own. Before using its output, run the Junyi benchmark
(`kwan graph augment --junyi raw/junyi --provider llm-json`) and choose a threshold from that run.

The request is the widely used "chat completions" shape: `POST <endpoint>` with
`{"model": ..., "messages": [{"role": "user", "content": <prompt>}], "temperature": 0}` and the
answer text in `choices[0].message.content`. Hosted services and local servers (for example Ollama
or llama.cpp) both speak it. Settings come from the environment:

    KWAN_LLM_ENDPOINT   full URL of the chat completions endpoint (required)
    KWAN_LLM_MODEL      model id (required)
    KWAN_LLM_API_KEY    bearer token (optional: a local server needs none)
    KWAN_LLM_COST_PER_CALL_USD   what one call costs, for the spend log (optional, default 0)

It never runs in CI.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from typing import Any

from kwan.graph.augment import PairQuestion, render_prompt

ENV_ENDPOINT = "KWAN_LLM_ENDPOINT"
ENV_MODEL = "KWAN_LLM_MODEL"
ENV_KEY = "KWAN_LLM_API_KEY"
ENV_COST = "KWAN_LLM_COST_PER_CALL_USD"

__all__ = ["LLMJsonDecisionModel", "parse_probability"]

_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)
_FIELD = re.compile(r"""["']?probability["']?\s*[:=]\s*["']?(-?\d+(?:\.\d+)?)\s*(%?)""", re.IGNORECASE)


def _as_probability(value: Any, percent: bool = False) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        percent = percent or text.endswith("%")
        try:
            value = float(text.rstrip("%").strip())
        except ValueError:
            return None
    if not isinstance(value, int | float) or value != value:  # NaN is not a probability
        return None
    p = float(value) / 100.0 if percent else float(value)
    return p if 0.0 <= p <= 1.0 else None


def parse_probability(text: str) -> float:
    """Read the probability out of a model's answer.

    Accepted: a JSON object with a `probability` key, the same inside a code fence or with words
    around it, a string or percentage value (`"0.7"`, `"70%"`), and a loose `probability: 0.7`. A
    value outside 0 to 1, or no value at all, raises `ValueError`; nothing is guessed.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the model returned no text")
    for chunk in [text.strip(), *_OBJECT.findall(text)]:
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(obj, dict):
            key = next((k for k in obj if isinstance(k, str) and k.strip().lower() == "probability"), None)
            if key is not None:
                p = _as_probability(obj[key])
                if p is None:
                    raise ValueError(f"probability out of range or not a number: {obj[key]!r}")
                return p
    m = _FIELD.search(text)
    if m:
        p = _as_probability(m.group(1), percent=bool(m.group(2)))
        if p is None:
            raise ValueError(f"probability out of range: {m.group(0)!r}")
        return p
    raise ValueError(f"no probability in the model's answer: {text[:120]!r}")


class LLMJsonDecisionModel:
    """Ask a chat model the question and parse `{"probability": float}` from its answer.

    Not run against a live service here (see the module docstring). `client` is an `httpx.Client`; the
    tests pass one built on `httpx.MockTransport`. A `calibrate` function, when given, maps the raw
    answer to a calibrated probability (fit it on the Junyi training split and keep it under `build/`).
    """

    name = "llm-json"
    licence = "depends on the model behind the endpoint"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        timeout: float = 60.0,
        retries: int = 2,
        cost_per_call_usd: float | None = None,
        calibrate: Callable[[float], float] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint or os.environ.get(ENV_ENDPOINT, "")
        self.model = model or os.environ.get(ENV_MODEL, "")
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_KEY, "")
        if not self.endpoint or not self.model:
            raise RuntimeError(
                f"llm-json needs an endpoint and a model: set {ENV_ENDPOINT} and {ENV_MODEL} "
                f"(and {ENV_KEY} when the service wants a key)."
            )
        if cost_per_call_usd is None:
            cost_per_call_usd = float(os.environ.get(ENV_COST, "0") or 0)
        self.cost_per_call_usd = cost_per_call_usd
        self.last_cost_usd = 0.0
        self.retries = retries
        self.calibrate = calibrate
        self._sleep = sleep
        self._timeout = timeout
        self._client = client

    def _http(self) -> Any:
        if self._client is None:
            import httpx  # optional dependency: graph-ai extra

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def prompt(self, q: PairQuestion) -> str:
        """The user message for `q`."""
        return render_prompt(q)

    def prerequisite_probability(self, q: PairQuestion) -> float:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": [{"role": "user", "content": self.prompt(q)}], "temperature": 0}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(min(2.0**attempt, 30.0))
            response = self._http().post(self.endpoint, headers=headers, json=body)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"llm-json: the service answered HTTP {response.status_code}")
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"llm-json: the service answered HTTP {response.status_code}")
            try:
                content = response.json()["choices"][0]["message"]["content"]
                p = parse_probability(content)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = ValueError(f"llm-json: could not read a probability from the answer ({exc})")
                continue
            self.last_cost_usd = self.cost_per_call_usd
            if self.calibrate is not None:
                p = min(1.0, max(0.0, float(self.calibrate(p))))
            return p
        assert last_error is not None
        raise last_error
