"""Kwan REST API (FastAPI).

Run:                      uvicorn kwan.api.app:app --port 8000
Interactive docs:         http://localhost:8000/docs

Two route families live here:

- **v1** (`/v1/...`, see `kwan.api.v1`): built on `kwan.Kwan` and `kwan.store`, with API-key auth
  (`KWAN_API_KEYS`), a persistent store (`KWAN_STORE`, SQLite by default), idempotent attempts and
  JSON logs. This is the API to build on.
- **v0** (`/health`, `/attempt`, `/recommend/{id}`, `/mastery/{id}`, `/learner/{id}`): the Round 1
  routes, kept for one minor release. Learner state is in memory only. Every v0 response carries a
  `Deprecation` header and a `Link` to the v1 route that replaces it.

`create_app(runtime)` builds an application around a given runtime (tests pass a memory store and
their own keys); the module-level `app` reads its settings from the environment on first use.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from kwan import ids
from kwan._version import __version__
from kwan.api import runtime as _runtime
from kwan.api import v1 as _v1
from kwan.graph.schema import load_graph
from kwan.learner.elo import EloLearner, LearnerState
from kwan.recommend.recommender import Recommender

G = load_graph()  # the committed v2 graph
RESOLVE = ids.Resolver(G)  # the v0 routes accept bare codes (as Round 1 clients send them) and namespaced ids
L = EloLearner(G=G)


def _bare(node: str) -> str:
    """The bare code of a node: what the v0 routes return, as they did on the v1 graph."""
    return G.nodes[node].get("code") or ids.code_of(node)


R = Recommender(G, L)
STATES: dict[str, LearnerState] = {}  # v0 in-memory state
LOCK = threading.RLock()  # sync endpoints run in a thread pool: one writer at a time

LEARNER_ID_MAX = 128
INDICATOR_ID_MAX = 64
K_MAX = 50
V0_PATHS = ("/health", "/attempt", "/recommend/", "/mastery/", "/learner/")

DESCRIPTION = """
Adaptive learning engine for Ghanaian pupils, on the NaCCA curriculum. This release covers Mathematics B1 to B6.

Typical flow for a platform such as Syto:

1. `POST /v1/attempts` each time a pupil answers a question. The reply carries the probability the
   engine predicted before seeing the answer and the pupil's updated mastery of that skill.
2. `GET /v1/learners/{learner_id}/recommendations` to get the next best skills, each with a
   plain-language reason and a practice setting from everyday Ghanaian life.
3. `GET /v1/learners/{learner_id}/mastery` to draw the mastery map for a teacher.

**Auth.** Send one of the keys in `KWAN_API_KEYS` in the `X-API-Key` header. When no key is
configured, every request is allowed (local use only).

**New learners.** A `learner_id` needs no registration. A learner who has never posted an attempt is
treated as a brand-new pupil. Reads never create or change stored state. A learner is stored on the
first `POST /v1/attempts` and removed by `DELETE /v1/learners/{learner_id}`.

**Ids.** Indicator ids are `math:B1.1.1.1.1`; the bare code `B1.1.1.1.1` is accepted everywhere.

**Errors.** `{"detail": "...", "code": "..."}`. An unknown indicator or goal gives 404; a request that
fails validation gives 422 with the offending field in `detail`; a missing key gives 401.

**v0 routes** (`/attempt`, `/recommend/{id}`, ...) are deprecated and answer with a `Deprecation`
header and a `Link` to their v1 replacement. They keep state in memory only.
"""

# --------------------------------------------------------------------------- v0 models


class Attempt(BaseModel):
    """One answer event from a pupil (v0)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"learner_id": "a", "indicator": "B1.1.1.1.1", "correct": True, "item_id": None}]
        }
    )

    learner_id: str = Field(
        min_length=1,
        max_length=LEARNER_ID_MAX,
        pattern=r"\S",
        description="Your identifier for the pupil. Any non-blank string, no personal data.",
    )
    indicator: str = Field(
        min_length=1,
        max_length=INDICATOR_ID_MAX,
        description="NaCCA indicator id of the skill that was practised, e.g. B1.1.1.1.1.",
    )
    correct: StrictBool = Field(description="Whether the answer was right. Must be a JSON boolean.")
    item_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional id of the question. Used for item difficulty when the engine "
        "has it; otherwise the item is treated as typical for the skill.",
    )


class AttemptResponse(BaseModel):
    predicted_p_correct: float = Field(
        ge=0,
        le=1,
        description="Probability of a correct answer that the engine "
        "predicted before seeing this attempt (4 decimals).",
    )
    new_mastery: float = Field(ge=0, le=1, description="Mastery of the indicator after this attempt (4 decimals).")
    model_config = ConfigDict(json_schema_extra={"examples": [{"predicted_p_correct": 0.3775, "new_mastery": 0.4839}]})


class HealthResponse(BaseModel):
    ok: bool = Field(description="True when the engine is loaded and able to serve.")
    nodes: int = Field(description="Number of indicators (skills) in the curriculum graph.")
    edges: int = Field(description="Number of prerequisite links in the curriculum graph.")
    model_config = ConfigDict(json_schema_extra={"examples": [{"ok": True, "nodes": 266, "edges": 287}]})


class Recommendation(BaseModel):
    indicator: str = Field(description="NaCCA indicator id of the recommended skill.")
    score: float = Field(description="Ranking score; higher is better. Comparable only within one response.")
    mastery: float = Field(ge=0, le=1, description="The learner's current mastery of this skill (3 decimals).")
    grade: int = Field(ge=1, le=6, description="Class level, 1 to 6 for B1 to B6.")
    strand: str = Field(description="Curriculum strand name.")
    description: str = Field(description="The indicator text from the NaCCA curriculum.")
    rationale: str = Field(description="Plain-language reason, built only from the signals behind the score.")
    ghana_context: str = Field(description="An everyday Ghanaian setting to practise the skill in.")


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation] = Field(
        description="Best first. May hold fewer than k items, or none, when fewer skills are ready for this learner."
    )
    latency_ms: float = Field(
        ge=0, description="Engine compute time for this call in milliseconds, excluding HTTP and JSON overhead."
    )


class ErrorResponse(BaseModel):
    detail: str = Field(description="What went wrong.")


LearnerId = Annotated[
    str,
    Path(
        min_length=1,
        max_length=LEARNER_ID_MAX,
        pattern=r"\S",
        description="Your identifier for the pupil. Any non-blank string.",
    ),
]

NOT_FOUND = {"model": ErrorResponse}

# --------------------------------------------------------------------------- v0 state


def state(lid: str) -> LearnerState:
    """Stored state for a learner, created on first use. Call with LOCK held."""
    return STATES.setdefault(lid, LearnerState(lid))


def peek(lid: str) -> LearnerState:
    """State to read from: the stored one, or a throwaway fresh one for a learner never seen. Call with LOCK held."""
    return STATES.get(lid) or LearnerState(lid)


# --------------------------------------------------------------------------- v0 endpoints


def v0_successor(path: str) -> str | None:
    """The v1 path that replaces a v0 path, or None when the path is not a v0 route."""
    if path == "/health":
        return "/v1/health"
    if path == "/attempt":
        return "/v1/attempts"
    for prefix, tail in (("/recommend/", "/recommendations"), ("/mastery/", "/mastery"), ("/learner/", "")):
        if path.startswith(prefix) and len(path) > len(prefix):
            return f"/v1/learners/{path[len(prefix) :]}{tail}"
    return None


def register_v0(app: FastAPI) -> None:
    """Add the Round 1 routes (deprecated, in-memory state) to an application."""

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["v0 (deprecated)"],
        summary="Liveness check and graph size",
        deprecated=True,
    )
    def health():
        return {"ok": True, "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

    @app.post(
        "/attempt",
        response_model=AttemptResponse,
        tags=["v0 (deprecated)"],
        summary="Record one answer",
        deprecated=True,
        responses={404: {**NOT_FOUND, "description": "Unknown indicator."}},
    )
    def attempt(a: Attempt):
        """Update the learner model with one answer and return the new mastery of that skill.

        The learner is created on first use. Updates for the same learner are applied one at a time,
        so concurrent requests cannot corrupt the state.
        """
        indicator = RESOLVE.get(a.indicator)
        if indicator is None:
            raise HTTPException(404, f"unknown indicator {a.indicator}")
        with LOCK:
            st = state(a.learner_id)
            p = L.update(st, indicator, a.correct, a.item_id)
            m = L.mastery(st, indicator)
        return {"predicted_p_correct": round(p, 4), "new_mastery": round(m, 4)}

    @app.get(
        "/recommend/{learner_id}",
        response_model=RecommendResponse,
        tags=["v0 (deprecated)"],
        summary="Next best skills for a learner",
        deprecated=True,
        responses={404: {**NOT_FOUND, "description": "Unknown goal."}},
    )
    def recommend(
        learner_id: LearnerId,
        k: Annotated[int, Query(ge=1, le=K_MAX, description="How many skills to return at most.")] = 3,
        goal: Annotated[
            str | None,
            Query(
                max_length=INDICATOR_ID_MAX,
                description="Indicator id of a target skill. "
                "Skills on the path to it are ranked first. Leave out or send empty for no goal.",
            ),
        ] = None,
        grade_max: Annotated[
            int | None,
            Query(
                ge=1,
                le=6,
                description="Class cap: nothing above this class level is recommended (4 means nothing above B4).",
            ),
        ] = None,
        goal_mode: Annotated[
            Literal["strict", "soft"],
            Query(
                description="How the goal is used. strict: every skill "
                "on the path to the goal comes before every other skill. soft: the goal only raises the "
                "score of skills on the path, so other skills can be mixed in."
            ),
        ] = "strict",
    ):
        """Skills the learner is ready for and has not mastered, best first.

        A learner who has never posted an attempt gets the starting recommendations for a new pupil.
        This call does not change any stored state.
        """
        if goal:
            resolved = RESOLVE.get(goal)
            if resolved is None:
                raise HTTPException(404, f"unknown goal {goal}")
            goal = resolved
        else:
            goal = None
        with LOCK:
            out = R.recommend(peek(learner_id), k=k, goal=goal, grade_max=grade_max, goal_mode=goal_mode)
        for rec in out["recommendations"]:
            rec["indicator"] = _bare(rec["indicator"])
        return out

    @app.get(
        "/mastery/{learner_id}",
        response_model=dict[str, float],
        tags=["v0 (deprecated)"],
        summary="Mastery of every skill for a learner",
        deprecated=True,
    )
    def mastery(learner_id: LearnerId):
        """Map of every indicator id in the graph to the learner's mastery, 0 to 1 (3 decimals).

        A learner who has never posted an attempt gets the starting estimates for a new pupil.
        This call does not change any stored state.
        """
        with LOCK:
            st = peek(learner_id)
            return {_bare(n): round(L.mastery(st, n), 3) for n in G}

    @app.delete(
        "/learner/{learner_id}",
        status_code=204,
        tags=["v0 (deprecated)"],
        summary="Forget a learner",
        deprecated=True,
        responses={404: {**NOT_FOUND, "description": "No stored state for this learner."}},
    )
    def delete_learner(learner_id: LearnerId):
        """Remove everything stored for a learner, so the next call starts from a new pupil.

        Returns 404 when the learner has no stored state, that is, has never posted an attempt
        or was already deleted.
        """
        with LOCK:
            if STATES.pop(learner_id, None) is None:
                raise HTTPException(404, f"unknown learner {learner_id}")


# --------------------------------------------------------------------------- the application


class _LazyRuntime:
    """Builds the runtime from the environment on the first v1 request, not at import time."""

    def __init__(self, rt: _runtime.Runtime | None) -> None:
        self._rt = rt
        self._lock = threading.Lock()

    def get(self) -> _runtime.Runtime:
        if self._rt is None:
            with self._lock:
                if self._rt is None:
                    self._rt = _runtime.from_env()
        return self._rt


def create_app(rt: _runtime.Runtime | None = None, *, v0: bool = True) -> FastAPI:
    """An application serving v1 (and the deprecated v0 routes when `v0` is true).

    `rt` is the runtime to serve; None reads the environment (`KWAN_API_KEYS`, `KWAN_STORE`,
    `KWAN_BUNDLE`) when the first v1 request arrives."""
    holder = _LazyRuntime(rt)
    app = FastAPI(
        title="Kwan Learning Path Engine",
        version=__version__,
        summary="Next best skill for each pupil, with a reason, in milliseconds.",
        description=DESCRIPTION,
        openapi_tags=[*_v1.TAGS, {"name": "v0 (deprecated)", "description": "Round 1 routes, in-memory state."}],
    )
    _v1.install_error_handlers(app)
    app.state.runtime = holder

    app.include_router(_v1.build_router(holder.get))

    @app.middleware("http")
    async def _guard_and_log(request: Request, call_next):  # type: ignore[no-untyped-def]
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > _v1.MAX_BODY_BYTES:
            return JSONResponse(
                {"detail": f"request body over {_v1.MAX_BODY_BYTES} bytes", "code": "payload_too_large"},
                status_code=413,
            )
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        path = request.url.path
        successor = v0_successor(path) if v0 else None
        if successor is not None:
            response.headers["Deprecation"] = "true"
            response.headers["Link"] = f'<{successor}>; rel="successor-version"'
        response.headers["X-Request-Id"] = request_id
        if path.startswith("/v1/") or successor is not None:
            _runtime.logger().info(
                "request",
                extra={
                    "event": "request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - t0) * 1000, 2),
                    "deprecated": successor is not None,
                },
            )
        return response

    if v0:
        register_v0(app)
    return app


def runtime_of(app: FastAPI) -> _runtime.Runtime:
    """The runtime an application created by `create_app` serves (built on first use)."""
    holder: _LazyRuntime = app.state.runtime
    return holder.get()


app = create_app()
