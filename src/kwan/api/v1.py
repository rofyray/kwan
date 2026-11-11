"""API v1 (PRD 6.4, 9.3): the routes, their request and response models, and the error shape.

    GET    /v1/health                            liveness, versions, graph size
    POST   /v1/attempts                          record one answer (idempotent with a key)
    GET    /v1/learners/{id}/recommendations     the next skills for a learner
    GET    /v1/learners/{id}/mastery             estimated mastery of every skill
    DELETE /v1/learners/{id}                     forget a learner

Every route but `/v1/health` needs the `X-API-Key` header when `KWAN_API_KEYS` is set. Errors are
`{"detail": "<what went wrong>", "code": "<machine-readable code>"}`; validation errors keep the
FastAPI shape (`detail` is a list). The routes are built on `kwan.Kwan` and `kwan.store` only.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from kwan._version import __version__
from kwan.api.runtime import Runtime
from kwan.store import VersionConflict, apply_attempt

LEARNER_ID_MAX = 128
INDICATOR_ID_MAX = 64
ITEM_ID_MAX = 128
IDEMPOTENCY_KEY_MAX = 128
K_MAX = 50
LANG_MAX = 8
MAX_BODY_BYTES = 64 * 1024  # the largest request body accepted (an attempt is a few hundred bytes)

TAGS = [
    {"name": "v1: service", "description": "Liveness, versions and graph size."},
    {"name": "v1: learning", "description": "Record answers, get recommendations, read mastery."},
    {"name": "v1: learners", "description": "Learner lifecycle."},
]

__all__ = ["ApiError", "MAX_BODY_BYTES", "TAGS", "build_router", "install_error_handlers"]


# ----------------------------------------------------------------------------- errors


class ApiError(HTTPException):
    """An HTTP error with a machine-readable `code` next to the message."""

    def __init__(self, status_code: int, code: str, detail: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code


class ErrorOut(BaseModel):
    detail: str = Field(description="What went wrong, in plain words.")
    code: str = Field(
        description="A stable code for programs: unknown_indicator, unknown_goal, unknown_learner, "
        "unauthorized, version_conflict, payload_too_large, invalid_time."
    )


def install_error_handlers(app: FastAPI) -> None:
    """Make every `ApiError` raised under this app answer with the v1 error shape."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail, "code": exc.code}, status_code=exc.status_code, headers=exc.headers)


ERR = {"model": ErrorOut}


# ----------------------------------------------------------------------------- models


def _check_time(value: str | None) -> str | None:
    """An ISO 8601 time kept as text (naive values are read as UTC by the engine)."""
    if value is None or value == "":
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("must be an ISO 8601 time such as 2026-11-10T09:30:00+00:00") from None
    return value


class AttemptIn(BaseModel):
    """One answer event from a pupil."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "learner_id": "pupil-0042",
                    "indicator": "math:B1.1.1.1.1",
                    "correct": True,
                    "item_id": "q-17",
                    "at": "2026-11-10T09:30:00+00:00",
                    "idempotency_key": "attempt-7f3a",
                }
            ]
        },
    )

    learner_id: str = Field(
        min_length=1, max_length=LEARNER_ID_MAX, pattern=r"\S", description="Your opaque id for the pupil. No names."
    )
    indicator: str = Field(
        min_length=1,
        max_length=INDICATOR_ID_MAX,
        description="The skill practised: a canonical id (math:B1.1.1.1.1) or a bare code (B1.1.1.1.1).",
    )
    correct: StrictBool = Field(description="Whether the answer was right. Must be a JSON boolean.")
    item_id: str | None = Field(
        default=None, max_length=ITEM_ID_MAX, description="Optional id of the question, for item difficulty."
    )
    at: str | None = Field(
        default=None,
        max_length=40,
        description="When the answer was given (ISO 8601). Send it so forgetting and review can work.",
    )
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=IDEMPOTENCY_KEY_MAX,
        pattern=r"\S",
        description="A key unique to this attempt. A retry with the same key returns the first response "
        "and is not applied twice. The `Idempotency-Key` header works too.",
    )

    @field_validator("at")
    @classmethod
    def _at(cls, value: str | None) -> str | None:
        return _check_time(value)


class AttemptOut(BaseModel):
    learner_id: str
    indicator: str = Field(description="The canonical id of the skill that was updated.")
    predicted_p_correct: float = Field(ge=0, le=1, description="Probability predicted before this answer (4 dp).")
    new_mastery: float = Field(ge=0, le=1, description="Mastery of the skill after this answer (4 dp).")
    version: int = Field(ge=1, description="Version of the stored learner state after this write.")
    bundle_version: str


class RecommendationOut(BaseModel):
    indicator: str = Field(description="Canonical id of the recommended skill.")
    code: str = Field(description="The bare curriculum code.")
    score: float = Field(description="Ranking score; higher is better. Comparable only within one response.")
    mastery: float = Field(ge=0, le=1, description="The learner's current mastery of this skill (3 dp).")
    grade: int = Field(ge=1, le=6, description="Class level, 1 to 6 for B1 to B6.")
    strand: str = Field(description="Curriculum strand name.")
    description: str = Field(description="The skill text (short label in a labels bundle).")
    short_label: str = Field(description="Kwan's short label for the skill.")
    rationale: str = Field(description="Plain-language reason, built only from the signals behind the score.")
    rationale_parts: list[dict[str, Any]] = Field(description="The reason as `{key, params}` parts.")
    ghana_context: str = Field(description="An everyday Ghanaian setting to practise the skill in.")
    is_review: bool = Field(description="True for the review slot: a skill learned before that is fading.")


class RecommendationsOut(BaseModel):
    learner_id: str
    recommendations: list[RecommendationOut] = Field(
        description="Best first. Fewer than k, or none, when fewer skills are ready."
    )
    latency_ms: float = Field(ge=0, description="Engine time for this call in milliseconds.")
    bundle_version: str


class MasteryOut(BaseModel):
    learner_id: str
    mastery: dict[str, float] = Field(description="Canonical indicator id to estimated mastery, 0 to 1 (4 dp).")
    bundle_version: str


class HealthOut(BaseModel):
    ok: bool
    version: str = Field(description="The kwan package version.")
    bundle_version: str
    subject: str
    nodes: int
    edges: int
    store: str = Field(description="`memory` or `sqlite`.")
    auth: bool = Field(description="True when API keys are configured.")


LearnerId = Annotated[
    str,
    Path(min_length=1, max_length=LEARNER_ID_MAX, pattern=r"\S", description="Your opaque id for the pupil."),
]


# ----------------------------------------------------------------------------- the router


def build_router(runtime: Runtime | Callable[[], Runtime]) -> APIRouter:
    """The v1 routes bound to one runtime (engine, store, keys).

    `runtime` is the runtime itself or a function that returns it, called on every request, so an
    application can build the runtime from the environment when the first request arrives."""
    current: Callable[[], Runtime] = runtime if callable(runtime) else (lambda: runtime)
    router = APIRouter(prefix="/v1")

    def require_key(x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> None:
        if not current().key_ok(x_api_key):
            raise ApiError(401, "unauthorized", "missing or unknown API key (send it in the X-API-Key header)")

    auth = [Depends(require_key)]

    def resolve(indicator: str, what: str) -> str:
        try:
            return current().engine.resolve(indicator)
        except KeyError:
            raise ApiError(404, f"unknown_{what}", f"unknown {what} {indicator!r}") from None

    @router.get("/health", response_model=HealthOut, tags=["v1: service"], summary="Liveness and versions")
    def health() -> dict[str, Any]:
        rt = current()
        engine = rt.engine
        G = engine.graph
        return {
            "ok": True,
            "version": __version__,
            "bundle_version": engine.bundle_version,
            "subject": engine.subject,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "store": rt.store_kind,
            "auth": rt.auth_enabled,
        }

    @router.post(
        "/attempts",
        response_model=AttemptOut,
        tags=["v1: learning"],
        summary="Record one answer",
        dependencies=auth,
        responses={401: ERR, 404: {**ERR, "description": "Unknown indicator."}, 409: ERR, 413: ERR},
    )
    def attempts(
        a: AttemptIn,
        response: Response,
        idempotency_header: Annotated[
            str | None, Header(alias="Idempotency-Key", max_length=IDEMPOTENCY_KEY_MAX)
        ] = None,
    ) -> dict[str, Any]:
        """Apply one answer to the learner's stored state and return the new mastery of that skill.

        The learner is created on first use. With an idempotency key (in the body or in the
        `Idempotency-Key` header) a retry returns the first response and the answer is applied once;
        the retry carries the header `Idempotent-Replayed: true`. Writes to one learner are serialised
        with optimistic versioning, so concurrent requests cannot lose an answer.
        """
        rt = current()
        engine, store = rt.engine, rt.store
        node = resolve(a.indicator, "indicator")
        key = a.idempotency_key or idempotency_header or None

        def apply(state: Any) -> dict[str, Any]:
            # The version this write will produce: the stored one plus 1. If another writer gets in
            # first, `apply_attempt` reads again and calls `apply` again, so the number stays right.
            found = store.get_with_version(a.learner_id)
            version = (found[1] if found is not None else 0) + 1
            p = engine.update(state, node, a.correct, a.item_id, a.at)
            m = engine.mastery(state, node, now=a.at)
            return {
                "learner_id": a.learner_id,
                "indicator": node,
                "predicted_p_correct": round(p, 4),
                "new_mastery": round(m, 4),
                "version": version,
                "bundle_version": engine.bundle_version,
            }

        try:
            outcome = apply_attempt(store, a.learner_id, key, apply, new_state=engine.new_state)
        except VersionConflict as exc:
            raise ApiError(409, "version_conflict", f"{exc}; retry the request") from None
        result = dict(outcome.result)
        if outcome.replayed:
            response.headers["Idempotent-Replayed"] = "true"
        rt.log.info(
            "attempt",
            extra={"event": "attempt", "learner": a.learner_id, "indicator": node, "replayed": outcome.replayed},
        )
        return result

    @router.get(
        "/learners/{learner_id}/recommendations",
        response_model=RecommendationsOut,
        tags=["v1: learning"],
        summary="Next best skills for a learner",
        dependencies=auth,
        responses={401: ERR, 404: {**ERR, "description": "Unknown goal."}, 422: {"description": "Bad argument."}},
    )
    def recommendations(
        learner_id: LearnerId,
        k: Annotated[int, Query(ge=1, le=K_MAX, description="How many skills to return at most.")] = 3,
        goal: Annotated[
            str | None, Query(max_length=INDICATOR_ID_MAX, description="A skill to work towards (id or bare code).")
        ] = None,
        grade_max: Annotated[int | None, Query(ge=1, le=6, description="Class cap: nothing above this grade.")] = None,
        goal_mode: Annotated[
            Literal["strict", "soft"],
            Query(description="strict: skills on the path to the goal come first. soft: the goal is one signal."),
        ] = "strict",
        now: Annotated[
            str | None,
            Query(max_length=40, description="The time of the request (ISO 8601), for forgetting and review."),
        ] = None,
        lang: Annotated[str, Query(min_length=2, max_length=LANG_MAX, description="Language of the reason.")] = "en",
        include_drafts: Annotated[bool, Query(description="Allow practice settings no teacher has reviewed.")] = False,
    ) -> dict[str, Any]:
        """Skills the learner is ready for and has not mastered, best first. Reads never change stored state;
        a learner never seen gets the starting recommendations of a new pupil."""
        rt = current()
        engine, store = rt.engine, rt.store
        goal_id = resolve(goal, "goal") if goal else None
        try:
            now_text = _check_time(now)
        except ValueError as exc:
            raise ApiError(422, "invalid_time", f"now: {exc}") from None
        state = store.get(learner_id) or engine.new_state(learner_id)
        recs = engine.recommend(
            state,
            k=k,
            goal=goal_id,
            grade_max=grade_max,
            goal_mode=goal_mode,
            now=now_text,
            lang=lang,
            include_drafts=include_drafts,
        )
        return {
            "learner_id": learner_id,
            "recommendations": [r.to_dict() for r in recs],
            "latency_ms": engine.last_latency_ms,
            "bundle_version": engine.bundle_version,
        }

    @router.get(
        "/learners/{learner_id}/mastery",
        response_model=MasteryOut,
        tags=["v1: learning"],
        summary="Mastery of every skill for a learner",
        dependencies=auth,
        responses={401: ERR},
    )
    def mastery(
        learner_id: LearnerId,
        now: Annotated[str | None, Query(max_length=40, description="The time of the request (ISO 8601).")] = None,
    ) -> dict[str, Any]:
        """Every indicator id with the learner's estimated mastery. A learner never seen gets the
        starting estimates. Reads never change stored state."""
        rt = current()
        engine, store = rt.engine, rt.store
        try:
            now_text = _check_time(now)
        except ValueError as exc:
            raise ApiError(422, "invalid_time", f"now: {exc}") from None
        state = store.get(learner_id) or engine.new_state(learner_id)
        table = engine.mastery_map(state, now=now_text)
        return {
            "learner_id": learner_id,
            "mastery": {n: round(v, 4) for n, v in table.items()},
            "bundle_version": engine.bundle_version,
        }

    @router.delete(
        "/learners/{learner_id}",
        status_code=204,
        tags=["v1: learners"],
        summary="Forget a learner",
        dependencies=auth,
        responses={401: ERR, 404: {**ERR, "description": "No stored state for this learner."}},
    )
    def delete_learner(learner_id: LearnerId) -> Response:
        """Remove everything stored for a learner (state and idempotency records)."""
        rt = current()
        if not rt.store.delete(learner_id):
            raise ApiError(404, "unknown_learner", f"no stored state for learner {learner_id!r}")
        rt.log.info("learner deleted", extra={"event": "delete", "learner": learner_id})
        return Response(status_code=204)

    return router
