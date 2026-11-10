"""Kwan REST API (FastAPI).

Run from the repo root:   uvicorn api:app --port 8000
Run from anywhere:        uvicorn --app-dir /path/to/kwan api:app --port 8000
Interactive docs:         http://localhost:8000/docs

Contract: PRD Section 5.4. Learner state is held in memory (Round 1 has no persistence), so it is
lost on restart and is not shared between worker processes: run a single worker.
"""
import threading
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StrictBool

import engine
import graph

G = graph.load()                                     # committed data/graph.json, independent of the working directory
L = engine.EloLearner(G=G)
R = engine.Recommender(G, L)
STATES: dict[str, engine.LearnerState] = {}          # in-memory; swap for SQLite/Redis in v2
LOCK = threading.RLock()                             # sync endpoints run in a thread pool: one writer at a time

LEARNER_ID_MAX = 128
INDICATOR_ID_MAX = 64
K_MAX = 50

DESCRIPTION = """
Adaptive learning path engine for Ghanaian primary maths (NaCCA B1 to B6).

Typical flow for a platform such as Syto:

1. `POST /attempt` each time a pupil answers a question. The reply carries the probability the
   engine predicted before seeing the answer and the pupil's updated mastery of that skill.
2. `GET /recommend/{learner_id}` to get the next best skills, each with a plain-language reason
   and a practice setting from everyday Ghanaian life.
3. `GET /mastery/{learner_id}` to draw the mastery map for a teacher.

**New learners.** A `learner_id` needs no registration. A learner who has never posted an attempt is
treated as a brand-new pupil: `GET /recommend` returns the starting skills and `GET /mastery` returns
the starting estimates. Reads never create or change stored state. A learner is stored on the first
`POST /attempt` and removed by `DELETE /learner/{learner_id}`.

**Storage.** Learner state lives in memory only. It is lost when the server restarts.

**Errors.** An unknown indicator or goal gives 404 with a message in `detail`. A request that fails
validation gives 422 with the offending field in `detail`.
"""

app = FastAPI(
    title="Kwan Learning Path Engine",
    version="0.1.0",
    summary="Next best maths skill for each pupil, with a reason, in milliseconds.",
    description=DESCRIPTION,
    openapi_tags=[
        {"name": "service", "description": "Liveness and graph size."},
        {"name": "learning", "description": "Record answers, read mastery, get recommendations."},
        {"name": "learners", "description": "Learner lifecycle."},
    ],
)

# --------------------------------------------------------------------------- models

class Attempt(BaseModel):
    """One answer event from a pupil."""
    model_config = ConfigDict(json_schema_extra={"examples": [
        {"learner_id": "a", "indicator": "B1.1.1.1.1", "correct": True, "item_id": None}]})

    learner_id: str = Field(min_length=1, max_length=LEARNER_ID_MAX, pattern=r"\S",
                            description="Your identifier for the pupil. Any non-blank string, no personal data.")
    indicator: str = Field(min_length=1, max_length=INDICATOR_ID_MAX,
                           description="NaCCA indicator id of the skill that was practised, e.g. B1.1.1.1.1.")
    correct: StrictBool = Field(description="Whether the answer was right. Must be a JSON boolean.")
    item_id: str | None = Field(default=None, max_length=128,
                                description="Optional id of the question. Used for item difficulty when the engine "
                                            "has it; otherwise the item is treated as typical for the skill.")

class AttemptResponse(BaseModel):
    predicted_p_correct: float = Field(ge=0, le=1, description="Probability of a correct answer that the engine "
                                       "predicted before seeing this attempt (4 decimals).")
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
    recommendations: list[Recommendation] = Field(description="Best first. May hold fewer than k items, or none, "
                                                  "when fewer skills are ready for this learner.")
    latency_ms: float = Field(ge=0, description="Engine compute time for this call in milliseconds, "
                              "excluding HTTP and JSON overhead.")

class ErrorResponse(BaseModel):
    detail: str = Field(description="What went wrong.")

LearnerId = Annotated[str, Path(min_length=1, max_length=LEARNER_ID_MAX, pattern=r"\S",
                                description="Your identifier for the pupil. Any non-blank string.")]

NOT_FOUND = {"model": ErrorResponse}

# --------------------------------------------------------------------------- state

def state(lid: str) -> engine.LearnerState:
    """Stored state for a learner, created on first use. Call with LOCK held."""
    return STATES.setdefault(lid, engine.LearnerState(lid))

def peek(lid: str) -> engine.LearnerState:
    """State to read from: the stored one, or a throwaway fresh one for a learner never seen. Call with LOCK held."""
    return STATES.get(lid) or engine.LearnerState(lid)

# --------------------------------------------------------------------------- endpoints

@app.get("/health", response_model=HealthResponse, tags=["service"], summary="Liveness check and graph size")
def health():
    return {"ok": True, "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}

@app.post("/attempt", response_model=AttemptResponse, tags=["learning"], summary="Record one answer",
          responses={404: {**NOT_FOUND, "description": "Unknown indicator."}})
def attempt(a: Attempt):
    """Update the learner model with one answer and return the new mastery of that skill.

    The learner is created on first use. Updates for the same learner are applied one at a time,
    so concurrent requests cannot corrupt the state.
    """
    if a.indicator not in G:
        raise HTTPException(404, f"unknown indicator {a.indicator}")
    with LOCK:
        st = state(a.learner_id)
        p = L.update(st, a.indicator, a.correct, a.item_id)
        m = L.mastery(st, a.indicator)
    return {"predicted_p_correct": round(p, 4), "new_mastery": round(m, 4)}

@app.get("/recommend/{learner_id}", response_model=RecommendResponse, tags=["learning"],
         summary="Next best skills for a learner", responses={404: {**NOT_FOUND, "description": "Unknown goal."}})
def recommend(
    learner_id: LearnerId,
    k: Annotated[int, Query(ge=1, le=K_MAX, description="How many skills to return at most.")] = 3,
    goal: Annotated[str | None, Query(max_length=INDICATOR_ID_MAX, description="Indicator id of a target skill. "
                    "Skills on the path to it are ranked higher. Leave out or send empty for no goal.")] = None,
    grade_max: Annotated[int | None, Query(ge=1, le=6, description="Class cap: nothing above this class level "
                         "is recommended (4 means nothing above B4).")] = None,
):
    """Skills the learner is ready for and has not mastered, best first.

    A learner who has never posted an attempt gets the starting recommendations for a new pupil.
    This call does not change any stored state.
    """
    goal = goal or None
    if goal is not None and goal not in G:
        raise HTTPException(404, f"unknown goal {goal}")
    with LOCK:
        return R.recommend(peek(learner_id), k=k, goal=goal, grade_max=grade_max)

@app.get("/mastery/{learner_id}", response_model=dict[str, float], tags=["learning"],
         summary="Mastery of every skill for a learner")
def mastery(learner_id: LearnerId):
    """Map of every indicator id in the graph to the learner's mastery, 0 to 1 (3 decimals).

    A learner who has never posted an attempt gets the starting estimates for a new pupil.
    This call does not change any stored state.
    """
    with LOCK:
        st = peek(learner_id)
        return {n: round(L.mastery(st, n), 3) for n in G}

@app.delete("/learner/{learner_id}", status_code=204, tags=["learners"], summary="Forget a learner",
            responses={404: {**NOT_FOUND, "description": "No stored state for this learner."}})
def delete_learner(learner_id: LearnerId):
    """Remove everything stored for a learner, so the next call starts from a new pupil.

    Returns 404 when the learner has no stored state, that is, has never posted an attempt
    or was already deleted.
    """
    with LOCK:
        if STATES.pop(learner_id, None) is None:
            raise HTTPException(404, f"unknown learner {learner_id}")
