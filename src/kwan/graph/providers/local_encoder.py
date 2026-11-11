"""`local-encoder`: an open-weight decision model that runs on this machine, with no key and no fee.

Design
------
1. **Encoder.** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 22M parameters) embeds the
   text of each skill. The candidate stage already does this, so the similarities reach this
   provider as `PairQuestion.signals`; when a question carries no signals the provider embeds the
   two texts itself.
2. **Pair features.** Eleven numbers that mean the same thing for any list of skills, so that what
   is learned on one curriculum can be used on another: how similar A and B are (`sim`), how that
   similarity stands among B's other possible sources (`z`, `log_rank`, `margin`) and among A's
   other neighbours (`z_rev`), whether A is the more "central" skill (`central`), word overlap
   (`jaccard`), how much of A's wording is inside B and of B's inside A (`cont_ab`, `cont_ba`),
   which text is longer (`len_ratio`) and which mentions the larger number (`num_order`). No feature
   is a raw embedding dimension, because those would memorise the vocabulary of the fitting data.
3. **Classifier and calibration.** A logistic regression on the standardised features, then a
   two-parameter Platt step fitted on out-of-fold scores. The output is the probability that the
   answer to "Must a pupil be able to do A before they can learn B?" is yes.

The classifier is fitted on the training split of Junyi (expert prerequisite links), by
`kwan.graph.junyi`. Junyi is licensed for non-commercial use, so the fitted numbers are an artefact
under `build/augment/` and are never committed or shipped. Without that file this provider cannot
answer, and says so.

What it does not know: grades. Junyi has no grades, so the classifier cannot learn from them; the
grade window is applied by the candidate stage instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kwan._paths import data_dir
from kwan.graph.augment import PairQuestion
from kwan.graph.candidates import MINILM, MINILM_LICENCE, Embedder

FEATURES = (
    "sim",
    "z",
    "log_rank",
    "margin",
    "z_rev",
    "central",
    "jaccard",
    "cont_ab",
    "cont_ba",
    "len_ratio",
    "num_order",
)
ARTIFACT_VERSION = 1

__all__ = [
    "FEATURES",
    "LocalEncoderDecisionModel",
    "default_artifact_path",
    "fit_artifact",
    "pair_features",
    "predict",
]

_WORD = re.compile(r"[a-z]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_STOP = frozenset(
    "a an and are as at be by for from in into is it its of on or that the their them this to up use using "
    "with within e g i etc such how what when which who can will".split()
)


def default_artifact_path() -> Path:
    """`build/augment/local_encoder.json`: the fitted classifier (git-ignored)."""
    return data_dir().parent / "build" / "augment" / "local_encoder.json"


def _words(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 3 and w.endswith("es"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.add(w)
    return out


def _largest_number(text: str) -> float | None:
    values = []
    for m in _NUMBER.findall(text):
        try:
            values.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return max(values) if values else None


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    return sum(x * y for x, y in zip(u, v)) / (nu * nv) if nu > 0 and nv > 0 else 0.0


def pair_features(q: PairQuestion, sim: float | None = None) -> list[float]:
    """The feature vector of a question, in the order of `FEATURES`.

    `sim` is used when the question carries no `sim` signal. The signal features default to neutral
    values (0) when the candidate stage did not supply them.
    """
    s = dict(q.signals)
    wa, wb = _words(q.a_text), _words(q.b_text)
    both = len(wa & wb)
    union = len(wa | wb)
    na, nb = _largest_number(q.a_text), _largest_number(q.b_text)
    num_order = 0.0 if na is None or nb is None or na == nb else (1.0 if nb > na else -1.0)
    return [
        float(s.get("sim", sim if sim is not None else 0.0)),
        float(s.get("z", 0.0)),
        math.log(max(float(s.get("rank", 1.0)), 1.0)),
        float(s.get("margin", 0.0)),
        float(s.get("z_rev", 0.0)),
        float(s.get("central", 0.0)),
        both / union if union else 0.0,
        both / len(wa) if wa else 0.0,
        both / len(wb) if wb else 0.0,
        math.log((1 + len(wb)) / (1 + len(wa))),
        num_order,
    ]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def predict(artifact: Mapping[str, Any], features: Sequence[float]) -> float:
    """The calibrated probability for one feature vector, in pure Python."""
    names = list(artifact["features"])
    by_name = dict(zip(FEATURES, features))
    score = float(artifact["intercept"])
    for name, mean, scale, coef in zip(names, artifact["mean"], artifact["scale"], artifact["coef"]):
        score += coef * ((by_name[name] - mean) / (scale or 1.0))
    platt = artifact.get("platt") or {"a": 1.0, "b": 0.0}
    return _sigmoid(float(platt["a"]) * score + float(platt["b"]))


def fit_artifact(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    groups: Sequence[Any],
    features: Sequence[str] = FEATURES,
    folds: int = 5,
    C: float = 1.0,
    seed: int = 0,
) -> tuple[dict[str, Any], list[float]]:
    """Fit the classifier and its Platt step. Returns the artefact and the out-of-fold probabilities.

    `X` rows are full `FEATURES` vectors; `features` picks the columns that are used. `groups` keeps
    all pairs of one target in the same fold. The out-of-fold probabilities are already calibrated
    and are what a threshold should be chosen on. Needs scikit-learn and numpy.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    cols = [FEATURES.index(f) for f in features]
    A = np.asarray(X, dtype=float)[:, cols]
    t = np.asarray(y, dtype=int)
    g = np.asarray([str(x) for x in groups])

    def fit(rows: Any) -> tuple[Any, Any, Any]:
        mean = A[rows].mean(axis=0)
        scale = A[rows].std(axis=0)
        scale[scale == 0] = 1.0
        clf = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        clf.fit((A[rows] - mean) / scale, t[rows])
        return mean, scale, clf

    oof = np.zeros(len(t))
    order = np.argsort(np.asarray([hashlib.sha256(x.encode()).hexdigest() for x in g]), kind="stable")
    for train, test in GroupKFold(n_splits=folds).split(A[order], t[order], g[order]):
        mean, scale, clf = fit(order[train])
        oof[order[test]] = clf.decision_function((A[order[test]] - mean) / scale)
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(oof.reshape(-1, 1), t)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    mean, scale, clf = fit(np.arange(len(t)))
    artifact: dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "provider": "local-encoder",
        "encoder": MINILM,
        "encoder_licence": MINILM_LICENCE,
        "features": list(features),
        "mean": [round(float(x), 8) for x in mean],
        "scale": [round(float(x), 8) for x in scale],
        "coef": [round(float(x), 8) for x in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 8),
        "platt": {"a": round(a, 8), "b": round(b, 8)},
        "C": C,
        "folds": folds,
        "n_pairs": int(len(t)),
        "n_positive": int(t.sum()),
    }
    oof_p = [_sigmoid(a * float(s) + b) for s in oof]
    return artifact, oof_p


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    """Eight hex characters that change whenever the fitted numbers change (part of the cache key)."""
    core = {k: artifact[k] for k in ("features", "mean", "scale", "coef", "intercept", "platt") if k in artifact}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()[:8]


class LocalEncoderDecisionModel:
    """MiniLM similarity signals and wording features into a calibrated logistic classifier.

    `artifact` is the fitted classifier (a mapping, or a path to its JSON file; the default is
    `build/augment/local_encoder.json`). `embedder` is used only for questions without signals.
    """

    name = "local-encoder"
    licence = f"{MINILM_LICENCE} (encoder {MINILM}); classifier fitted on Junyi (non-commercial), kept under build/"
    last_cost_usd = 0.0

    def __init__(
        self,
        artifact: Mapping[str, Any] | str | os.PathLike[str] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if artifact is None or isinstance(artifact, str | os.PathLike):
            path = Path(artifact) if artifact is not None else default_artifact_path()
            if not path.is_file():
                raise FileNotFoundError(
                    f"local-encoder has no fitted classifier at {path}. Fit it on Junyi first: "
                    "`kwan graph augment --junyi raw/junyi` (the file stays under build/ and is never committed)."
                )
            with open(path, encoding="utf-8") as f:
                artifact = json.load(f)
        assert isinstance(artifact, Mapping)
        unknown = [f for f in artifact["features"] if f not in FEATURES]
        if unknown:
            raise ValueError(f"local-encoder artefact uses unknown features: {unknown}")
        self.artifact = dict(artifact)
        self.model = f"{str(artifact.get('encoder', MINILM)).rsplit('/', 1)[-1]}+logreg@{artifact_digest(artifact)}"
        self.threshold = artifact.get("threshold")
        self._embedder = embedder
        self._vectors: dict[str, list[float]] = {}

    def prompt(self, q: PairQuestion) -> str:
        """The exact input of the classifier: both texts and the candidate-stage signals."""
        return json.dumps({"a": q.a_text, "b": q.b_text, "signals": list(q.signals)}, ensure_ascii=False)

    def _vector(self, text: str) -> list[float]:
        if text not in self._vectors:
            if self._embedder is None:
                from kwan.graph.candidates import MiniLMEmbedder

                self._embedder = MiniLMEmbedder(str(self.artifact.get("encoder", MINILM)))
            self._vectors[text] = self._embedder.encode([text])[0]
        return self._vectors[text]

    def prerequisite_probability(self, q: PairQuestion) -> float:
        sim = None
        if "sim" not in dict(q.signals):
            sim = _cosine(self._vector(q.a_text), self._vector(q.b_text))
        return predict(self.artifact, pair_features(q, sim))
