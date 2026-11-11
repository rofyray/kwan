"""Phase 7 providers. The HTTP providers are tested with a mock transport only: nothing here touches a network."""

import json
import math

import pytest

from kwan.graph import augment
from kwan.graph.candidates import HashingEmbedder
from kwan.graph.providers import get_provider, llm_json, typesafe_jev
from kwan.graph.providers import local_encoder as le

httpx = pytest.importorskip("httpx")


def _q(**kw):
    base = dict(a_id="math:B1.1.1.1.1", a_text="Count objects to 20", a_grade=1)
    base.update(b_id="math:B2.1.2.1.1", b_text="Add numbers to 100 by counting objects", b_grade=2)
    return augment.PairQuestion(**{**base, **kw})


# ---------------------------------------------------------------- llm-json


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"probability": 0.82}', 0.82),
        ('Here is my answer:\n```json\n{"probability": 0.4}\n```', 0.4),
        ('{"probability": "0.7"}', 0.7),
        ('{"probability": "70%"}', 0.7),
        ('{"Probability": 1}', 1.0),
        ('{"reason": "counting comes first", "probability": 0.9}', 0.9),
        ("probability: 0.35", 0.35),
        ("Probability = 35%", 0.35),
        ("{'probability': 0.2}", 0.2),
        ('{"probability": 0}', 0.0),
    ],
)
def test_parse_probability_reads_the_usual_shapes(text, expected):
    assert llm_json.parse_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "I think yes.", '{"probability": 1.7}', '{"probability": -0.1}', '{"probability": null}', '{"p": 0.5}',
     '{"probability": true}', '{"probability": "high"}'],
)  # fmt: skip
def test_parse_probability_never_guesses(text):
    with pytest.raises(ValueError):
        llm_json.parse_probability(text)


def _chat(content, status=200):
    return httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def _llm(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    args = dict(endpoint="https://llm.test/v1/chat/completions", model="some-model", api_key="k-test", client=client)
    return llm_json.LLMJsonDecisionModel(**{**args, "sleep": lambda s: None, **kw})


def test_llm_json_sends_the_prompt_and_reads_the_answer():
    seen = []

    def handler(request):
        seen.append(request)
        return _chat('{"probability": 0.75}')

    model = _llm(handler, cost_per_call_usd=0.001)
    assert model.prerequisite_probability(_q()) == 0.75
    (request,) = seen
    body = json.loads(request.content)
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k-test"
    assert body["model"] == "some-model" and body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": augment.render_prompt(_q())}]
    assert model.last_cost_usd == 0.001 and model.name == "llm-json"


def test_llm_json_sends_no_auth_header_without_a_key(monkeypatch):
    monkeypatch.delenv("KWAN_LLM_API_KEY", raising=False)
    seen = []
    model = _llm(lambda r: (seen.append(r), _chat('{"probability": 0.5}'))[1], api_key="")
    model.prerequisite_probability(_q())
    assert "authorization" not in seen[0].headers


def test_llm_json_retries_a_busy_service_and_a_bad_answer_then_gives_up():
    answers = iter([httpx.Response(503), _chat("no number here"), _chat('{"probability": 0.6}')])
    model = _llm(lambda r: next(answers), retries=2)
    assert model.prerequisite_probability(_q()) == 0.6

    always_bad = _llm(lambda r: _chat("still no number"), retries=1)
    with pytest.raises(ValueError, match="could not read a probability"):
        always_bad.prerequisite_probability(_q())
    with pytest.raises(RuntimeError, match="HTTP 401"):
        _llm(lambda r: httpx.Response(401)).prerequisite_probability(_q())


def test_llm_json_applies_a_calibration_function():
    model = _llm(lambda r: _chat('{"probability": 0.8}'), calibrate=lambda p: p / 2)
    assert model.prerequisite_probability(_q()) == pytest.approx(0.4)


def test_llm_json_reads_its_settings_from_the_environment(monkeypatch):
    for name in ("KWAN_LLM_ENDPOINT", "KWAN_LLM_MODEL", "KWAN_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="KWAN_LLM_ENDPOINT"):
        llm_json.LLMJsonDecisionModel()
    monkeypatch.setenv("KWAN_LLM_ENDPOINT", "http://localhost:11434/v1/chat/completions")
    monkeypatch.setenv("KWAN_LLM_MODEL", "local-model")
    model = get_provider("llm-json")
    assert (model.endpoint, model.model, model.api_key) == (
        "http://localhost:11434/v1/chat/completions",
        "local-model",
        "",
    )


def test_llm_json_docstring_says_it_was_not_run_live():
    assert "NOT RUN AGAINST A LIVE SERVICE" in llm_json.__doc__


def test_llm_json_answers_are_cached(tmp_path):
    calls = []
    model = _llm(lambda r: (calls.append(1), _chat('{"probability": 0.9}'))[1], cost_per_call_usd=0.002)
    cached = augment.CachedDecisionModel(model, tmp_path / "cache.jsonl")
    assert [cached.prerequisite_probability(_q()) for _ in range(3)] == [0.9, 0.9, 0.9]
    assert len(calls) == 1 and cached.spend()["cost_usd"] == pytest.approx(0.002)


# ---------------------------------------------------------------- typesafe-jev


def test_jev_constructor_raises_a_clear_error_when_it_is_not_set_up(monkeypatch):
    for name in ("KWAN_JEV_ENDPOINT", "KWAN_JEV_API_KEY", "KWAN_JEV_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as err:
        get_provider("typesafe-jev")
    message = str(err.value)
    assert "KWAN_JEV_ENDPOINT" in message and "not available" in message and "typesafe_jev.py" in message
    assert "NOT RUN AGAINST A LIVE SERVICE" in typesafe_jev.__doc__


def test_jev_follows_the_documented_gateway_contract_with_a_mock_transport():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "jev-test",
                "answers": {"prerequisite": {"type": "noul", "noul": 0.91}},
                "usage": {"input_tokens": 100, "output_tokens": 10},
                "meta": {"usd_spent": 0.00002},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = typesafe_jev.TypesafeJevDecisionModel(
        endpoint="https://gateway.test/v1/decisions", api_key="k", client=client
    )
    assert model.prerequisite_probability(_q()) == 0.91
    body = json.loads(seen[0].content)
    assert seen[0].headers["authorization"] == "Bearer k"
    assert body["model"] == "typesafe/jev"
    assert body["questions"] == {"prerequisite": {"type": "noul", "instructions": augment.QUESTION}}
    assert "Count objects to 20" in body["state"] and "Answer with JSON" not in body["state"]
    assert model.last_cost_usd == pytest.approx(0.00002)


@pytest.mark.parametrize("payload", [{}, {"answers": {}}, {"answers": {"prerequisite": {"noul": 1.4}}}])
def test_jev_rejects_a_response_without_a_probability(payload):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    model = typesafe_jev.TypesafeJevDecisionModel(
        endpoint="https://gateway.test/v1/decisions", api_key="k", client=client
    )
    with pytest.raises(ValueError, match="typesafe-jev"):
        model.prerequisite_probability(_q())


# ---------------------------------------------------------------- local-encoder


def _artifact(**kw):
    n = len(le.FEATURES)
    base = {
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "features": list(le.FEATURES),
        "mean": [0.0] * n,
        "scale": [1.0] * n,
        "coef": [0.0] * n,
        "intercept": 0.0,
        "platt": {"a": 1.0, "b": 0.0},
    }
    return {**base, **kw}


def test_pair_features_are_named_and_read_the_candidate_signals():
    signals = (("sim", 0.6), ("rank", 3.0), ("z", 1.5), ("margin", 0.1), ("z_rev", 0.7), ("central", -0.02))
    f = dict(zip(le.FEATURES, le.pair_features(_q(signals=signals))))
    assert f["sim"] == 0.6 and f["z"] == 1.5 and f["log_rank"] == pytest.approx(math.log(3))
    assert f["margin"] == 0.1 and f["z_rev"] == 0.7 and f["central"] == -0.02
    # words: A = {count, object}, B = {add, number, count, object}; numbers: 20 and 100
    assert f["jaccard"] == pytest.approx(2 / 4) and f["cont_ab"] == 1.0 and f["cont_ba"] == pytest.approx(0.5)
    assert f["len_ratio"] == pytest.approx(math.log(5 / 3)) and f["num_order"] == 1.0
    assert le.pair_features(_q(a_text="Add to 1,000", b_text="Add to 100"))[-1] == -1.0
    assert le.pair_features(_q(a_text="Add", b_text="Add to 100"))[-1] == 0.0


def test_predict_is_a_logistic_model_with_a_platt_step():
    coef = [0.0] * len(le.FEATURES)
    coef[0] = 2.0  # sim
    art = _artifact(coef=coef, mean=[0.5] + [0.0] * (len(coef) - 1), intercept=-1.0, platt={"a": 0.5, "b": 0.1})
    features = le.pair_features(_q(signals=(("sim", 0.9),)))
    expected = 1 / (1 + math.exp(-(0.5 * (2.0 * (0.9 - 0.5) - 1.0) + 0.1)))
    assert le.predict(art, features) == pytest.approx(expected)
    model = le.LocalEncoderDecisionModel(art)
    assert model.prerequisite_probability(_q(signals=(("sim", 0.9),))) == pytest.approx(expected)
    assert model.name == "local-encoder" and model.model.startswith("all-MiniLM-L6-v2+logreg@")
    assert "Apache-2.0" in model.licence and model.last_cost_usd == 0.0


def test_local_encoder_embeds_only_when_a_question_has_no_signals():
    class Counting(HashingEmbedder):
        calls = 0

        def encode(self, texts):
            Counting.calls += len(texts)
            return super().encode(texts)

    coef = [0.0] * len(le.FEATURES)
    coef[0] = 1.0
    model = le.LocalEncoderDecisionModel(_artifact(coef=coef), embedder=Counting())
    with_signals = model.prerequisite_probability(_q(signals=(("sim", 0.3),)))
    assert Counting.calls == 0 and with_signals == pytest.approx(1 / (1 + math.exp(-0.3)))
    p = model.prerequisite_probability(_q())
    assert Counting.calls == 2 and 0.0 < p < 1.0
    model.prerequisite_probability(_q())
    assert Counting.calls == 2  # the vectors are remembered


def test_local_encoder_needs_its_fitted_file_and_says_how_to_make_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="--junyi"):
        le.LocalEncoderDecisionModel(tmp_path / "local_encoder.json")
    path = tmp_path / "local_encoder.json"
    path.write_text(json.dumps(_artifact(threshold=0.4)), encoding="utf-8")
    assert le.LocalEncoderDecisionModel(path).threshold == 0.4
    with pytest.raises(ValueError, match="unknown features"):
        le.LocalEncoderDecisionModel(_artifact(features=["sim", "magic"]))


def test_local_encoder_cache_key_changes_with_the_fitted_numbers_and_the_signals(tmp_path):
    one = augment.CachedDecisionModel(le.LocalEncoderDecisionModel(_artifact()), tmp_path / "c.jsonl")
    two = augment.CachedDecisionModel(le.LocalEncoderDecisionModel(_artifact(intercept=0.5)), tmp_path / "c.jsonl")
    q = _q(signals=(("sim", 0.5),))
    assert one.key(q) != two.key(q)
    assert one.key(q) != one.key(_q(signals=(("sim", 0.6),)))
    assert one.key(q) == one.key(_q(signals=(("sim", 0.5),)))


def test_fit_artifact_learns_a_simple_rule_and_returns_calibrated_out_of_fold_scores():
    pytest.importorskip("sklearn")
    import random

    rng = random.Random(0)
    X, y, groups = [], [], []
    for k in range(600):
        row = [rng.random() for _ in le.FEATURES]
        X.append(row)
        y.append(int(row[0] + 0.3 * rng.gauss(0, 1) > 0.8))  # the first feature decides, with noise
        groups.append(f"target-{k % 60}")
    artifact, oof = le.fit_artifact(X, y, groups)
    assert artifact["features"] == list(le.FEATURES) and len(artifact["coef"]) == len(le.FEATURES)
    assert artifact["coef"][0] == max(artifact["coef"]) and artifact["coef"][0] > 0.5
    assert len(oof) == 600 and all(0.0 <= p <= 1.0 for p in oof)
    assert abs(sum(oof) / len(oof) - sum(y) / len(y)) < 0.03  # calibrated on average
    hi = [p for p, lab in zip(oof, y) if lab]
    lo = [p for p, lab in zip(oof, y) if not lab]
    assert sum(hi) / len(hi) > sum(lo) / len(lo) + 0.1
    assert le.predict(artifact, [0.95] + [0.5] * 10) > le.predict(artifact, [0.05] + [0.5] * 10)
    json.dumps(artifact)  # plain numbers only: it can be written as JSON, no pickle

    fewer, _ = le.fit_artifact(X, y, groups, features=("sim", "jaccard"))
    assert fewer["features"] == ["sim", "jaccard"] and len(fewer["coef"]) == 2
    assert 0.0 < le.predict(fewer, X[0]) < 1.0
