"""Decision-model evaluation without network access: a fake API stands in for OpenRouter."""
import io
import json
import math
from urllib.error import HTTPError

import pytest

from examples import decision_models as dm
from probabilitybench.evaluation import openrouter


def fake_response(probability, cost=2e-4):
    return {"model": "fake/decider-20261014", "answers": {"positive": {"type": "noul", "noul": probability}},
            "usage": {"input_tokens": 4000, "output_tokens": 0, "cost": cost}}


def test_requests_hold_memos_and_training_labels_but_no_test_targets():
    rows = list(dm.protocol_requests(limit=2))
    assert len(rows) == len(dm.WORLDS) * len(dm.SEEDS) * 2
    truth = dm.exact_probabilities(limit=2)
    for world, seed, case_id, inputs in rows:
        assert len(inputs["state"]["training_examples"]) == dm.SHOTS
        assert set(inputs["questions"]) == {"positive"}
        assert case_id not in json.dumps(inputs)
        assert "posterior" not in json.dumps(inputs)
        assert (world, seed, case_id) in truth


def test_test_cases_follow_the_training_pool():
    world, train, test = dm.protocol_cases("paper_review", 2032, limit=3)
    hidden = next(level["hidden"] for level in world.hiding_levels if level["name"] == dm.LEVEL)
    rows = world.generate_dataset(dm.FIRST_TEST_ROW + 3, hidden, dm.np.random.default_rng(2032))
    assert [r.case_id for r in test] == [r.case_id for r in rows[dm.FIRST_TEST_ROW:]]
    assert train[0]["rendered_text"] == world.render(rows[0].visible)


def test_run_saves_answers_resumes_and_scores(monkeypatch, tmp_path):
    truth = dm.exact_probabilities(limit=2)
    sent = []

    def decide(payload, key):
        sent.append(payload["model"])
        return fake_response(0.5)

    monkeypatch.chdir(tmp_path)  # Never read a developer's real .env.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "create_decision", decide)
    dm.run(["fake/decider"], tmp_path, limit=2, concurrency=2)
    assert len(sent) == len(truth)
    dm.run(["fake/decider"], tmp_path, limit=2, concurrency=2)
    assert len(sent) == len(truth)  # Nothing is sent twice.

    records = dm.read_jsonl(dm.run_file(tmp_path, "fake/decider"))
    assert {(r["world"], r["seed"], r["case_id"]) for r in records} == truth.keys()
    results = dm.score(tmp_path, limit=2)
    assert [r["world"] for r in results] == list(dm.WORLDS)
    for r in results:
        expected = [math.fsum(dm.bernoulli_kl(truth[r["world"], s, c], 0.5, dm.EPSILON)
                              for (w, s, c) in truth if w == r["world"] and s == seed) / 2 for seed in dm.SEEDS]
        assert r["kl_nats"] == pytest.approx(sum(expected) / 3)
        assert r["cost_per_1000"] == pytest.approx(0.2)
        assert r["n_cases"] == r["expected_cases"] == 6


def test_exact_answers_reach_the_clipping_floor_and_duplicates_are_rejected(tmp_path):
    truth = dm.exact_probabilities(limit=1)
    lines = [{"model": "oracle", "world": w, "seed": s, "case_id": c, "probability_positive": p, "cost_usd": 0.001}
             for (w, s, c), p in truth.items()]
    path = tmp_path / "oracle.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    for r in dm.score(tmp_path, limit=1):
        # Clipping exact 0 or 1 to 0.001 or 0.999 costs about 0.001 nats, so the floor is not zero.
        floor = [dm.bernoulli_kl(p, p, dm.EPSILON) for (w, _, _), p in truth.items() if w == r["world"]]
        assert r["kl_nats"] == pytest.approx(sum(floor) / len(floor))
        assert r["kl_nats"] < 0.002
    path.write_text(path.read_text() + json.dumps(lines[0]) + "\n")
    with pytest.raises(ValueError, match="more than once"):
        dm.score(tmp_path, limit=1)


def test_plot_draws_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    results = [{"model": m, "world": w, "kl_nats": kl, "cost_per_1000": cost, "n_cases": 900,
                "expected_cases": 900}
               for w in dm.WORLDS for m, kl, cost in (("a/cheap", 0.3, 0.1), ("b/good", 0.2, 1.0),
                                                       ("c/dominated", 0.4, 2.0))]
    output = tmp_path / "cost_quality.png"
    dm.plot(results, output)
    assert output.stat().st_size > 10_000
    assert dm.frontier([(0.1, 0.3), (1.0, 0.2), (2.0, 0.4)]) == [(0.1, 0.3), (1.0, 0.2)]


def test_kl_clips_only_prediction_and_preserves_endpoint_conventions():
    assert dm.bernoulli_kl(0, 0) == dm.bernoulli_kl(1, 1) == 0
    assert math.isinf(dm.bernoulli_kl(.3, 0))
    assert dm.bernoulli_kl(1, 0, 1e-3) == pytest.approx(math.log(1000))
    assert dm.bernoulli_kl(0, 0, 1e-3) == pytest.approx(-math.log(.999))


def test_probability_parsing_rejects_malformed_answers():
    assert openrouter.positive_probability(fake_response(0.25)) == 0.25
    for bad in (fake_response(1.5), fake_response(float("nan")), fake_response("0.5"), {"answers": {}}, []):
        with pytest.raises(ValueError):
            openrouter.positive_probability(bad)


def test_key_comes_from_environment_or_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter.api_key()
    (tmp_path / ".env").write_text("# comment\nexport OPENROUTER_API_KEY='from-file'\n")
    assert openrouter.api_key() == "from-file"
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-environment")
    assert openrouter.api_key() == "from-environment"


def test_transient_errors_are_retried_and_others_raise(monkeypatch):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise HTTPError(req.full_url, 429, "rate limited", {}, io.BytesIO(b"slow down"))
        return io.BytesIO(json.dumps(fake_response(0.7)).encode())

    monkeypatch.setattr(openrouter.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(openrouter.request, "urlopen", urlopen)
    assert openrouter.positive_probability(openrouter.create_decision({"model": "m"}, "key")) == 0.7
    assert calls == [openrouter.DECISIONS_URL] * 2

    def unauthorized(req, timeout):
        raise HTTPError(req.full_url, 401, "unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(openrouter.request, "urlopen", unauthorized)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        openrouter.create_decision({"model": "m"}, "key")
