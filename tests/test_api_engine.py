"""Tests for the remote OpenAI-compatible engine: configuration
priority, .env parsing, top-k entropy approximation and mocked HTTP."""

import math
import os

import pytest

import api_engine


@pytest.fixture(autouse=True)
def clean_config(monkeypatch, tmp_path):
    """Isolate every test from real env vars and .env files."""
    for key in api_engine.DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(api_engine, "ENV_LOCATIONS", [str(env_file)])
    monkeypatch.setattr(api_engine, "_config", None)
    return env_file


def write_env(path, content):
    path.write_text(content, encoding="utf-8")


class TestEnvParsing:
    def test_basic_key_value(self, clean_config):
        write_env(clean_config, "LAB_API_BASE=http://host:1234/v1\n")
        assert api_engine.get_config()["LAB_API_BASE"] == "http://host:1234/v1"

    def test_comments_and_blank_lines_ignored(self, clean_config):
        write_env(clean_config, "# comment\n\nLAB_MODEL_A=foo\n")
        assert api_engine.get_config()["LAB_MODEL_A"] == "foo"

    def test_quoted_values_unquoted(self, clean_config):
        write_env(clean_config, 'LAB_MODEL_A="quoted-model"\n')
        assert api_engine.get_config()["LAB_MODEL_A"] == "quoted-model"

    def test_whitespace_tolerated(self, clean_config):
        write_env(clean_config, "  LAB_MODEL_A  =  spaced  \n")
        assert api_engine.get_config()["LAB_MODEL_A"] == "spaced"

    def test_missing_file_uses_defaults(self, clean_config):
        assert api_engine.get_config()["LAB_API_BASE"] == "http://localhost:1234/v1"


class TestConfigPriority:
    def test_env_file_overrides_defaults(self, clean_config):
        write_env(clean_config, "LAB_TOP_LOGPROBS=50\n")
        assert api_engine.get_config()["LAB_TOP_LOGPROBS"] == "50"

    def test_real_env_overrides_env_file(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_MODEL_A=file-model\n")
        monkeypatch.setenv("LAB_MODEL_A", "env-model")
        monkeypatch.setattr(api_engine, "_config", None)
        assert api_engine.get_config()["LAB_MODEL_A"] == "env-model"

    def test_unknown_keys_ignored(self, clean_config):
        write_env(clean_config, "UNRELATED_VAR=x\nLAB_MODEL_A=ok\n")
        cfg = api_engine.get_config()
        assert cfg["LAB_MODEL_A"] == "ok"
        assert "UNRELATED_VAR" not in cfg


class TestModelRegistry:
    def test_registry_from_config(self, clean_config):
        write_env(clean_config, "LAB_MODEL_A=qa\nLAB_MODEL_A_LABEL=QA\n"
                                "LAB_MODEL_B=mb\nLAB_MODEL_B_LABEL=MB\n")
        reg = api_engine.model_registry()
        assert reg["model_a"]["remote_id"] == "qa"
        assert reg["model_a"]["label"] == "QA"
        assert reg["model_b"]["remote_id"] == "mb"
        assert reg["model_b"]["label"] == "MB"


class TestEntropyFromTopLogprobs:
    def test_empty_returns_nan(self):
        h, mass = api_engine._entropy_from_top_logprobs([])
        assert math.isnan(h) and mass == 0.0

    def test_single_token_zero_entropy(self):
        h, mass = api_engine._entropy_from_top_logprobs(
            [{"token": "x", "logprob": -0.2}]
        )
        assert h == pytest.approx(0.0)
        assert mass == pytest.approx(math.exp(-0.2))

    def test_known_distribution(self):
        # two tokens at 50/50 after renormalization
        h, mass = api_engine._entropy_from_top_logprobs(
            [{"token": "a", "logprob": math.log(0.4)},
             {"token": "b", "logprob": math.log(0.4)}]
        )
        assert h == pytest.approx(math.log(2), rel=1e-6)
        assert mass == pytest.approx(0.8)

    def test_mass_tracks_coverage(self):
        h, mass = api_engine._entropy_from_top_logprobs(
            [{"token": "a", "logprob": math.log(0.95)},
             {"token": "b", "logprob": math.log(0.04)}]
        )
        assert mass == pytest.approx(0.99)
        assert h > 0


FAKE_LOGPROBS = [
    {"token": "Paris", "logprob": -0.1,
     "top_logprobs": [{"token": "Paris", "logprob": -0.1},
                      {"token": "Lyon", "logprob": -3.0}]},
    {"token": ".", "logprob": -0.05,
     "top_logprobs": [{"token": ".", "logprob": -0.05},
                      {"token": "!", "logprob": -4.0}]},
]


def fake_response(content="Paris.", logprobs=True):
    return {
        "choices": [{
            "message": {"content": content},
            "logprobs": {"content": FAKE_LOGPROBS} if logprobs else None,
        }]
    }


class TestGenerateAndJudge:
    def test_generate_records_entropy(self, clean_config, monkeypatch):
        monkeypatch.setattr(api_engine, "_post", lambda e, p: fake_response())
        r = api_engine.generate("model_a", "Q?", temperature=0.8, seed=1)
        assert r["answer"] == "Paris."
        assert r["n_tokens"] == 2
        assert r["mean_entropy"] > 0
        assert 0.9 < r["topk_mass_mean"] <= 1.0

    def test_generate_sends_sampling_params(self, clean_config, monkeypatch):
        captured = {}
        def spy(endpoint, payload):
            captured.update(payload)
            return fake_response()
        monkeypatch.setattr(api_engine, "_post", spy)
        api_engine.generate("model_a", "Q?", temperature=0.7, seed=99, top_p=0.9)
        assert captured["temperature"] == 0.7
        assert captured["seed"] == 99
        assert captured["top_p"] == 0.9
        assert captured["logprobs"] is True

    def test_generate_uses_configured_model_id(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_MODEL_A=my-served-model\n")
        captured = {}
        def spy(endpoint, payload):
            captured.update(payload)
            return fake_response()
        monkeypatch.setattr(api_engine, "_post", spy)
        api_engine.generate("model_a", "Q?", temperature=0.8, seed=1)
        assert captured["model"] == "my-served-model"

    def test_answer_is_stripped(self, clean_config, monkeypatch):
        monkeypatch.setattr(
            api_engine, "_post", lambda e, p: fake_response(content="  padded  ")
        )
        r = api_engine.generate("model_a", "Q?", temperature=0.8, seed=1)
        assert r["answer"] == "padded"

    def test_judge_uses_greedy(self, clean_config, monkeypatch):
        captured = {}
        def spy(endpoint, payload):
            captured.update(payload)
            return fake_response(content="A", logprobs=False)
        monkeypatch.setattr(api_engine, "_post", spy)
        out = api_engine.judge("model_a", "pick one")
        assert out == "A"
        assert captured["temperature"] == 0.0
        assert "logprobs" not in captured

    def test_missing_logprobs_tolerated(self, clean_config, monkeypatch):
        monkeypatch.setattr(
            api_engine, "_post", lambda e, p: fake_response(logprobs=False)
        )
        r = api_engine.generate("model_a", "Q?", temperature=0.8, seed=1)
        assert math.isnan(r["mean_entropy"])
        assert r["topk_mass_mean"] is None


class TestRequireParameters:
    """OpenRouter-style strict provider routing (logprobs safety)."""

    def _captured_payload(self, clean_config, monkeypatch, env_text):
        write_env(clean_config, env_text)
        captured = {}

        def fake_post(endpoint, payload):
            captured["p"] = payload
            return {"choices": [{"message": {"content": "ok"},
                                 "logprobs": {"content": []}}]}

        monkeypatch.setattr(api_engine, "_post", fake_post)
        api_engine.generate("model_a", "Q?", 0.7, 1, 4)
        return captured["p"]

    def test_provider_field_added_when_enabled(self, clean_config, monkeypatch):
        payload = self._captured_payload(
            clean_config, monkeypatch, "LAB_REQUIRE_PARAMETERS=true\n")
        assert payload["provider"] == {"require_parameters": True}

    def test_provider_field_absent_by_default(self, clean_config, monkeypatch):
        payload = self._captured_payload(clean_config, monkeypatch, "")
        assert "provider" not in payload

    def test_temps_have_defaults(self, clean_config):
        cfg = api_engine.get_config()
        assert float(cfg["LAB_TEMP_LOW"]) == 0.6
        assert float(cfg["LAB_TEMP_HIGH"]) == 1.2
