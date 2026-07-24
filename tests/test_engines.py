"""Tests for the engine dispatch layer and engine-aware trial runners."""

import sys

import pytest

import api_engine
import engines
import experiment


@pytest.fixture
def clean_config(monkeypatch, tmp_path):
    """Isolate config: no real .env, no inherited environment."""
    env_file = tmp_path / ".env"
    monkeypatch.setattr(api_engine, "ENV_LOCATIONS", [str(env_file)])
    monkeypatch.setattr(api_engine, "_config", None)
    for key in api_engine.DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    return env_file


def write_env(path, text):
    path.write_text(text, encoding="utf-8")
    api_engine._config = None


class TestActiveEngine:
    def test_default_is_local(self, clean_config):
        assert engines.active_engine() == "local"

    def test_env_file_selects_api(self, clean_config):
        write_env(clean_config, "LAB_ENGINE=api\n")
        assert engines.active_engine() == "api"

    def test_real_env_overrides_env_file(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_ENGINE=api\n")
        monkeypatch.setenv("LAB_ENGINE", "gguf")
        api_engine._config = None
        assert engines.active_engine() == "gguf"

    def test_value_is_normalized(self, clean_config):
        write_env(clean_config, "LAB_ENGINE= API \n")
        assert engines.active_engine() == "api"


class TestGetRegistry:
    def test_api_registry(self, clean_config, monkeypatch):
        write_env(
            clean_config,
            "LAB_ENGINE=api\nLAB_MODEL_A=qwen7b\nLAB_MODEL_A_LABEL=Qwen 7B\n",
        )
        registry = engines.get_registry()
        assert list(registry.keys()) == ["model_a", "model_b"]
        assert registry["model_a"]["label"] == "Qwen 7B"
        assert registry["model_a"]["remote_id"] == "qwen7b"

    def test_gguf_registry(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_ENGINE=gguf\n")
        fake = {"m1": {"label": "M1"}, "m2": {"label": "M2"}}
        module = pytest.importorskip("gguf_engine")
        monkeypatch.setattr(module, "MODEL_REGISTRY_GGUF", fake)
        assert engines.get_registry() is fake


class TestDispatch:
    def test_generate_api(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_ENGINE=api\n")
        calls = {}

        def fake_generate(model_key, question_text, temperature, seed,
                          max_new_tokens=48, **kw):
            calls["args"] = (model_key, question_text, temperature, seed,
                             max_new_tokens)
            return {"answer": "hi", "n_tokens": 1, "mean_entropy": 0.5}

        monkeypatch.setattr(api_engine, "generate", fake_generate)
        out = engines.generate("model_a", "Q?", 0.7, 42, 16)
        assert out["mean_entropy"] == 0.5
        assert calls["args"] == ("model_a", "Q?", 0.7, 42, 16)

    def test_judge_raw_api(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_ENGINE=api\n")
        monkeypatch.setattr(api_engine, "judge", lambda k, p: "B")
        assert engines.judge_raw("model_b", "prompt") == "B"

    def test_estimate_api_parses_value(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_ENGINE=api\n")
        monkeypatch.setattr(api_engine, "judge", lambda k, p: "about 7.5/10")
        est = engines.estimate("model_a", "prompt")
        assert est["raw"] == "about 7.5/10"
        assert est["value"] == 7.5


class TestRefreshEngines:
    def test_syncs_estimator_keys(self, clean_config, monkeypatch):
        monkeypatch.setattr(experiment, "ESTIMATOR_KEYS", ["qwen", "smollm"])
        monkeypatch.setattr(
            engines, "get_registry",
            lambda: {"model_a": {"label": "A"}, "model_b": {"label": "B"}},
        )
        keys = experiment.refresh_engines()
        assert keys == ["model_a", "model_b"]
        assert experiment.ESTIMATOR_KEYS == ["model_a", "model_b"]


class TestTrialsInApiMode:
    """Trial runners must route through the active engine, not torch."""

    @pytest.fixture
    def api_mode(self, clean_config, monkeypatch):
        write_env(clean_config, "LAB_ENGINE=api\n")
        monkeypatch.setattr(
            experiment, "ESTIMATOR_KEYS", ["model_a", "model_b"]
        )

    def test_pairwise_trial_uses_api_judge(self, api_mode, monkeypatch):
        judged = {}

        def fake_judge(model_key, prompt):
            judged["estimator"] = model_key
            return "A"

        monkeypatch.setattr(engines, "judge_raw", fake_judge)
        trial = experiment.run_pairwise_trial(
            question={"id": "q1", "text": "Q?", "lang": "en",
                      "category": "factual"},
            generator_key="model_a",
            relation="self",
            gen_low={"answer": "low", "temperature": 0.6,
                     "mean_entropy": 1.0},
            gen_high={"answer": "high", "temperature": 1.2,
                      "mean_entropy": 3.0},
            high_is_a=True,
            reveal_identity=False,
        )
        assert judged["estimator"] == "model_a"
        assert trial["choice"] == "A"
        assert trial["choice_raw"] == "A"
        assert trial["correct"] is True
        assert trial["entropy_gap"] == pytest.approx(2.0)

    def test_run_trial_uses_api_engine(self, api_mode, monkeypatch):
        monkeypatch.setattr(
            engines, "generate",
            lambda *a, **kw: {
                "answer": "some answer",
                "n_tokens": 5,
                "mean_entropy": 2.25,
                "topk_mass_mean": 0.97,
            },
        )
        monkeypatch.setattr(
            engines, "estimate",
            lambda k, p: {"raw": "6", "value": 6.0},
        )
        trial = experiment.run_trial(
            question={"id": "q1", "text": "Q?", "lang": "en",
                      "category": "factual"},
            generator_key="model_a",
            timing="post",
            relation="cross",
            temperature=0.9,
            seed=1,
            max_new_tokens=16,
            reveal_identity=False,
        )
        assert trial["estimator"] == "model_b"
        assert trial["true_entropy"] == 2.25
        assert trial["topk_mass_mean"] == 0.97
        assert trial["estimate"] == 6.0
