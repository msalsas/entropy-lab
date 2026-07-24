"""Tests for the GGUF engine parts that do not require loading models:
prompt templating and the manual temperature/top-p sampler."""

import math

import numpy as np
import pytest

llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")

from gguf_engine import _format_prompt, _sample_step


class TestFormatPrompt:
    def test_chatml(self):
        p = _format_prompt("chatml", "Hello")
        assert "<|im_start|>user" in p
        assert "Hello" in p
        assert p.endswith("<|im_start|>assistant\n")

    def test_phi3(self):
        p = _format_prompt("phi3", "Hello")
        assert p == "<|user|>\nHello<|end|>\n<|assistant|>\n"

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError):
            _format_prompt("mystery", "Hello")


class TestSampleStep:
    VOCAB = 500

    def _rng(self):
        return np.random.default_rng(42)

    def test_peaked_logits_entropy_near_zero(self):
        logits = np.full(self.VOCAB, -10.0)
        logits[11] = 10.0
        token, entropy = _sample_step(logits, 1.0, 1.0, self._rng())
        assert token == 11
        assert entropy == pytest.approx(0.0, abs=1e-6)

    def test_uniform_logits_max_entropy(self):
        logits = np.zeros(self.VOCAB)
        token, entropy = _sample_step(logits, 1.0, 1.0, self._rng())
        assert entropy == pytest.approx(math.log(self.VOCAB), rel=1e-3)

    def test_low_temperature_picks_argmax(self):
        logits = np.random.default_rng(1).normal(0, 1, self.VOCAB)
        argmax = int(np.argmax(logits))
        for _ in range(5):
            token, _ = _sample_step(logits, 0.01, 1.0, self._rng())
            assert token == argmax

    def test_top_p_restricts_support(self):
        # One dominant token: top_p=0.5 must exclude everything else.
        logits = np.full(self.VOCAB, -20.0)
        logits[3] = 0.0
        logits[4] = -1.0
        for _ in range(10):
            token, entropy = _sample_step(logits, 1.0, 0.5, self._rng())
            assert token == 3
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_top_p_one_keeps_full_support(self):
        logits = np.zeros(self.VOCAB)
        _, entropy = _sample_step(logits, 1.0, 1.0, self._rng())
        assert entropy == pytest.approx(math.log(self.VOCAB), rel=1e-3)

    def test_seed_determinism(self):
        logits = np.random.default_rng(7).normal(0, 2, self.VOCAB)
        t1, _ = _sample_step(logits, 1.0, 0.95, np.random.default_rng(123))
        t2, _ = _sample_step(logits, 1.0, 0.95, np.random.default_rng(123))
        assert t1 == t2

    def test_sampled_token_within_vocab(self):
        logits = np.random.default_rng(3).normal(0, 1, self.VOCAB)
        for _ in range(20):
            token, _ = _sample_step(logits, 0.9, 0.95, self._rng())
            assert 0 <= token < self.VOCAB

    def test_entropy_matches_manual_computation(self):
        logits = np.array([2.0, 1.0, 0.0, -1.0] + [-30.0] * 96)
        _, entropy = _sample_step(logits, 1.0, 1.0, self._rng())
        p = np.exp(logits - logits.max())
        p /= p.sum()
        expected = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        assert entropy == pytest.approx(expected, rel=1e-9)
