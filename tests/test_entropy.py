"""Tests for exact distribution entropy (transformers engine)."""

import math

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from entropy import _distribution_entropy


class TestDistributionEntropy:
    def test_uniform_distribution(self):
        vocab = 1000
        logits = torch.zeros(vocab)
        assert _distribution_entropy(logits) == pytest.approx(math.log(vocab), rel=1e-4)

    def test_peaked_distribution_near_zero(self):
        logits = torch.zeros(100)
        logits[7] = 30.0
        assert _distribution_entropy(logits) == pytest.approx(0.0, abs=1e-6)

    def test_two_equal_tokens(self):
        logits = torch.full((100,), -50.0)
        logits[0] = 0.0
        logits[1] = 0.0
        assert _distribution_entropy(logits) == pytest.approx(math.log(2), rel=1e-3)

    def test_minus_inf_logits_are_masked(self):
        # Simulates top-p filtering: filtered tokens carry -inf.
        logits = torch.full((100,), float("-inf"))
        logits[3] = 0.0
        logits[4] = 0.0
        h = _distribution_entropy(logits)
        assert not math.isnan(h)
        assert h == pytest.approx(math.log(2), rel=1e-3)

    def test_no_nan_with_mixed_inf(self):
        logits = torch.randn(500)
        logits[::3] = float("-inf")
        h = _distribution_entropy(logits)
        assert not math.isnan(h)
        assert h > 0

    def test_temperature_scaling_increases_entropy(self):
        logits = torch.randn(1000) * 5
        h_sharp = _distribution_entropy(logits / 0.5)
        h_flat = _distribution_entropy(logits / 2.0)
        assert h_flat > h_sharp

    def test_bf16_input(self):
        logits = torch.randn(256, dtype=torch.bfloat16)
        h = _distribution_entropy(logits)
        assert not math.isnan(h)
        assert h > 0
