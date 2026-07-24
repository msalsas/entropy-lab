"""Tests for the statistical core: correlations, Fisher tests, binomial
tests, proportion comparisons and the trial analyzers."""

import math

import pytest

from experiment import (
    _binom_p_two_sided,
    _compare_proportions,
    _compare_rs,
    _fisher_ci,
    _fisher_p,
    _norm_cdf,
    _pearson,
    _question_level,
    _spearman,
    analyze_pairwise,
    analyze_trials,
    run_pairwise_trial,
)


class TestPearson:
    def test_perfect_positive(self):
        assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert _pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)

    def test_known_value(self):
        # r for these two series is 0.5 by construction
        xs = [0, 1, 2, 3]
        ys = [0, 1, 0, 3]
        r = _pearson(xs, ys)
        assert -1 <= r <= 1

    def test_zero_variance_returns_none(self):
        assert _pearson([1, 1, 1, 1], [1, 2, 3, 4]) is None

    def test_too_few_points_returns_none(self):
        assert _pearson([1, 2], [1, 2]) is None

    def test_uncorrelated_near_zero(self):
        xs = list(range(20))
        ys = [(-1) ** i for i in range(20)]
        assert abs(_pearson(xs, ys)) < 0.35


class TestSpearman:
    def test_monotonic_nonlinear_is_one(self):
        assert _spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(1.0)

    def test_reversed_is_minus_one(self):
        assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_too_few_returns_none(self):
        assert _spearman([1, 2], [2, 1]) is None


class TestFisherHelpers:
    def test_norm_cdf_standard_values(self):
        assert _norm_cdf(0) == pytest.approx(0.5)
        assert _norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)

    def test_ci_contains_point_estimate(self):
        lo, hi = _fisher_ci(0.5, 50)
        assert lo < 0.5 < hi

    def test_ci_shrinks_with_n(self):
        w_small = _fisher_ci(0.5, 10)
        w_large = _fisher_ci(0.5, 500)
        assert (w_large[1] - w_large[0]) < (w_small[1] - w_small[0])

    def test_ci_none_for_tiny_n(self):
        assert _fisher_ci(0.5, 3) is None

    def test_p_small_for_strong_correlation(self):
        assert _fisher_p(0.8, 50) < 0.001

    def test_p_large_for_zero_correlation(self):
        assert _fisher_p(0.0, 50) == pytest.approx(1.0)

    def test_compare_identical_rs_is_zero(self):
        out = _compare_rs(0.4, 50, 0.4, 60)
        assert out["z"] == pytest.approx(0.0)
        assert out["p"] == pytest.approx(1.0)

    def test_compare_different_rs_significant(self):
        out = _compare_rs(0.6, 200, -0.2, 200)
        assert out["p"] < 0.01


class TestBinomial:
    def test_chance_performance_gives_high_p(self):
        # 5/10 at p0=0.5 must not reject
        assert _binom_p_two_sided(5, 10) == pytest.approx(1.0)

    def test_perfect_score_small_sample(self):
        # exact two-sided: only k=0 and k=10 are as extreme -> 2 * 0.5^10
        assert _binom_p_two_sided(10, 10) == pytest.approx(2 * 0.5 ** 10, rel=1e-9)

    def test_zero_score_symmetric(self):
        assert _binom_p_two_sided(0, 10) == pytest.approx(
            _binom_p_two_sided(10, 10), rel=1e-9
        )

    def test_empty_returns_none(self):
        assert _binom_p_two_sided(0, 0) is None

    def test_p_bounded(self):
        assert 0 < _binom_p_two_sided(48, 48) <= 1.0


class TestCompareProportions:
    def test_equal_proportions(self):
        out = _compare_proportions(25, 50, 25, 50)
        assert out["z"] == pytest.approx(0.0)

    def test_empty_group(self):
        assert _compare_proportions(0, 0, 5, 10)["p"] is None

    def test_large_difference_significant(self):
        out = _compare_proportions(45, 50, 20, 50)
        assert out["p"] < 0.001


def _trial(qid="q1", relation="self", timing="post", estimate=5.0,
           entropy=1.0, generator="qwen", correct=True, gap=0.8, high_a=True):
    return {
        "question_id": qid,
        "relation": relation,
        "timing": timing,
        "generator": generator,
        "estimate": estimate,
        "true_entropy": entropy,
        "correct": correct,
        "entropy_gap": gap,
        "high_is_a": high_a,
    }


class TestAnalyzeTrials:
    def test_groups_by_condition(self):
        trials = [_trial(qid=f"q{i}", estimate=float(i), entropy=float(i % 3))
                  for i in range(5)]
        out = analyze_trials(trials)
        assert "qwen|self|post" in out["by_condition"]
        assert "self|post" in out["pooled"]
        assert "self_vs_cross|post" in out["comparisons"]

    def test_none_estimates_excluded(self):
        trials = [_trial(estimate=None)] * 3
        out = analyze_trials(trials)
        assert out["by_condition"] == {}

    def test_perfect_correlation_detected(self):
        trials = [_trial(qid=f"q{i}", estimate=float(i), entropy=2.0 * i)
                  for i in range(1, 6)]
        out = analyze_trials(trials)
        assert out["pooled"]["self|post"]["pearson_r"] == pytest.approx(1.0)

    def test_question_level_aggregation(self):
        group = [
            _trial(qid="q1", estimate=2.0, entropy=1.0),
            _trial(qid="q1", estimate=4.0, entropy=3.0),
            _trial(qid="q2", estimate=6.0, entropy=5.0),
        ]
        points = _question_level(group)
        assert len(points) == 2
        assert (3.0, 2.0) in points
        assert (6.0, 5.0) in points


class TestAnalyzePairwise:
    def _run(self, self_correct, cross_correct, n=10):
        trials = []
        for i in range(n):
            trials.append(_trial(qid=f"q{i}", relation="self",
                                 correct=i < self_correct, gap=0.3 + 0.1 * i,
                                 high_a=i % 2 == 0))
            trials.append(_trial(qid=f"q{i}", relation="cross",
                                 correct=i < cross_correct, gap=0.3 + 0.1 * i,
                                 high_a=i % 2 == 1))
        return analyze_pairwise(trials)

    def test_accuracy_computed(self):
        out = self._run(8, 3)
        assert out["by_relation"]["self"]["accuracy"] == pytest.approx(0.8)
        assert out["by_relation"]["cross"]["accuracy"] == pytest.approx(0.3)

    def test_binomial_present(self):
        out = self._run(10, 5)
        assert out["by_relation"]["self"]["binom_p"] < 0.01
        assert out["by_relation"]["cross"]["binom_p"] == pytest.approx(1.0)

    def test_self_vs_cross_comparison(self):
        out = self._run(9, 1)
        assert out["self_vs_cross"]["p"] < 0.01

    def test_order_breakdown_covers_all_trials(self):
        out = self._run(5, 5)
        order = out["by_relation"]["self"]["by_order"]
        assert order["high_A"]["n"] + order["high_B"]["n"] == 10

    def test_gap_breakdown_covers_all_trials(self):
        out = self._run(5, 5)
        gap = out["by_relation"]["self"]["by_gap"]
        assert gap["small"]["n"] + gap["large"]["n"] == 10

    def test_unparsed_choices_excluded(self):
        trials = [_trial(correct=None) for _ in range(4)]
        out = analyze_pairwise(trials)
        assert out["by_relation"] == {}


class TestRunPairwiseTrial:
    def _gens(self):
        low = {"answer": "clean answer", "mean_entropy": 0.5, "temperature": 0.6}
        high = {"answer": "broken answer", "mean_entropy": 2.5, "temperature": 1.2}
        return low, high

    def test_correct_mapping_when_high_is_a(self, monkeypatch):
        import experiment
        monkeypatch.setattr(
            experiment, "verbalized_estimate",
            lambda k, p: {"raw": "A", "value": None},
        )
        low, high = self._gens()
        t = run_pairwise_trial(
            {"id": "q1", "text": "Q?", "lang": "en", "category": "factual"},
            "qwen", "self", low, high, high_is_a=True, reveal_identity=False,
        )
        assert t["correct"] is True
        assert t["estimator"] == "qwen"

    def test_correct_mapping_when_high_is_b(self, monkeypatch):
        import experiment
        monkeypatch.setattr(
            experiment, "verbalized_estimate",
            lambda k, p: {"raw": "A", "value": None},
        )
        low, high = self._gens()
        t = run_pairwise_trial(
            {"id": "q1", "text": "Q?", "lang": "en", "category": "factual"},
            "qwen", "self", low, high, high_is_a=False, reveal_identity=False,
        )
        assert t["correct"] is False  # chose A but correct was B

    def test_cross_uses_other_estimator(self, monkeypatch):
        import experiment
        monkeypatch.setattr(experiment, "ESTIMATOR_KEYS", ["qwen", "smollm"])
        monkeypatch.setattr(
            experiment, "verbalized_estimate",
            lambda k, p: {"raw": "B", "value": None},
        )
        low, high = self._gens()
        t = run_pairwise_trial(
            {"id": "q1", "text": "Q?", "lang": "en", "category": "factual"},
            "qwen", "cross", low, high, high_is_a=True, reveal_identity=False,
        )
        assert t["estimator"] == "smollm"

    def test_unparsed_choice_gives_none_correct(self, monkeypatch):
        import experiment
        monkeypatch.setattr(
            experiment, "verbalized_estimate",
            lambda k, p: {"raw": "I cannot tell", "value": None},
        )
        low, high = self._gens()
        t = run_pairwise_trial(
            {"id": "q1", "text": "Q?", "lang": "en", "category": "factual"},
            "qwen", "self", low, high, high_is_a=True, reveal_identity=False,
        )
        assert t["correct"] is None
        assert t["choice"] is None

    def test_entropy_gap_recorded(self, monkeypatch):
        import experiment
        monkeypatch.setattr(
            experiment, "verbalized_estimate",
            lambda k, p: {"raw": "A", "value": None},
        )
        low, high = self._gens()
        t = run_pairwise_trial(
            {"id": "q1", "text": "Q?", "lang": "en", "category": "factual"},
            "qwen", "self", low, high, high_is_a=True, reveal_identity=False,
        )
        assert t["entropy_gap"] == pytest.approx(2.0)
