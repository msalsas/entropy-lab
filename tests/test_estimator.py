"""Tests for estimation prompt construction and response parsing."""

import pytest

from estimator import (
    build_estimation_prompt,
    build_pairwise_prompt,
    parse_choice,
    parse_estimate,
    _label_for,
)


class TestParseEstimate:
    def test_plain_integer(self):
        assert parse_estimate("3") == 3.0

    def test_decimal(self):
        assert parse_estimate("7.5") == 7.5

    def test_number_embedded_in_text(self):
        assert parse_estimate("I would rate it 4 out of 10") == 4.0

    def test_clamps_above_ten(self):
        assert parse_estimate("42") == 10.0

    def test_no_number_returns_none(self):
        assert parse_estimate("very unpredictable") is None

    def test_empty_string(self):
        assert parse_estimate("") is None

    def test_first_number_wins(self):
        assert parse_estimate("8, maybe 9") == 8.0

    def test_zero_is_valid(self):
        assert parse_estimate("0") == 0.0


class TestParseChoice:
    def test_single_letter_a(self):
        assert parse_choice("A") == "A"

    def test_single_letter_b(self):
        assert parse_choice("B") == "B"

    def test_lowercase_normalized(self):
        assert parse_choice("b") == "B"

    def test_sentence(self):
        assert parse_choice("The more unpredictable one is B.") == "B"

    def test_no_letter_returns_none(self):
        assert parse_choice("I cannot decide") is None

    def test_empty(self):
        assert parse_choice("") is None

    def test_first_standalone_letter_wins(self):
        assert parse_choice("A or B") == "A"


class TestLabelFor:
    def test_transformers_registry_key(self, monkeypatch):
        import estimator
        monkeypatch.setattr(
            estimator, "MODEL_REGISTRY",
            {"qwen": {"label": "Qwen2.5-0.5B-Instruct"}},
        )
        assert _label_for("qwen") == "Qwen2.5-0.5B-Instruct"

    def test_gguf_registry_key(self):
        assert _label_for("qwen3b") == "Qwen2.5-3B-Instruct"

    def test_unknown_key_returns_key(self):
        assert _label_for("some-remote-model") == "some-remote-model"


class TestBuildEstimationPrompt:
    QUESTION = "What is the capital of France?"
    ANSWER = "Paris."

    @pytest.fixture(autouse=True)
    def registry(self, monkeypatch):
        import estimator
        monkeypatch.setattr(
            estimator, "MODEL_REGISTRY",
            {"qwen": {"label": "Qwen2.5-0.5B-Instruct"},
             "smollm": {"label": "SmolLM2-360M-Instruct"}},
        )

    def test_pre_self_mentions_no_answer(self):
        p = build_estimation_prompt("pre", "self", self.QUESTION, None, "qwen", False)
        assert self.QUESTION in p
        assert "Your answer" not in p
        assert "about to answer" in p

    def test_pre_cross_hidden_omits_model_label(self):
        p = build_estimation_prompt("pre", "cross", self.QUESTION, None, "qwen", False)
        assert "Qwen2.5-0.5B-Instruct" not in p
        assert "A language model" in p

    def test_pre_cross_named_includes_model_label(self):
        p = build_estimation_prompt("pre", "cross", self.QUESTION, None, "qwen", True)
        assert "Qwen2.5-0.5B-Instruct" in p

    def test_post_self_includes_answer(self):
        p = build_estimation_prompt("post", "self", self.QUESTION, self.ANSWER, "qwen", False)
        assert self.ANSWER in p
        assert "You answered" in p

    def test_post_cross_hidden_omits_label_but_includes_answer(self):
        p = build_estimation_prompt("post", "cross", self.QUESTION, self.ANSWER, "smollm", False)
        assert self.ANSWER in p
        assert "SmolLM2-360M-Instruct" not in p

    def test_post_cross_named_includes_label(self):
        p = build_estimation_prompt("post", "cross", self.QUESTION, self.ANSWER, "smollm", True)
        assert "SmolLM2-360M-Instruct" in p

    def test_all_prompts_define_the_scale(self):
        for timing in ("pre", "post"):
            for relation in ("self", "cross"):
                p = build_estimation_prompt(
                    timing, relation, self.QUESTION, self.ANSWER, "qwen", False
                )
                assert "0" in p and "10" in p

    def test_all_prompts_ask_for_single_number(self):
        for timing in ("pre", "post"):
            for relation in ("self", "cross"):
                p = build_estimation_prompt(
                    timing, relation, self.QUESTION, self.ANSWER, "qwen", False
                )
                assert "single number" in p


class TestBuildPairwisePrompt:
    def test_self_framing(self):
        p = build_pairwise_prompt("self", "Q?", "ans one", "ans two", "qwen", False)
        assert "You answered" in p
        assert "ans one" in p and "ans two" in p

    def test_cross_hidden_omits_label(self):
        p = build_pairwise_prompt("cross", "Q?", "ans one", "ans two", "qwen", False)
        assert "Qwen2.5-0.5B-Instruct" not in p
        assert "A language model" in p

    def test_cross_named_includes_label(self):
        p = build_pairwise_prompt("cross", "Q?", "ans one", "ans two", "qwen3b", True)
        assert "Qwen2.5-3B-Instruct" in p

    def test_asks_for_single_letter(self):
        p = build_pairwise_prompt("self", "Q?", "a", "b", "qwen", False)
        assert "A or B" in p

    def test_answers_not_swapped(self):
        p = build_pairwise_prompt("cross", "Q?", "FIRST", "SECOND", "qwen", False)
        assert p.index("FIRST") < p.index("SECOND")
