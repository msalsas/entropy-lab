"""Integrity tests for the candidate question pool."""

from questions import QUESTION_POOL


class TestQuestionPool:
    def test_unique_ids(self):
        ids = [q["id"] for q in QUESTION_POOL]
        assert len(ids) == len(set(ids))

    def test_required_fields(self):
        for q in QUESTION_POOL:
            assert q["id"]
            assert q["lang"] in ("es", "en")
            assert q["category"]
            assert len(q["text"]) > 5

    def test_both_languages_present(self):
        langs = {q["lang"] for q in QUESTION_POOL}
        assert langs == {"es", "en"}

    def test_language_balance(self):
        n_es = sum(1 for q in QUESTION_POOL if q["lang"] == "es")
        n_en = sum(1 for q in QUESTION_POOL if q["lang"] == "en")
        assert n_es == n_en

    def test_category_diversity(self):
        categories = {q["category"] for q in QUESTION_POOL}
        assert len(categories) >= 4

    def test_minimum_pool_size(self):
        assert len(QUESTION_POOL) >= 20
