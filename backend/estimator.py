"""Verbalized entropy estimation.

The estimator model is asked to rate, on a 0-10 scale, how unpredictable
the word-by-word generation of an answer was (post-hoc) or will be
(a priori). Conditions:

- self  : the estimator judges its own answer.
- cross : the estimator judges another model's answer (the textual
          baseline: same visible information, different observer).
- pre   : estimation happens *before* the answer is generated, so no
          textual evidence exists yet.
- post  : estimation happens with the answer text visible.

When ``reveal_identity`` is False the prompt never names the generating
model, so the estimator cannot fall back on trained knowledge about a
specific model's typical behaviour.
"""

import re

try:  # transformers stack is optional when only GGUF models are used
    import torch
    from models import load_model, MODEL_REGISTRY
except ImportError:  # pragma: no cover - depends on environment
    torch = None
    load_model = None
    MODEL_REGISTRY = {}

if torch is None:  # keep @torch.no_grad() decorators harmless
    class _NoGrad:
        def __call__(self, fn):
            return fn

    class _TorchStub:
        @staticmethod
        def no_grad():
            return _NoGrad()

    torch = _TorchStub()

MAX_ESTIMATE_TOKENS = 12

# Shared scale anchors so every condition uses the same yardstick. The
# anchors make estimates comparable across trials and encourage the full
# 0-10 range (small models otherwise compress to low integers).
_SCALE = (
    "Use the full scale, decimals allowed. Reference points: a short "
    "factual answer with one obvious wording is about 1; an open "
    "creative answer with many equally valid wordings is about 8.\n\n"
)

_PRE_SELF = (
    "You are about to answer a question. Before answering, rate how "
    "unpredictable your word-by-word choices will be while writing the "
    "answer, on a scale from 0 (completely predictable) to 10 "
    "(completely unpredictable). " + _SCALE +
    "Question: {question}\n\n"
    "Reply with a single number only."
)

_PRE_CROSS_HIDDEN = (
    "A language model is about to answer a question. Rate how "
    "unpredictable its word-by-word choices will be while writing the "
    "answer, on a scale from 0 (completely predictable) to 10 "
    "(completely unpredictable). " + _SCALE +
    "Question: {question}\n\n"
    "Reply with a single number only."
)

_PRE_CROSS_NAMED = (
    "The language model {model_label} is about to answer a question. "
    "Rate how unpredictable its word-by-word choices will be while "
    "writing the answer, on a scale from 0 (completely predictable) to "
    "10 (completely unpredictable). " + _SCALE +
    "Question: {question}\n\n"
    "Reply with a single number only."
)

_POST_SELF = (
    "You answered a question. Rate how unpredictable your word-by-word "
    "choices were while writing this answer, on a scale from 0 "
    "(completely predictable) to 10 (completely unpredictable). " + _SCALE +
    "Question: {question}\n\nYour answer: {answer}\n\n"
    "Reply with a single number only."
)

_POST_CROSS_HIDDEN = (
    "A language model answered a question. Rate how unpredictable its "
    "word-by-word choices were while writing this answer, on a scale "
    "from 0 (completely predictable) to 10 (completely unpredictable). "
    + _SCALE +
    "Question: {question}\n\nAnswer: {answer}\n\n"
    "Reply with a single number only."
)

_POST_CROSS_NAMED = (
    "The language model {model_label} answered a question. Rate how "
    "unpredictable its word-by-word choices were while writing this "
    "answer, on a scale from 0 (completely predictable) to 10 "
    "(completely unpredictable). " + _SCALE +
    "Question: {question}\n\n"
    "Answer: {answer}\n\nReply with a single number only."
)


def build_estimation_prompt(
    timing: str,
    estimator_relation: str,
    question: str,
    answer: str | None,
    generator_key: str,
    reveal_identity: bool,
) -> str:
    """Assemble the estimation prompt for the given condition."""
    if timing == "pre":
        if estimator_relation == "self":
            return _PRE_SELF.format(question=question)
        template = _PRE_CROSS_NAMED if reveal_identity else _PRE_CROSS_HIDDEN
        return template.format(
            question=question,
            model_label=_label_for(generator_key),
        )

    if estimator_relation == "self":
        return _POST_SELF.format(question=question, answer=answer or "")
    template = _POST_CROSS_NAMED if reveal_identity else _POST_CROSS_HIDDEN
    return template.format(
        question=question,
        answer=answer or "",
        model_label=_label_for(generator_key),
    )


@torch.no_grad()
def verbalized_estimate(model_key: str, prompt: str) -> dict:
    """Ask a model for a numeric unpredictability estimate.

    Decoding is deterministic (greedy) so estimates do not add sampling
    noise to the measurement itself.
    """
    tokenizer, model = load_model(model_key)
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    out = model.generate(
        **enc, max_new_tokens=MAX_ESTIMATE_TOKENS, do_sample=False
    )
    raw = tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return {"raw": raw, "value": parse_estimate(raw)}


def parse_estimate(text: str) -> float | None:
    """Extract the first 0-10 number from the model's reply."""
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return min(max(value, 0.0), 10.0)


# ---------------------------------------------------------------------------
# Pairwise (comparative) elicitation
#
# Instead of an absolute number, the judge picks which of two answers was
# generated with more unpredictable word choices. The correct answer is
# objectively known from the true entropies, so performance is measured as
# accuracy against 50% chance - a much friendlier task for small models.
# ---------------------------------------------------------------------------

_PAIR_SELF = (
    "You answered the same question twice, at two different moments. In "
    "which answer were your word-by-word choices more unpredictable?\n\n"
    "Question: {question}\n\nAnswer A: {answer_a}\n\nAnswer B: {answer_b}\n\n"
    "Reply with a single letter only: A or B."
)

_PAIR_CROSS_HIDDEN = (
    "A language model answered the same question twice, at two different "
    "moments. In which answer were its word-by-word choices more "
    "unpredictable?\n\n"
    "Question: {question}\n\nAnswer A: {answer_a}\n\nAnswer B: {answer_b}\n\n"
    "Reply with a single letter only: A or B."
)

_PAIR_CROSS_NAMED = (
    "The language model {model_label} answered the same question twice, "
    "at two different moments. In which answer were its word-by-word "
    "choices more unpredictable?\n\n"
    "Question: {question}\n\nAnswer A: {answer_a}\n\nAnswer B: {answer_b}\n\n"
    "Reply with a single letter only: A or B."
)


def _label_for(model_key: str) -> str:
    """Resolve a display label across both model registries."""
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]["label"]
    try:
        from gguf_engine import MODEL_REGISTRY_GGUF

        if model_key in MODEL_REGISTRY_GGUF:
            return MODEL_REGISTRY_GGUF[model_key]["label"]
    except Exception:  # noqa: BLE001 - label is cosmetic
        pass
    return model_key


def build_pairwise_prompt(
    relation: str,
    question: str,
    answer_a: str,
    answer_b: str,
    generator_key: str,
    reveal_identity: bool,
) -> str:
    """Assemble the comparative-judgment prompt for the given condition."""
    if relation == "self":
        return _PAIR_SELF.format(
            question=question, answer_a=answer_a, answer_b=answer_b
        )
    template = _PAIR_CROSS_NAMED if reveal_identity else _PAIR_CROSS_HIDDEN
    return template.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        model_label=_label_for(generator_key),
    )


def parse_choice(text: str) -> str | None:
    """Extract the chosen letter (A or B) from the model's reply."""
    match = re.search(r"\b([AB])\b", text.upper())
    return match.group(1) if match else None
