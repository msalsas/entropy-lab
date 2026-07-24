"""Engine dispatch: route generation and judgment to the active backend.

The active engine is selected with ``LAB_ENGINE`` (configuration
priority: built-in defaults < ``.env`` < real environment variables):

- ``local``: local Hugging Face transformers models, exact
  full-vocabulary entropy (default).
- ``gguf``: local GGUF models via llama.cpp, exact full-vocabulary
  entropy.
- ``api``: remote OpenAI-compatible server (e.g. LM Studio on another
  machine), entropy approximated over the top-k logprobs.

All heavy imports are lazy so that e.g. an api-only deployment does not
need torch or llama-cpp-python installed.
"""

from api_engine import get_config


def active_engine() -> str:
    """Name of the active engine: ``local``, ``gguf`` or ``api``."""
    return get_config().get("LAB_ENGINE", "local").strip().lower()


def get_registry() -> dict:
    """Model registry of the active engine: ``{key: {"label": ...}}``."""
    eng = active_engine()
    if eng == "api":
        from api_engine import model_registry

        return model_registry()
    if eng == "gguf":
        from gguf_engine import MODEL_REGISTRY_GGUF

        return MODEL_REGISTRY_GGUF
    from models import MODEL_REGISTRY

    return MODEL_REGISTRY


def generate(
    model_key: str,
    question_text: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> dict:
    """Generate an answer with per-token entropy via the active engine.

    Returns a dict with at least ``answer``, ``n_tokens`` and
    ``mean_entropy`` (api engine also carries ``topk_mass_mean``).
    """
    eng = active_engine()
    if eng == "api":
        from api_engine import generate as api_generate

        return api_generate(
            model_key, question_text, temperature, seed, max_new_tokens
        )
    if eng == "gguf":
        from gguf_engine import generate as gguf_generate

        return gguf_generate(
            model_key, question_text, temperature, seed, max_new_tokens
        )
    from entropy import generate_with_entropy

    return generate_with_entropy(
        model_key=model_key,
        question_text=question_text,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )


def judge_raw(model_key: str, prompt: str) -> str:
    """Raw deterministic completion from a judge model."""
    eng = active_engine()
    if eng == "api":
        from api_engine import judge

        return judge(model_key, prompt)
    if eng == "gguf":
        from gguf_engine import judge

        return judge(model_key, prompt)
    from estimator import verbalized_estimate

    return verbalized_estimate(model_key, prompt)["raw"]


def estimate(model_key: str, prompt: str) -> dict:
    """Numeric verbalized estimate: ``{"raw": str, "value": float|None}``."""
    from estimator import parse_estimate

    raw = judge_raw(model_key, prompt)
    return {"raw": raw, "value": parse_estimate(raw)}
