"""Remote engine: OpenAI-compatible servers (LM Studio, vLLM, OpenAI...).

Talks to any server exposing the chat-completions API with logprobs
support. All connection and model parameters live in a ``.env`` file
(project root or backend/) or in real environment variables, so no code
change is needed to point the lab at a different machine or model pair:

    LAB_API_BASE=http://192.168.1.50:1234/v1
    LAB_API_KEY=not-needed            # LM Studio ignores it; OpenAI needs it
    LAB_MODEL_A=qwen2.5-7b-instruct
    LAB_MODEL_A_LABEL=Qwen2.5-7B-Instruct
    LAB_MODEL_B=mistral-7b-instruct
    LAB_MODEL_B_LABEL=Mistral-7B-Instruct
    LAB_TOP_LOGPROBS=20

Entropy note: APIs do not return the full-vocabulary distribution, only
the top-k token logprobs. Entropy is therefore approximated over the
renormalized top-k mass (with the default sampling parameters, top-20
usually covers well above 95% of the probability mass). Each generation
records ``topk_mass_mean`` so approximation quality is auditable; trials
below 0.90 mean mass should be treated with caution.
"""

import json
import math
import os
import urllib.request

ENV_LOCATIONS = [
    os.path.join(os.path.dirname(__file__), ".env"),
    os.path.join(os.path.dirname(__file__), "..", ".env"),
]

DEFAULTS = {
    # Active engine for the web app and scripts: local | gguf | api.
    "LAB_ENGINE": "local",
    "LAB_API_BASE": "http://localhost:1234/v1",
    "LAB_API_KEY": "not-needed",
    "LAB_MODEL_A": "model-a",
    "LAB_MODEL_A_LABEL": "Model A",
    "LAB_MODEL_B": "model-b",
    "LAB_MODEL_B_LABEL": "Model B",
    "LAB_TOP_LOGPROBS": "20",
    # Set to "true" for aggregators like OpenRouter: only route to
    # providers that support every request parameter (logprobs!). Without
    # it, some requests silently drop logprobs and entropy comes out NaN.
    "LAB_REQUIRE_PARAMETERS": "false",
    # Temperature extremes used by the pairwise scripts (low vs high).
    "LAB_TEMP_LOW": "0.6",
    "LAB_TEMP_HIGH": "1.2",
}

_config: dict | None = None


def _load_env_file() -> dict:
    """Minimal .env parser (real environment variables take priority)."""
    file_vars = {}
    for path in ENV_LOCATIONS:
        path = os.path.normpath(path)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                file_vars[key.strip()] = value.strip().strip('"').strip("'")
    return file_vars


def get_config() -> dict:
    """Merged config: defaults < .env file < real environment."""
    global _config
    if _config is None:
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in _load_env_file().items() if k in DEFAULTS})
        merged.update({k: os.environ[k] for k in DEFAULTS if k in os.environ})
        _config = merged
    return _config


def model_registry() -> dict:
    """Model A/B registry built entirely from configuration."""
    cfg = get_config()
    return {
        "model_a": {"label": cfg["LAB_MODEL_A_LABEL"], "remote_id": cfg["LAB_MODEL_A"]},
        "model_b": {"label": cfg["LAB_MODEL_B_LABEL"], "remote_id": cfg["LAB_MODEL_B"]},
    }


def _post(endpoint: str, payload: dict) -> dict:
    cfg = get_config()
    url = cfg["LAB_API_BASE"].rstrip("/") + endpoint
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['LAB_API_KEY']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def _get(endpoint: str) -> dict:
    cfg = get_config()
    url = cfg["LAB_API_BASE"].rstrip("/") + endpoint
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {cfg['LAB_API_KEY']}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def list_remote_models() -> list[str]:
    """Model ids currently served (useful to check the .env matches)."""
    data = _get("/models")
    return [m.get("id", "?") for m in data.get("data", [])]


def _entropy_from_top_logprobs(top_logprobs: list[dict]) -> tuple[float, float]:
    """Approximate entropy over the renormalized top-k distribution.

    Returns (entropy_nats, top_k_probability_mass).
    """
    if not top_logprobs:
        return math.nan, 0.0
    probs = [math.exp(t["logprob"]) for t in top_logprobs]
    mass = sum(probs)
    if mass <= 0:
        return math.nan, 0.0
    # Entropy of the truncated (renormalized) top-k distribution.
    entropy = -sum((p / mass) * math.log(p / mass) for p in probs if p > 0)
    return entropy, mass


def generate(model_key: str, question_text: str, temperature: float,
             seed: int, max_new_tokens: int = 48, top_p: float = 0.95,
             prompt_override: str | None = None,
             logprobs: bool = True) -> dict:
    """Generate via the remote API, recording approximate per-token entropy."""
    registry = model_registry()
    info = registry[model_key]
    cfg = get_config()

    payload = {
        "model": info["remote_id"],
        "messages": [{"role": "user", "content": prompt_override or question_text}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "seed": seed,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(cfg["LAB_TOP_LOGPROBS"])
    if cfg["LAB_REQUIRE_PARAMETERS"].strip().lower() in ("1", "true", "yes"):
        payload["provider"] = {"require_parameters": True}

    data = _post("/chat/completions", payload)
    choice = data["choices"][0]
    answer = choice["message"]["content"] or ""

    entropies, masses = [], []
    lp = choice.get("logprobs") or {}
    for token_info in lp.get("content") or []:
        entropy, mass = _entropy_from_top_logprobs(token_info.get("top_logprobs") or [])
        if not math.isnan(entropy):
            entropies.append(entropy)
            masses.append(mass)

    return {
        "model_key": model_key,
        "question": question_text,
        "answer": answer.strip(),
        "n_tokens": len(entropies),
        "token_entropies": entropies,
        "mean_entropy": sum(entropies) / len(entropies) if entropies else math.nan,
        "topk_mass_mean": sum(masses) / len(masses) if masses else None,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }


def judge(model_key: str, user_text: str, max_new_tokens: int = 8) -> str:
    """Greedy short judgment from the given remote model."""
    result = generate(
        model_key=model_key,
        question_text=user_text,
        temperature=0.0,
        seed=0,
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        logprobs=False,
    )
    return result["answer"]
