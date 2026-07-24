"""GGUF engine: 3B-class models via llama.cpp with exact entropy.

Two models from different families, quantized to Q4_K_M so they fit in
this machine's RAM (one loaded at a time):

- qwen3b : Qwen2.5-3B-Instruct (strongly multilingual)
- phi35  : Phi-3.5-mini-instruct (English-centric)

Generation is done token by token with manual sampling so that the
*full-vocabulary* distribution is available at every step. The recorded
entropy is the Shannon entropy of the actual sampling distribution
(after temperature scaling and top-p filtering), matching the
methodology of the transformers-based engine.
"""

import gc
import math
import os

import numpy as np

MODEL_REGISTRY_GGUF = {
    "qwen3b": {
        "label": "Qwen2.5-3B-Instruct",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "template": "chatml",
        "stop": ["<|im_end|>"],
        # Known special-token ids (GGUF tokenize() may not map the
        # literal string to the special id).
        "stop_id_fallback": [151645, 151643],
    },
    "phi35": {
        "label": "Phi-3.5-mini-instruct",
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "template": "phi3",
        "stop": ["<|end|>", "<|endoftext|>"],
        "stop_id_fallback": [32007, 32000],
    },
    "smollm17b": {
        # Same family ladder as the small-model run (SmolLM2 360M ->
        # 1.7B), so results are directly comparable across scales.
        "label": "SmolLM2-1.7B-Instruct",
        "repo": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "file": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "template": "chatml",
        "stop": ["<|im_end|>"],
        "stop_id_fallback": [],
    },
}

# NOTE: llama-cpp-python only fills its scores buffer when
# logits_all=True (with logits_all=False eval() saves nothing). The
# buffer is (n_ctx, vocab) float32, so n_ctx is kept small to bound RAM:
# 512 x 152k x 4B ~ 311 MB for Qwen.
_CTX = 512
_current = {}


def _format_prompt(template: str, user_text: str) -> str:
    if template == "chatml":
        return (
            "<|im_start|>user\n" + user_text + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if template == "phi3":
        return "<|user|>\n" + user_text + "<|end|>\n<|assistant|>\n"
    raise ValueError(f"unknown template {template}")


def load_gguf(model_key: str):
    """Load a GGUF model (only one resident at a time: RAM is tight)."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    unload_gguf()
    info = MODEL_REGISTRY_GGUF[model_key]
    path = hf_hub_download(info["repo"], info["file"])
    llm = Llama(
        model_path=path,
        n_ctx=_CTX,
        n_threads=os.cpu_count() or 2,
        logits_all=True,
        verbose=False,
    )
    stop_ids = set(info.get("stop_id_fallback", []))
    for s in info["stop"]:
        ids = llm.tokenize(s.encode(), add_bos=False)
        if len(ids) == 1:
            stop_ids.add(ids[0])
    stop_ids.add(llm.token_eos())
    _current["llm"] = llm
    _current["key"] = model_key
    _current["stop_ids"] = stop_ids
    return llm


def unload_gguf() -> None:
    if "llm" in _current:
        del _current["llm"]
        _current.clear()
        gc.collect()


def _sample_step(logits: np.ndarray, temperature: float, top_p: float,
                 rng: np.random.Generator):
    """Sample one token and return (token_id, entropy of sampling dist)."""
    logits = logits.astype(np.float64) / max(temperature, 1e-5)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()

    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    keep = np.ones(len(p), dtype=bool)
    # Keep tokens until cumulative mass reaches top_p (inclusive).
    cut = int(np.searchsorted(cum, top_p)) + 1
    keep[order[cut:]] = False
    p_f = np.where(keep, p, 0.0)
    p_f /= p_f.sum()

    entropy = float(-(p_f[p_f > 0] * np.log(p_f[p_f > 0])).sum())
    token = int(rng.choice(len(p_f), p=p_f))
    return token, entropy


def generate(model_key: str, question_text: str, temperature: float,
             seed: int, max_new_tokens: int = 48, top_p: float = 0.95,
             prompt_override: str | None = None) -> dict:
    """Generate an answer token by token, recording per-step entropy."""
    assert _current.get("key") == model_key, "load the model first"
    llm = _current["llm"]
    info = MODEL_REGISTRY_GGUF[model_key]

    prompt = prompt_override or _format_prompt(info["template"], question_text)
    tokens = list(llm.tokenize(prompt.encode(), add_bos=True))
    rng = np.random.default_rng(seed)

    llm.reset()
    llm.eval(tokens)

    entropies = []
    out_ids = []
    for _ in range(max_new_tokens):
        # Valid next-token logits live at row n_tokens-1 of the buffer.
        logits = np.array(llm.scores[llm.n_tokens - 1], dtype=np.float64, copy=True)
        token, entropy = _sample_step(logits, temperature, top_p, rng)
        if token in _current["stop_ids"]:
            break
        entropies.append(entropy)
        out_ids.append(token)
        llm.eval([token])

    answer = llm.detokenize(out_ids).decode("utf-8", errors="replace")
    for s in info["stop"]:
        answer = answer.replace(s, "")
    answer = answer.strip()
    return {
        "model_key": model_key,
        "question": question_text,
        "answer": answer,
        "n_tokens": len(entropies),
        "token_entropies": entropies,
        "mean_entropy": sum(entropies) / len(entropies) if entropies else math.nan,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }


def judge(model_key: str, user_text: str, max_new_tokens: int = 8) -> str:
    """Greedy one-letter/short judgment from the given model."""
    result = generate(
        model_key=model_key,
        question_text=user_text,
        temperature=0.01,
        seed=0,
        max_new_tokens=max_new_tokens,
        top_p=1.0,
    )
    return result["answer"]
