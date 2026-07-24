"""Model registry and lazy loading.

Two small instruction-tuned models from different families are used as
model A and model B. Qwen2.5 is strongly multilingual; SmolLM2 is more
English-centric, which maximizes the chance of finding questions where
their next-token entropy profiles diverge.
"""

import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(os.cpu_count() or 4)

MODEL_REGISTRY = {
    "qwen": {
        "hf_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen2.5-0.5B-Instruct",
        "family": "qwen",
    },
    "smollm": {
        "hf_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "label": "SmolLM2-360M-Instruct",
        "family": "smollm",
    },
}

_cache = {}


def load_model(model_key: str):
    """Load (and cache) tokenizer + model for the given registry key."""
    if model_key in _cache:
        return _cache[model_key]
    info = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(info["hf_name"])
    model = AutoModelForCausalLM.from_pretrained(info["hf_name"], torch_dtype="auto")
    model.eval()
    _cache[model_key] = (tokenizer, model)
    return _cache[model_key]


def unload_model(model_key: str) -> None:
    """Drop a model from the cache to free RAM."""
    if model_key in _cache:
        del _cache[model_key]
        gc.collect()
