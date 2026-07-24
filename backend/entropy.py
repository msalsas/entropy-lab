"""Generation with per-token entropy recording.

The entropy measured here is the *true* entropy of the full-vocabulary
next-token distribution at every generated position, computed from the
model logits. This is the quantity the models will later be asked to
estimate verbally, and it is only observable from outside the model.
"""

import math
import torch

try:  # transformers stack is optional when only GGUF/API engines are used
    from models import load_model
except ImportError:  # pragma: no cover - depends on environment
    load_model = None


def _distribution_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy (nats) of a full-vocabulary logits vector.

    The scores returned by ``generate`` are the *processed* logits (after
    temperature scaling and top-p filtering), i.e. the actual sampling
    distribution. Filtered tokens carry ``-inf`` logits, so they are
    masked out before the softmax to avoid ``0 * -inf`` NaNs.
    """
    logits = logits.float()
    logits = logits[torch.isfinite(logits)]
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return float(-(p * logp).sum().item())


@torch.no_grad()
def generate_with_entropy(
    model_key: str,
    question_text: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 48,
    seed: int = 0,
    system_prompt: str | None = None,
) -> dict:
    """Generate an answer and record the entropy of every sampled step.

    Returns a dict with the answer text, the list of per-token entropies
    (nats), their mean, and the sampling parameters actually used.
    """
    tokenizer, model = load_model(model_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_text})

    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )

    torch.manual_seed(seed)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        output_scores=True,
        return_dict_in_generate=True,
    )

    prompt_len = enc["input_ids"].shape[1]
    new_ids = out.sequences[0][prompt_len:]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)

    token_entropies = [_distribution_entropy(step[0]) for step in out.scores]
    mean_entropy = (
        sum(token_entropies) / len(token_entropies) if token_entropies else math.nan
    )

    return {
        "model_key": model_key,
        "question": question_text,
        "answer": answer,
        "n_tokens": len(token_entropies),
        "token_entropies": token_entropies,
        "mean_entropy": mean_entropy,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }
