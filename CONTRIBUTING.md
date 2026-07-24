# Contributing

Contributions are welcome. The most valuable directions right now:

- **Frontier-scale arm**: run the pairwise paradigm against larger models
  via the remote engine (`api_engine.py` + `.env`).
- **Larger n**: more trials per condition to tighten the confidence
  intervals on the cross/clean signal.
- **Human baseline**: collect human judgments on the same pairs to anchor
  the "textual baseline" comparison.
- **New engines**: any backend that exposes per-token logits or top-k
  logprobs can be added following `gguf_engine.py` / `api_engine.py`.

## Ground rules

1. **Code in English** (identifiers, comments, docs).
2. **Run the tests before opening a PR**: `python -m pytest` must pass.
   Add tests for new behavior — the suite runs in ~2 s and mocks all
   model/network access.
3. **Never commit `.env` or `results/`** (see `.gitignore`). Put new
   configuration knobs in `.env.example` with sensible defaults.
4. Keep experimental controls intact: hidden identity by default, hidden
   temperature, order randomization in pairwise judgments. If you change
   any of these, document it prominently — they are what makes the null
   result interpretable.

## Setup

```bash
pip install -r backend/requirements.txt
python -m pytest
```
