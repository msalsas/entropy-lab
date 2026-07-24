# Experimental data

All results reported in the README's headline table ship in this
directory. Every file is self-contained: configuration, per-trial
records and analysis, so each number in the table is recomputable from
the corresponding JSON.

| File | Scale | Paradigm | Contents |
|---|---|---|---|
| `local_0.5b_absolute.json` | Qwen2.5-0.5B / SmolLM2-360M (local, exact entropy) | absolute | 240 trials: hidden-temperature generations with true entropy and verbalized estimates across self/cross x pre/post |
| `local_0.5b_pairwise.json` | same | pairwise | 96 comparative judgments (accuracy, order, gap breakdowns) |
| `local_3b_pairwise.json` | Qwen2.5-3B / SmolLM2-1.7B (local GGUF, exact entropy) | pairwise | 96 comparative judgments |
| `local_gradient_inversion.json` | 3B pair | re-judging attack | original vs entropy-literate judgments on clean/degenerate pairs |
| `remote_lmstudio_8b_absolute.json` | Hermes-3-8B / Mistral-Nemo-12B (LM Studio, top-k) | absolute | 192 trials, full pool |
| `remote_lmstudio_8b_pairwise.json` | same | pairwise | 192 judgments, temps 0.2/2.0 |
| `remote_openrouter_70b_pairwise.json` | Llama-3.3-70B / Ling-2.6-1T (OpenRouter, top-k) | pairwise | 96 judgments, temps 0.2/2.0 |

New remote runs are added here as they are produced, following the
naming pattern `remote_<backend>_<scale>_<paradigm>.json`.

## Trial record schema (pairwise)

`question_id`, `question`, `generator`, `estimator`, `relation`
(self/cross), `temp_low`/`temp_high`, `entropy_low`/`entropy_high`/`gap`,
`high_is_a` (presentation order), `choice`/`choice_raw`/`correct`, and
for remote runs `topk_mass_low`/`topk_mass_high` (approximation audit)
plus `answer_low`/`answer_high` (raw texts, for degeneracy analyses).

Absolute-paradigm trials carry `temperature`, `true_entropy`,
`estimate`, `timing` and `reveal_identity` instead.

## Adding your own runs

Runs write to the gitignored `results/` directory. To publish one, copy
it here:

```bash
cp results/pairwise_api_run.json data/remote_openrouter_70b_pairwise.json
```

The duplicates of the local files under `frontend/` are the same data,
kept there so the UI can render its demo without a backend; they are
not independent results.
