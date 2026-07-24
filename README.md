# Entropy Self-Estimation Lab

[![tests](https://github.com/msalsas/entropy-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/msalsas/entropy-lab/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Can a language model estimate the true entropy of its own sampling
distribution, beyond what is inferable from visible text? This project
runs that experiment end to end and exposes it through a web interface.
It works with two small local models out of the box, and scales up to
remote frontier models through any OpenAI-compatible server (LM Studio,
OpenRouter, ...).

Default local pair:

- **Model A**: `Qwen/Qwen2.5-0.5B-Instruct` (strongly multilingual)
- **Model B**: `HuggingFaceTB/SmolLM2-360M-Instruct` (English-centric)

The *true* entropy is the Shannon entropy of the full-vocabulary
next-token distribution at every generated position, computed from the
model logits during sampling. The *verbalized estimate* is a 0-10
unpredictability rating produced by a model in natural language.

## Experimental design

Four controls make a null result interpretable:

1. **Textual baseline** (`cross`): another model estimates the entropy
   given the same visible information (question + answer). A `self`
   advantage only counts above this baseline.
2. **Identity control**: by default the estimator is never told which
   model generated the answer, blocking trained knowledge about a
   specific model's typical behavior (`reveal_identity` toggles this).
3. **Temporal control** (`pre`): the estimate is requested *before* the
   answer exists, so it cannot be read off the generated text.
4. **Hidden temperature**: the sampling temperature is drawn uniformly
   per trial and never disclosed to the estimator.

Primary endpoint per condition: Pearson correlation between verbalized
estimates and true entropy, reported with a 95% Fisher CI and p-value,
plus Spearman r as a robustness check. The decisive comparisons are
`self` vs `cross` within each timing condition, tested with a Fisher z
difference test. Question-level correlations (one aggregated point per
question) are reported alongside trial-level ones to guard against
pseudoreplication within questions. Estimation prompts use shared scale
anchors (factual ~1, creative ~8, decimals allowed) so all conditions
use the same yardstick.

## Phase 0: divergent question selection

Divergence is a property of (question x model A x model B), not of the
model pair. `scripts/phase0_select.py` profiles the whole candidate pool
with both models at two temperature/seed combinations and selects the
questions where the entropy gap is large and sign-consistent:

```bash
python scripts/phase0_select.py
# writes results/phase0_profile.json and results/selected_questions.json
```

## Running the web app

```bash
pip install -r backend/requirements.txt
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

The frontend also works standalone in demo mode: open
`frontend/index.html` and it renders the bundled validation results
(`frontend/demo_results.json`) without a backend.

## Pairwise (comparative) elicitation

Absolute numeric estimates compress badly in small models, so the lab
includes a comparative-judgment paradigm: the judge sees two answers to
the same question (one generated at a low and one at a high hidden
temperature, presentation order randomized) and must pick the one with
more unpredictable word choices. The correct answer is objectively
known from the true entropies, so performance is accuracy against 50%
chance (exact binomial test), with a two-proportion z test for the
self vs cross difference and breakdowns by presentation order
(position-bias check) and entropy-gap size.

```bash
python scripts/run_pairwise.py --questions 8 --reps 3
# writes results/pairwise_run.json and frontend/pairwise_results.json
```

## Scale ladder: 3B-class models (GGUF)

`scripts/run_pairwise_gguf.py` repeats the comparative paradigm with the
same two families scaled up (Qwen2.5 0.5B -> 3B, SmolLM2 360M -> 1.7B),
quantized to Q4_K_M via llama.cpp so they fit in 4 GB of RAM. Generation
is token-by-token with manual sampling so the full-vocabulary sampling
distribution (and its exact entropy) is recorded at every step. Because
RAM only fits one model at a time, the run is phased (all generations
for one model, then the other, then judgments) and persists generations
to disk for resumability. A per-question divergence re-check for the new
pair is included in the output.

## Remote models (LM Studio / OpenAI-compatible)

The whole lab, web app included, can run against any OpenAI-compatible
server, e.g. LM Studio on another machine in your local network. The
active engine is chosen with `LAB_ENGINE` in `.env`:

| `LAB_ENGINE` | Backend | Entropy |
|---|---|---|
| `local` (default) | transformers, 0.5B models on this machine | exact, full vocabulary |
| `gguf` | llama.cpp GGUF, 3B-class models on this machine | exact, full vocabulary |
| `api` | remote OpenAI-compatible server | top-k approximation |

```bash
cp .env.example .env   # then edit:
# LAB_ENGINE=api
# LAB_API_BASE=http://192.168.1.50:1234/v1
# LAB_MODEL_A=qwen2.5-7b-instruct
# LAB_MODEL_B=mistral-7b-instruct-v0.3
```

Configuration priority: built-in defaults < `.env` < real environment
variables. In LM Studio: load both models, start the local server
(Developer tab), and make sure the model ids in `.env` match what
`GET /models` reports. Then start the web app as usual. The header
shows a badge with the active engine and the remote endpoint, and the
model labels come from `.env`:

```bash
cd backend && uvicorn app:app --host 0.0.0.0 --port 8000
```

The headless pairwise runner uses the same configuration:

```bash
python scripts/run_pairwise_api.py --questions 8 --reps 3
```

One warning about question selection: `selected_questions.json` is a
generated profiling artifact, not part of the repository (it lives in
the gitignored `results/` directory). If it is missing, the runner
falls back to the full question pool with a warning. Either way, for a
remote model pair you should verify divergence yourself: the bundled
profile was measured on the 0.5B pair, and divergence is not guaranteed
to transfer. In the UI, choose the "Full pool" radio button to bypass
the profiled subset.

Entropy note: remote APIs only return top-k token logprobs, so true
full-vocabulary entropy is approximated over the renormalized top-k
distribution (top-20 by default). Every trial stores `topk_mass_*`
fields with the probability mass covered, so approximation quality is
auditable; treat trials with mean mass below ~0.90 with caution. Local
runs (transformers/GGUF engines) use exact full-vocabulary entropy.

### Frontier models via OpenRouter

OpenRouter routes each request to one of several providers, and many of
them silently ignore `logprobs`. The entropy then comes out NaN with no
error message. Two defenses, both in `.env`:

```
LAB_REQUIRE_PARAMETERS=true   # only use providers that support logprobs
LAB_TEMP_LOW=0.2              # confident frontier models need wide
LAB_TEMP_HIGH=2.0             # temperature extremes to move entropy
```

Coverage is model- and provider-dependent. A 2026 analysis of
OpenRouter endpoints found logprobs returned on roughly a quarter of
them, and some providers cap `top_logprobs` (5 instead of 20, for
example), so verify your pair with a single API call before a run and
keep `LAB_TOP_LOGPROBS` identical for both models; an asymmetric
approximation would bias the comparison. One pair we ran end to end:
`meta-llama/llama-3.3-70b-instruct` and `inclusionai/ling-2.6-1t`,
with top-20 mass around 0.99. Anthropic and the Kimi Code endpoint do
not expose logprobs at all: usable as judges, not as generators.
Pairwise trials store the raw answer texts (`answer_low`/`answer_high`)
so degeneracy analyses are possible post-hoc.

## Gradient-inversion attack

Cross judges showed *worse* accuracy on large entropy gaps. The attack
(`scripts/compare_rejudge.py` over cached generations) showed big gaps
are produced by degenerate (broken) high-temperature text, and judges
read broken text as defective rather than unpredictable: 11% accuracy
on degenerate pairs vs 64% on clean ones. `scripts/run_rejudge.py`
re-judges the same pairs with an entropy-literate prompt that defines
broken text as the signature of maximum randomness; cross accuracy on
degenerate pairs jumps to 56%, and the self model's 100% on its own
degenerate answers collapses to 50%. Conclusion: the inversion was a
conceptual mismatch (verbalized "unpredictability" differs from
sampling entropy), not privileged access.

## Data availability

All results behind the headline table ship in [`data/`](data/): the
full trial records (generations, entropies, judgments) for every local
scale, plus the gradient-inversion attack data, each file
self-describing (configuration + trials + analysis). Remote runs are
added there as they are produced. The copies under `frontend/` are the
same data, kept so the UI can render its demo without a backend.
`results/` stays gitignored as a working directory; `data/` is the
published record.

## Tests

```bash
pip install pytest
python -m pytest            # from the project root
```

The suite covers the statistical core (Pearson/Spearman, Fisher
CI/p/difference tests, exact binomial, proportion tests), prompt
construction and response parsing, the exact-entropy computation
(including the -inf masking from top-p filtering), the GGUF sampler
(temperature/top-p behavior, entropy correctness, seed determinism),
the remote API engine (.env priority, top-k entropy approximation,
strict provider routing, mocked HTTP), the engine dispatch layer
(LAB_ENGINE selection, registry resolution, api-mode trial routing),
an end-to-end integration test against a mock OpenAI-compatible server,
and question-pool integrity. Model loading and HTTP calls are fully
mocked, so the suite runs in seconds without GPU, models or network;
modules whose optional dependency is absent
(torch/transformers/llama-cpp-python) are skipped cleanly.

## Headless validation run

```bash
python scripts/run_demo.py
# writes results/demo_run.json (copied to frontend/demo_results.json)
```

## Project layout

```
backend/
  models.py       model registry and lazy loading (transformers)
  entropy.py      generation with per-token true-entropy recording
  estimator.py    verbalized-estimation prompts and parsing
  experiment.py   trial runner, conditions, correlation analysis
  questions.py    candidate question pool (ES/EN)
  gguf_engine.py  local 3B-class models via llama.cpp (exact entropy)
  api_engine.py   remote OpenAI-compatible engine (.env-configured)
  engines.py      engine dispatch: routes generation/judgment to the
                  active backend selected with LAB_ENGINE
  app.py          FastAPI backend (API + static frontend)
frontend/
  index.html      control panel and charts
  app.js          run/poll/render logic, demo-mode fallback
  style.css
data/
  README.md           dataset descriptions and trial schema
  local_*.json        published trial-level results (see data/README.md)
scripts/
  phase0_select.py    divergent-question profiling and selection
  run_demo.py         headless end-to-end validation run
  run_pairwise.py     pairwise paradigm (0.5B, transformers)
  run_pairwise_gguf.py  pairwise paradigm (3B-class, GGUF)
  run_pairwise_api.py  pairwise paradigm (remote: LM Studio, OpenRouter)
  run_rejudge.py      entropy-literate re-judging of cached pairs
  compare_rejudge.py  gradient-inversion analysis
tests/                pytest suite (mocked models/network, runs in seconds)
.env.example          template for the remote-engine configuration
results/              working directory for run artifacts (gitignored)
```

## Headline results

A null result, consistent across four scales and two paradigms:

| Scale (engines) | Paradigm | Outcome |
|---|---|---|
| 0.5B / 360M (local, exact entropy) | absolute + pairwise | self-correlation *negative* (r = -0.27); pairwise at chance (52-56%) |
| 3B / 1.7B (local GGUF, exact) | pairwise | chance (51-54%) |
| 8B / 12B (LM Studio, top-k) | absolute + pairwise | self ≈ cross (both r ≈ 0.5; diff. p = 0.85); pairwise diff. p = 0.31 |
| 70B / 1T MoE (OpenRouter, top-k) | pairwise | cross (71%) ≥ self (67%); diff. p = 0.66 |

The clearest pattern appears at frontier scale: judges detect entropy
gaps in proportion to how readable they are in the text. On 2-nat gaps
both self and cross reach ~85% accuracy; on 0.5-nat gaps both sit at
chance. The introspection hypothesis predicts a self advantage exactly
where the text is silent; there, self performed at chance (46%) and
slightly below cross.

Secondary findings:

- At 8-12B both models track true entropy from visible text
  (r ≈ 0.5, p ≈ 1e-4), something the 0.5B models cannot do at all, but
  they do it identically in self and cross conditions. The information
  is in the text; nothing privileged is required to explain it.
- Position bias replicated in three independent setups: uncertain
  judges default to answer "A" (up to 63% vs 36% accuracy by
  presentation order). Any LLM-judge design without order
  randomization will inherit this.
- Gradient inversion: judges do *worse* on larger entropy gaps when
  those gaps come from degenerate high-temperature text, because
  broken reads as defective rather than unpredictable. An
  entropy-literate prompt partially rescues cross judgments and
  collapses the apparent self advantage.
- One cautionary data point: a promising pairwise result
  (68.75%, p = 0.013, n = 48) dissolved when the sample was doubled
  (57%, p = 0.18). Every number reported here survived a power
  increase.

Across all four scales, up to a 1T-total-parameter MoE, we find no
evidence of privileged introspective access to sampling entropy.

## Contributing, license, citation

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for
the ground rules. The code is MIT-licensed ([LICENSE](LICENSE)). If you
use the lab in academic work, citation metadata is in
[CITATION.cff](CITATION.cff).
