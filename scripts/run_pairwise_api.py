"""Pairwise (comparative) elicitation against a remote OpenAI-compatible
server (e.g. LM Studio running on another machine in the local network).

Same design as run_pairwise_gguf.py, but generation and judgments go
through the remote API configured in .env (see .env.example). Entropy is
the top-k approximation recorded by api_engine; per-trial approximation
quality (topk_mass_mean) is saved for auditing.

Usage:
    cp .env.example .env   # edit with your server and model ids
    python scripts/run_pairwise_api.py [--questions 8] [--reps 3]

Output: results/pairwise_api_run.json (copied to
frontend/pairwise_api_results.json for the standalone frontend).
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from estimator import build_pairwise_prompt, parse_choice
from experiment import analyze_pairwise
from api_engine import (
    get_config,
    model_registry,
    list_remote_models,
    generate,
    judge,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

MAX_NEW_TOKENS = 48
MODEL_KEYS = ["model_a", "model_b"]

# Temperature extremes from .env (LAB_TEMP_LOW / LAB_TEMP_HIGH). For
# confident frontier models the defaults barely move entropy; 0.2/2.0
# worked well in practice.
TEMP_LOW = float(get_config()["LAB_TEMP_LOW"])
TEMP_HIGH = float(get_config()["LAB_TEMP_HIGH"])


def load_questions(n: int) -> list[dict]:
    """Questions for the run: profiled divergent subset if available,
    otherwise the full pool (with a warning) so the script works on a
    fresh clone."""
    path = os.path.join(RESULTS_DIR, "selected_questions.json")
    if not os.path.exists(path):
        from questions import QUESTION_POOL

        print("WARNING: results/selected_questions.json not found; "
              "using the full question pool (no divergence profiling).",
              flush=True)
        return [dict(q) for q in QUESTION_POOL[:n]]
    with open(path, encoding="utf-8") as fh:
        all_gaps = json.load(fh)["all_gaps"]
    rows = [r for r in all_gaps if r["sign_consistent"]]
    rows.sort(key=lambda r: abs(r["mean_gap"]), reverse=True)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    registry = model_registry()
    print("Remote models served:", list_remote_models(), flush=True)
    print("Using:", {k: v["remote_id"] for k, v in registry.items()}, flush=True)

    selected = load_questions(args.questions)
    if not selected:
        raise SystemExit("Run scripts/phase0_select.py first.")

    rng = random.Random(11)
    units = [
        (q, gen, rep, rng.randint(0, 2**31 - 1), rng.random() < 0.5)
        for q in selected for gen in MODEL_KEYS for rep in range(args.reps)
    ]
    total = len(units) * 2
    done = 0
    t_start = time.time()
    trials = []

    for (q, gen, rep, seed, high_is_a) in units:
        t0 = time.time()
        low = generate(gen, q["text"], TEMP_LOW, seed, MAX_NEW_TOKENS)
        high = generate(gen, q["text"], TEMP_HIGH, seed + 1, MAX_NEW_TOKENS)
        # Empty completions (rare at extreme temperatures) yield NaN
        # entropy; retry with shifted seeds before accepting them.
        for attempt in range(2):
            if low["n_tokens"] > 0 and high["n_tokens"] > 0:
                break
            low = generate(gen, q["text"], TEMP_LOW,
                           seed + 100 + attempt, MAX_NEW_TOKENS)
            high = generate(gen, q["text"], TEMP_HIGH,
                            seed + 200 + attempt, MAX_NEW_TOKENS)
        for relation in ["self", "cross"]:
            judge_key = gen if relation == "self" else (
                "model_b" if gen == "model_a" else "model_a"
            )
            answer_a = high["answer"] if high_is_a else low["answer"]
            answer_b = low["answer"] if high_is_a else high["answer"]
            prompt = build_pairwise_prompt(
                relation=relation,
                question=q["text"],
                answer_a=answer_a,
                answer_b=answer_b,
                generator_key=gen,
                reveal_identity=False,
            )
            raw = judge(judge_key, prompt)
            choice = parse_choice(raw)
            correct_letter = "A" if high_is_a else "B"
            trials.append({
                "question_id": q["id"],
                "question": q["text"],
                "generator": gen,
                "estimator": judge_key,
                "relation": relation,
                "reveal_identity": False,
                "temp_low": TEMP_LOW,
                "temp_high": TEMP_HIGH,
                "entropy_low": low["mean_entropy"],
                "entropy_high": high["mean_entropy"],
                "entropy_gap": high["mean_entropy"] - low["mean_entropy"],
                "topk_mass_low": low["topk_mass_mean"],
                "topk_mass_high": high["topk_mass_mean"],
                # Answer texts are kept so degeneracy (gradient-inversion)
                # analyses and paper examples are possible post-hoc.
                "answer_low": low["answer"],
                "answer_high": high["answer"],
                "high_is_a": high_is_a,
                "choice": choice,
                "choice_raw": raw,
                "correct": (choice == correct_letter) if choice else None,
            })
            done += 1
            eta = (time.time() - t_start) / done * (total - done)
            print(
                f"[{done}/{total}] {q['id']} gen={gen} {relation} "
                f"gap={trials[-1]['entropy_gap']:+.2f} choice={choice} "
                f"ok={trials[-1]['correct']} (eta {eta/60:.0f}m)",
                flush=True,
            )
            _save({"trials": trials, "status": "running"})

    output = {
        "status": "done",
        "scale": "remote",
        "models": {k: v["label"] for k, v in registry.items()},
        "config": {
            "question_ids": [q["id"] for q in selected],
            "reps": args.reps,
            "temp_low": TEMP_LOW,
            "temp_high": TEMP_HIGH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "reveal_identity": False,
            "entropy_note": "top-k logprobs approximation; see topk_mass_* fields",
        },
        "trials": trials,
        "analysis": analyze_pairwise(trials),
    }
    _save(output)
    shutil.copy(
        os.path.join(RESULTS_DIR, "pairwise_api_run.json"),
        os.path.join(FRONTEND_DIR, "pairwise_api_results.json"),
    )
    print("\nRemote pairwise analysis:", flush=True)
    for relation, stats in output["analysis"]["by_relation"].items():
        print(
            f"  {relation:5s} n={stats['n']} acc={stats['accuracy']:.3f} "
            f"binom_p={stats['binom_p']:.4f}",
            flush=True,
        )
    comp = output["analysis"]["self_vs_cross"]
    print(f"  self vs cross: z={comp['z']}, p={comp['p']}", flush=True)


def _save(data) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "pairwise_api_run.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
