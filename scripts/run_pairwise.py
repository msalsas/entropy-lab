"""Pairwise (comparative) elicitation run.

For each divergent question and each generator we produce two answers at
a low and a high hidden temperature, then ask the generator itself
(self) and the other model (cross) which answer involved more
unpredictable word choices. Presentation order is randomized per trial
to cancel position bias. The correct answer is known from the true
entropies, so performance is accuracy against 50% chance.

Usage:
    python scripts/run_pairwise.py [--questions 8] [--reps 3]
                                   [--temp-low 0.6] [--temp-high 1.2]

Output: results/pairwise_run.json (copied to
frontend/pairwise_results.json for the standalone frontend).
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from entropy import generate_with_entropy
from experiment import run_pairwise_trial, analyze_pairwise
from models import MODEL_REGISTRY, unload_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

MAX_NEW_TOKENS = 48


def load_questions(n: int) -> list[dict]:
    path = os.path.join(RESULTS_DIR, "selected_questions.json")
    with open(path, encoding="utf-8") as fh:
        all_gaps = json.load(fh)["all_gaps"]
    rows = [r for r in all_gaps if r["sign_consistent"]]
    rows.sort(key=lambda r: abs(r["mean_gap"]), reverse=True)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--temp-low", type=float, default=0.6)
    parser.add_argument("--temp-high", type=float, default=1.2)
    args = parser.parse_args()

    selected = load_questions(args.questions)
    if not selected:
        raise SystemExit("Run scripts/phase0_select.py first.")

    rng = random.Random(11)
    trials = []
    n_pairs = len(selected) * len(MODEL_REGISTRY) * args.reps
    total = n_pairs * 2  # one self + one cross judgment per pair
    done = 0
    t_start = time.time()

    for question in selected:
        for generator in MODEL_REGISTRY:
            for rep in range(args.reps):
                seed = rng.randint(0, 2**31 - 1)
                t0 = time.time()
                gen_low = generate_with_entropy(
                    model_key=generator,
                    question_text=question["text"],
                    temperature=args.temp_low,
                    seed=seed,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                gen_high = generate_with_entropy(
                    model_key=generator,
                    question_text=question["text"],
                    temperature=args.temp_high,
                    seed=seed + 1,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                high_is_a = rng.random() < 0.5
                for relation in ["self", "cross"]:
                    trial = run_pairwise_trial(
                        question=question,
                        generator_key=generator,
                        relation=relation,
                        gen_low=gen_low,
                        gen_high=gen_high,
                        high_is_a=high_is_a,
                        reveal_identity=False,
                    )
                    trials.append(trial)
                    done += 1
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (total - done)
                    print(
                        f"[{done}/{total}] {question['id']} gen={generator} "
                        f"{relation} gap={trial['entropy_gap']:+.2f} "
                        f"choice={trial['choice']} ok={trial['correct']} "
                        f"(eta {eta/60:.0f}m)",
                        flush=True,
                    )
                    _save({"trials": trials, "status": "running"})

    output = {
        "status": "done",
        "config": {
            "question_ids": [q["id"] for q in selected],
            "reps": args.reps,
            "temp_low": args.temp_low,
            "temp_high": args.temp_high,
            "max_new_tokens": MAX_NEW_TOKENS,
            "reveal_identity": False,
        },
        "trials": trials,
        "analysis": analyze_pairwise(trials),
    }
    _save(output)
    shutil.copy(
        os.path.join(RESULTS_DIR, "pairwise_run.json"),
        os.path.join(FRONTEND_DIR, "pairwise_results.json"),
    )
    print("\nPairwise analysis:", flush=True)
    for relation, stats in output["analysis"]["by_relation"].items():
        print(
            f"  {relation:5s} n={stats['n']} acc={stats['accuracy']:.3f} "
            f"binom_p={stats['binom_p']:.4f} "
            f"order={stats['by_order']} gap={stats['by_gap']}",
            flush=True,
        )
    comp = output["analysis"]["self_vs_cross"]
    print(f"  self vs cross: z={comp['z']}, p={comp['p']}", flush=True)


def _save(data) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "pairwise_run.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
