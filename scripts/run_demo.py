"""Headless end-to-end experiment run.

Uses the most divergent sign-consistent questions found in phase 0 and
executes the full grid: both generators x both relations x both timings
x one hidden temperature per unit. The generation of each (question,
generator, temperature, seed) unit is shared across its four estimation
conditions, exactly as the API runner does.

Usage:
    python scripts/run_demo.py [--questions 10] [--temps 0.6 0.9 1.2]
                               [--output demo_run]

Output: results/<output>.json (also copied to frontend/demo_results.json
when --output is demo_run, so the standalone frontend can render it).
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from experiment import run_trial, analyze_trials
from models import MODEL_REGISTRY

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

MAX_NEW_TOKENS = 48


def load_questions(n: int) -> list[dict]:
    """Top-n divergent questions by absolute sign-consistent gap."""
    path = os.path.join(RESULTS_DIR, "selected_questions.json")
    with open(path, encoding="utf-8") as fh:
        all_gaps = json.load(fh)["all_gaps"]
    rows = [r for r in all_gaps if r["sign_consistent"]]
    rows.sort(key=lambda r: abs(r["mean_gap"]), reverse=True)
    return rows[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--temps", type=float, nargs="+", default=[0.7, 1.1])
    parser.add_argument("--output", default="demo_run")
    args = parser.parse_args()

    selected = load_questions(args.questions)
    if not selected:
        raise SystemExit("Run scripts/phase0_select.py first.")

    trials = []
    rng = random.Random(7)
    generation_cache: dict = {}
    total = len(selected) * len(MODEL_REGISTRY) * len(args.temps) * 4
    done = 0
    t_start = time.time()

    for question in selected:
        for generator in MODEL_REGISTRY:
            for temperature in args.temps:
                seed = rng.randint(0, 2**31 - 1)
                for timing in ["pre", "post"]:
                    for relation in ["self", "cross"]:
                        t0 = time.time()
                        trial = run_trial(
                            question=question,
                            generator_key=generator,
                            timing=timing,
                            relation=relation,
                            temperature=temperature,
                            seed=seed,
                            max_new_tokens=MAX_NEW_TOKENS,
                            reveal_identity=False,
                            generation_cache=generation_cache,
                        )
                        trials.append(trial)
                        done += 1
                        elapsed = time.time() - t_start
                        eta = elapsed / done * (total - done)
                        print(
                            f"[{done}/{total}] {question['id']} gen={generator} "
                            f"{relation}-{timing} T={temperature} "
                            f"H={trial['true_entropy']:.3f} est={trial['estimate']} "
                            f"({time.time()-t0:.0f}s, eta {eta/60:.0f}m)",
                            flush=True,
                        )
                        _save({"trials": trials, "status": "running"}, args.output)

    output = {
        "status": "done",
        "config": {
            "question_ids": [q["id"] for q in selected],
            "temperatures": args.temps,
            "max_new_tokens": MAX_NEW_TOKENS,
            "reveal_identity": False,
        },
        "trials": trials,
        "analysis": analyze_trials(trials),
    }
    _save(output, args.output)
    if args.output == "demo_run":
        shutil.copy(
            os.path.join(RESULTS_DIR, f"{args.output}.json"),
            os.path.join(FRONTEND_DIR, "demo_results.json"),
        )
    print("\nAnalysis (pooled):", flush=True)
    for key, stats in output["analysis"]["pooled"].items():
        ci = stats.get("pearson_ci95")
        ci_txt = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "-"
        print(
            f"  {key:12s} n={stats['n']} r={stats['pearson_r']:+.3f} "
            f"ci95={ci_txt} p={stats.get('pearson_p')} "
            f"q-level r={stats['question_level']['pearson_r']}",
            flush=True,
        )
    print("Comparisons:", flush=True)
    for key, comp in output["analysis"]["comparisons"].items():
        print(f"  {key}: z={comp['z']}, p={comp['p']}", flush=True)


def _save(data, name: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
