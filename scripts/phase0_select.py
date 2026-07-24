"""Phase 0: entropy profiling and divergent-question selection.

For every candidate question we generate answers with both models at two
temperature/seed combinations and record the true per-token entropy of
the next-token distribution. A question is selected as "divergent" when
the entropy gap (H_qwen - H_smollm) has a consistent sign across both
repetitions; selection ranks questions by the absolute mean gap.

Outputs:
  results/phase0_profile.json      - raw per-question, per-model profiles
  results/selected_questions.json  - ranked divergent subset
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from questions import QUESTION_POOL
from entropy import generate_with_entropy
from models import unload_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_KEYS = ["qwen", "smollm"]
REPS = [
    {"temperature": 0.7, "seed": 42},
    {"temperature": 1.0, "seed": 1234},
]
MAX_NEW_TOKENS = 48


def main() -> None:
    profile = {q["id"]: {} for q in QUESTION_POOL}

    for model_key in MODEL_KEYS:
        for question in QUESTION_POOL:
            runs = []
            for rep in REPS:
                t0 = time.time()
                result = generate_with_entropy(
                    model_key=model_key,
                    question_text=question["text"],
                    temperature=rep["temperature"],
                    seed=rep["seed"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                runs.append({
                    "temperature": rep["temperature"],
                    "seed": rep["seed"],
                    "mean_entropy": result["mean_entropy"],
                    "n_tokens": result["n_tokens"],
                    "answer": result["answer"],
                })
                print(
                    f"[{model_key}] {question['id']} T={rep['temperature']} "
                    f"H={result['mean_entropy']:.3f} "
                    f"({result['n_tokens']} tok, {time.time()-t0:.0f}s)",
                    flush=True,
                )
            profile[question["id"]][model_key] = runs
            _save(os.path.join(RESULTS_DIR, "phase0_profile.json"), profile)
        unload_model(model_key)

    selected = _select_divergent(profile)
    _save(os.path.join(RESULTS_DIR, "selected_questions.json"), selected)
    print(f"\nSelected {len(selected['questions'])} divergent questions:", flush=True)
    for q in selected["questions"]:
        print(f"  {q['id']:18s} gap={q['mean_gap']:+.3f} nats", flush=True)


def _select_divergent(profile: dict) -> dict:
    rows = []
    for question in QUESTION_POOL:
        qid = question["id"]
        runs_a = profile[qid].get("qwen", [])
        runs_b = profile[qid].get("smollm", [])
        if not runs_a or not runs_b:
            continue
        gaps = [
            ra["mean_entropy"] - rb["mean_entropy"] for ra, rb in zip(runs_a, runs_b)
        ]
        consistent = all(g > 0 for g in gaps) or all(g < 0 for g in gaps)
        rows.append({
            "id": qid,
            "text": question["text"],
            "lang": question["lang"],
            "category": question["category"],
            "gaps": gaps,
            "mean_gap": sum(gaps) / len(gaps),
            "sign_consistent": consistent,
            "mean_entropy": {
                "qwen": sum(r["mean_entropy"] for r in runs_a) / len(runs_a),
                "smollm": sum(r["mean_entropy"] for r in runs_b) / len(runs_b),
            },
        })

    consistent_rows = [r for r in rows if r["sign_consistent"]]
    consistent_rows.sort(key=lambda r: abs(r["mean_gap"]), reverse=True)

    # Balance directions when possible: top gaps in each direction.
    positive = [r for r in consistent_rows if r["mean_gap"] > 0][:5]
    negative = [r for r in consistent_rows if r["mean_gap"] < 0][:5]
    selected = positive + negative
    selected.sort(key=lambda r: abs(r["mean_gap"]), reverse=True)

    return {
        "config": {"reps": REPS, "max_new_tokens": MAX_NEW_TOKENS},
        "all_gaps": rows,
        "questions": selected,
    }


def _save(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
