"""Pairwise (comparative) elicitation with 3B-class GGUF models.

Same design as run_pairwise.py but with Qwen2.5-3B vs Phi-3.5-mini via
llama.cpp. RAM only fits one model at a time, so the run is organized in
phases: (1) generate all answers for one model, unload; (2) repeat for
the other; (3) load each model once more to produce its judgments (self
judgments of its own answers + cross judgments of the other's answers).

Divergence is re-verified for this model pair: the phase-0 gap of every
selected question is recomputed from the new generations and reported.

Usage:
    python scripts/run_pairwise_gguf.py [--questions 8] [--reps 3]
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
from gguf_engine import MODEL_REGISTRY_GGUF, load_gguf, unload_gguf, generate, judge

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

MAX_NEW_TOKENS = 32
TEMP_LOW, TEMP_HIGH = 0.6, 1.2
# Scale ladder of the original pair: Qwen2.5 0.5B -> 3B,
# SmolLM2 360M -> 1.7B (Phi-3.5-mini was too slow on this CPU).
MODEL_KEYS = ["qwen3b", "smollm17b"]


def load_questions(n: int) -> list[dict]:
    path = os.path.join(RESULTS_DIR, "selected_questions.json")
    with open(path, encoding="utf-8") as fh:
        all_gaps = json.load(fh)["all_gaps"]
    rows = [r for r in all_gaps if r["sign_consistent"]]
    rows.sort(key=lambda r: abs(r["mean_gap"]), reverse=True)
    return rows[:n]


def other(key: str) -> str:
    return next(k for k in MODEL_KEYS if k != key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    selected = load_questions(args.questions)
    if not selected:
        raise SystemExit("Run scripts/phase0_select.py first.")

    rng = random.Random(11)
    units = [
        (q, gen, rep, rng.randint(0, 2**31 - 1), rng.random() < 0.5)
        for q in selected for gen in MODEL_KEYS for rep in range(args.reps)
    ]
    total = len(units)

    # --- Phase 1+2: generate all answers, one model resident at a time.
    # Generations are persisted after each model phase so the run can
    # resume if the judgment phase fails.
    gen_cache_path = os.path.join(RESULTS_DIR, "pairwise_3b_generations.json")
    generations = {}
    if os.path.exists(gen_cache_path):
        with open(gen_cache_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        generations = {tuple(k.split("|")): tuple(v) for k, v in raw.items()}
        generations = {
            (qid, gen, int(rep)): tuple(v) for (qid, gen, rep), v in generations
        }

    for model_key in MODEL_KEYS:
        pending = [u for u in units if u[1] == model_key
                   and (u[0]["id"], u[1], u[2]) not in generations]
        if not pending:
            continue
        load_gguf(model_key)
        for (q, gen, rep, seed, _) in pending:
            t0 = time.time()
            low = generate(model_key, q["text"], TEMP_LOW, seed, MAX_NEW_TOKENS)
            high = generate(model_key, q["text"], TEMP_HIGH, seed + 1, MAX_NEW_TOKENS)
            generations[(q["id"], gen, rep)] = (low, high)
            print(
                f"[gen] {q['id']} {gen} rep{rep} "
                f"H_low={low['mean_entropy']:.2f} H_high={high['mean_entropy']:.2f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
        unload_gguf()
        with open(gen_cache_path, "w", encoding="utf-8") as fh:
            json.dump(
                {f"{qid}|{gen}|{rep}": list(v)
                 for (qid, gen, rep), v in generations.items()},
                fh, ensure_ascii=False,
            )

    # Divergence re-check for this model pair (per question, per temp).
    divergence = {}
    for q in selected:
        gaps = []
        for rep in range(args.reps):
            a = generations[(q["id"], MODEL_KEYS[0], rep)]
            b = generations[(q["id"], MODEL_KEYS[1], rep)]
            gaps.append({
                "low": a[0]["mean_entropy"] - b[0]["mean_entropy"],
                "high": a[1]["mean_entropy"] - b[1]["mean_entropy"],
            })
        divergence[q["id"]] = gaps

    # --- Phase 3: judgments, one judge resident at a time.
    trials = []
    done = 0
    t_start = time.time()
    for judge_key in MODEL_KEYS:
        load_gguf(judge_key)
        for (q, gen, rep, seed, high_is_a) in units:
            if judge_key == gen:
                relation = "self"
            else:
                relation = "cross"
            low, high = generations[(q["id"], gen, rep)]
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
                "lang": q["lang"],
                "category": q["category"],
                "generator": gen,
                "estimator": judge_key,
                "relation": relation,
                "reveal_identity": False,
                "temp_low": TEMP_LOW,
                "temp_high": TEMP_HIGH,
                "entropy_low": low["mean_entropy"],
                "entropy_high": high["mean_entropy"],
                "entropy_gap": high["mean_entropy"] - low["mean_entropy"],
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
        unload_gguf()

    output = {
        "status": "done",
        "scale": "3B",
        "models": {k: v["label"] for k, v in MODEL_REGISTRY_GGUF.items()},
        "config": {
            "question_ids": [q["id"] for q in selected],
            "reps": args.reps,
            "temp_low": TEMP_LOW,
            "temp_high": TEMP_HIGH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "reveal_identity": False,
        },
        "divergence_check": divergence,
        "trials": trials,
        "analysis": analyze_pairwise(trials),
    }
    _save(output, name="pairwise_3b_run.json")
    shutil.copy(
        os.path.join(RESULTS_DIR, "pairwise_3b_run.json"),
        os.path.join(FRONTEND_DIR, "pairwise_3b_results.json"),
    )
    print("\nPairwise 3B analysis:", flush=True)
    for relation, stats in output["analysis"]["by_relation"].items():
        print(
            f"  {relation:5s} n={stats['n']} acc={stats['accuracy']:.3f} "
            f"binom_p={stats['binom_p']:.4f}",
            flush=True,
        )
    comp = output["analysis"]["self_vs_cross"]
    print(f"  self vs cross: z={comp['z']}, p={comp['p']}", flush=True)


def _save(data, name="pairwise_3b_run.json") -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
