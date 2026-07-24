"""Re-judgment with an entropy-literate prompt (gradient-inversion test).

The original pairwise prompt asked which answer had "more unpredictable
word choices". The gradient-inversion analysis showed big entropy gaps
are produced by *degenerate* text at high temperature, and cross judges
read broken text as defective rather than unpredictable (11% accuracy on
those pairs vs 64% on clean ones).

This run re-judges the SAME cached generations with a prompt that
operationalizes unpredictability as high-temperature sampling, stating
explicitly that broken/incoherent text is the signature of maximum
unpredictability. If accuracy on degenerate pairs jumps, the inversion
was a conceptual mismatch (judges' notion of unpredictability differs
from sampling entropy), not a perceptual limitation.

Output: results/rejudge_run.json
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from estimator import parse_choice
from experiment import analyze_pairwise
from gguf_engine import load_gguf, unload_gguf, judge

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_KEYS = ["qwen3b", "smollm17b"]

_LITERATE = (
    "Two answers to the same question were produced by sampling a "
    "language model's word-by-word probability distribution. One was "
    "sampled with HIGH randomness: its word choices came from a flat, "
    "uncertain distribution, which tends to produce incoherent, broken "
    "or nonsensical text. The other was sampled with LOW randomness: "
    "near-deterministic choices producing coherent, predictable text.\n\n"
    "Question: {question}\n\nAnswer A: {answer_a}\n\nAnswer B: {answer_b}\n\n"
    "Which answer was sampled with HIGH randomness? "
    "Reply with a single letter only: A or B."
)


def main() -> None:
    with open(os.path.join(RESULTS_DIR, "pairwise_3b_generations.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    generations = {}
    for key, (low, high) in raw.items():
        qid, gen, rep = key.split("|")
        generations[(qid, gen, int(rep))] = (low, high)

    with open(os.path.join(RESULTS_DIR, "selected_questions.json"), encoding="utf-8") as fh:
        questions = {q["id"]: q for q in json.load(fh)["all_gaps"]}

    rng = random.Random(23)
    units = sorted(generations.keys())
    order_randomization = {u: rng.random() < 0.5 for u in units}

    trials = []
    total = len(units) * 2
    done = 0
    t_start = time.time()
    for judge_key in MODEL_KEYS:
        load_gguf(judge_key)
        for (qid, gen, rep) in units:
            low, high = generations[(qid, gen, rep)]
            relation = "self" if judge_key == gen else "cross"
            high_is_a = order_randomization[(qid, gen, rep)]
            answer_a = high["answer"] if high_is_a else low["answer"]
            answer_b = low["answer"] if high_is_a else high["answer"]
            prompt = _LITERATE.format(
                question=questions[qid]["text"],
                answer_a=answer_a,
                answer_b=answer_b,
            )
            raw_choice = judge(judge_key, prompt)
            choice = parse_choice(raw_choice)
            correct_letter = "A" if high_is_a else "B"
            trials.append({
                "question_id": qid,
                "question": questions[qid]["text"],
                "generator": gen,
                "estimator": judge_key,
                "relation": relation,
                "prompt_version": "literate_v1",
                "entropy_low": low["mean_entropy"],
                "entropy_high": high["mean_entropy"],
                "entropy_gap": high["mean_entropy"] - low["mean_entropy"],
                "high_is_a": high_is_a,
                "choice": choice,
                "choice_raw": raw_choice,
                "correct": (choice == correct_letter) if choice else None,
            })
            done += 1
            eta = (time.time() - t_start) / done * (total - done)
            print(
                f"[{done}/{total}] {qid} gen={gen} {relation} "
                f"choice={choice} ok={trials[-1]['correct']} (eta {eta/60:.0f}m)",
                flush=True,
            )
        unload_gguf()

    output = {
        "status": "done",
        "prompt_version": "literate_v1",
        "trials": trials,
        "analysis": analyze_pairwise(trials),
    }
    with open(os.path.join(RESULTS_DIR, "rejudge_run.json"), "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print("\nRejudge analysis:", flush=True)
    for relation, stats in output["analysis"]["by_relation"].items():
        print(
            f"  {relation:5s} n={stats['n']} acc={stats['accuracy']:.3f} "
            f"binom_p={stats['binom_p']:.4f}",
            flush=True,
        )
    comp = output["analysis"]["self_vs_cross"]
    print(f"  self vs cross: z={comp['z']}, p={comp['p']}", flush=True)


if __name__ == "__main__":
    main()
