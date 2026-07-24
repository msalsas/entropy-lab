"""Compare original vs entropy-literate re-judgment.

The gradient-inversion hypothesis: big entropy gaps are produced by
degenerate text, and cross judges read broken text as defective rather
than unpredictable. This script contrasts accuracy on degenerate vs
clean pairs between the original prompt and the entropy-literate
prompt, which explicitly frames broken text as the signature of maximum
randomness.
"""

import json
import os
import re
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def degeneration(text: str) -> float:
    if "im_start" in text or "im_end" in text:
        return 1.0
    words = re.findall(r"\S+", text)
    if not words:
        return 0.0
    bad = sum(
        1
        for w in words
        if re.search(r"[^\w\s.,;:!?¿¡()«»\"'\-áéíóúñüÁÉÍÓÚÑ]", w)
    )
    return bad / len(words)


def load_generations():
    with open(os.path.join(RESULTS_DIR, "pairwise_3b_generations.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw


def trial_degenerate(trial, generations):
    for key, (low, high) in generations.items():
        qid, gen, rep = key.split("|")
        if qid != trial["question_id"] or gen != trial["generator"]:
            continue
        if (
            abs(low["mean_entropy"] - trial["entropy_low"]) < 1e-9
            and abs(high["mean_entropy"] - trial["entropy_high"]) < 1e-9
        ):
            return degeneration(high["answer"]) > 0.05
    return None


def summarize(trials, generations, label):
    buckets = {}
    for t in trials:
        if t["correct"] is None:
            continue
        deg = trial_degenerate(t, generations)
        if deg is None:
            continue
        key = (t["relation"], "degenerate" if deg else "clean")
        buckets.setdefault(key, []).append(t["correct"])
    print(f"\n{label}")
    for (rel, kind), vals in sorted(buckets.items()):
        n = len(vals)
        acc = sum(vals) / n
        print(f"  {rel:5s} {kind:10s} n={n:3d} acc={acc:.0%}")
    return buckets


def main():
    generations = load_generations()
    with open(os.path.join(RESULTS_DIR, "pairwise_3b_run.json"), encoding="utf-8") as fh:
        original = json.load(fh)["trials"]
    with open(os.path.join(RESULTS_DIR, "rejudge_run.json"), encoding="utf-8") as fh:
        rejudge = json.load(fh)["trials"]

    summarize(original, generations, "ORIGINAL PROMPT (unpredictable word choices)")
    summarize(rejudge, generations, "ENTROPY-LITERATE PROMPT (high vs low randomness)")


if __name__ == "__main__":
    main()
