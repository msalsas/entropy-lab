"""Experiment runner and analysis.

A trial measures one thing: the correlation between a model's verbalized
unpredictability estimate and the true entropy of the sampling
distribution, which is only measurable from outside the model.

Design controls implemented here:

1. Textual baseline: the ``cross`` + ``post`` condition gives another
   model the same visible information (question + answer). A self
   advantage only counts above this baseline.
2. Identity control: by default the estimator is never told which model
   generated the answer, blocking trained knowledge about a specific
   model's behaviour (``reveal_identity`` toggles this).
3. Temporal control: the ``pre`` condition asks for the estimate before
   the answer exists, so it cannot be read off the generated text.
4. Hidden temperature: the sampling temperature is drawn uniformly from
   [temp_min, temp_max] per trial and never shown to the estimator, so
   the entropy level itself is not disclosed.
"""

import json
import math
import os
import random
import threading
import time
import uuid

from estimator import (
    build_estimation_prompt,
    build_pairwise_prompt,
    parse_choice,
    verbalized_estimate,
)

import engines

try:  # transformers-based generation is optional (GGUF/api-only runs)
    from entropy import generate_with_entropy
    from models import MODEL_REGISTRY
except ImportError:  # pragma: no cover - depends on environment
    generate_with_entropy = None
    try:
        from gguf_engine import MODEL_REGISTRY_GGUF as MODEL_REGISTRY
    except ImportError:
        MODEL_REGISTRY = {}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RUNS: dict[str, dict] = {}

ESTIMATOR_KEYS = list(MODEL_REGISTRY.keys())


def refresh_engines() -> list[str]:
    """Sync the estimator key list with the active engine's registry.

    Called by entry points (web app, run scripts) after ``LAB_ENGINE``
    is resolved, so ``_other`` and default configs see the right models.
    """
    global ESTIMATOR_KEYS
    keys = list(engines.get_registry().keys())
    if keys:
        ESTIMATOR_KEYS = keys
    return ESTIMATOR_KEYS


def _engine() -> str:
    return engines.active_engine()


def _generate(
    generator_key: str,
    question_text: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> dict:
    if _engine() == "local":
        return generate_with_entropy(
            model_key=generator_key,
            question_text=question_text,
            temperature=temperature,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
    return engines.generate(
        generator_key, question_text, temperature, seed, max_new_tokens
    )


def _estimate(estimator_key: str, prompt: str) -> dict:
    """Numeric estimate; local path stays patchable for tests."""
    if _engine() == "local":
        return verbalized_estimate(estimator_key, prompt)
    return engines.estimate(estimator_key, prompt)


def _judge_text(estimator_key: str, prompt: str) -> str:
    """Raw judge completion; local path stays patchable for tests."""
    if _engine() == "local":
        return verbalized_estimate(estimator_key, prompt)["raw"]
    return engines.judge_raw(estimator_key, prompt)


def _other(model_key: str) -> str:
    return next(k for k in ESTIMATOR_KEYS if k != model_key)


def run_trial(
    question: dict,
    generator_key: str,
    timing: str,
    relation: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
    reveal_identity: bool,
    generation_cache: dict | None = None,
) -> dict:
    """Execute one trial and return its full record."""
    estimator_key = (
        generator_key if relation == "self" else _other(generator_key)
    )

    estimate = None
    estimate_raw = None

    # Temporal control: in the "pre" condition the estimate is produced
    # before the answer exists.
    if timing == "pre":
        prompt = build_estimation_prompt(
            timing="pre",
            estimator_relation=relation,
            question=question["text"],
            answer=None,
            generator_key=generator_key,
            reveal_identity=reveal_identity,
        )
        est = _estimate(estimator_key, prompt)
        estimate, estimate_raw = est["value"], est["raw"]

    cache_key = (question["id"], generator_key, temperature, seed)
    if generation_cache is not None and cache_key in generation_cache:
        gen = generation_cache[cache_key]
    else:
        gen = _generate(
            generator_key, question["text"], temperature, seed, max_new_tokens
        )
        if generation_cache is not None:
            generation_cache[cache_key] = gen

    if timing == "post":
        prompt = build_estimation_prompt(
            timing="post",
            estimator_relation=relation,
            question=question["text"],
            answer=gen["answer"],
            generator_key=generator_key,
            reveal_identity=reveal_identity,
        )
        est = _estimate(estimator_key, prompt)
        estimate, estimate_raw = est["value"], est["raw"]

    return {
        "question_id": question["id"],
        "question": question["text"],
        "lang": question["lang"],
        "category": question["category"],
        "generator": generator_key,
        "estimator": estimator_key,
        "relation": relation,
        "timing": timing,
        "reveal_identity": reveal_identity,
        "temperature": temperature,
        "seed": seed,
        "answer": gen["answer"],
        "n_tokens": gen["n_tokens"],
        "true_entropy": gen["mean_entropy"],
        "topk_mass_mean": gen.get("topk_mass_mean"),
        "estimate": estimate,
        "estimate_raw": estimate_raw,
    }


def start_run(config: dict) -> str:
    """Launch an experiment run in a background thread."""
    run_id = uuid.uuid4().hex[:10]
    RUNS[run_id] = {
        "id": run_id,
        "config": config,
        "status": "running",
        "progress": {"done": 0, "total": 0, "current": ""},
        "trials": [],
        "started_at": time.time(),
        "error": None,
    }
    thread = threading.Thread(target=_run_worker, args=(run_id,), daemon=True)
    thread.start()
    return run_id


def _run_worker(run_id: str) -> None:
    run = RUNS[run_id]
    cfg = run["config"]
    try:
        questions = cfg["questions"]
        combos = []
        for rep in range(cfg.get("reps", 1)):
            for question in questions:
                for generator in cfg.get("generators", ESTIMATOR_KEYS):
                    for timing in cfg.get("timings", ["pre", "post"]):
                        for relation in cfg.get("relations", ["self", "cross"]):
                            combos.append((rep, question, generator, timing, relation))

        run["progress"]["total"] = len(combos)
        rng = random.Random(cfg.get("master_seed", 7))
        generation_cache: dict = {}

        # Draw one hidden temperature per (rep, question, generator) so
        # that all conditions of the same unit share the sampled answer.
        for rep, question, generator, timing, relation in combos:
            unit_key = (rep, question["id"], generator)
            if unit_key not in generation_cache:
                generation_cache[unit_key] = True
            run["progress"]["current"] = (
                f"{question['id']} | gen={generator} | {relation}-{timing}"
            )
            temperature = rng.uniform(
                cfg.get("temp_min", 0.6), cfg.get("temp_max", 1.2)
            )
            seed = rng.randint(0, 2**31 - 1)
            trial = run_trial(
                question=question,
                generator_key=generator,
                timing=timing,
                relation=relation,
                temperature=temperature,
                seed=seed,
                max_new_tokens=cfg.get("max_new_tokens", 48),
                reveal_identity=cfg.get("reveal_identity", False),
                generation_cache=None,
            )
            run["trials"].append(trial)
            run["progress"]["done"] += 1

        run["analysis"] = analyze_trials(run["trials"])
        run["status"] = "done"
        run["finished_at"] = time.time()
        _persist(run)
    except Exception as exc:  # noqa: BLE001 - surfaced through the API
        run["status"] = "error"
        run["error"] = f"{type(exc).__name__}: {exc}"
        raise


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _fisher_ci(r: float | None, n: int, z_alpha: float = 1.96):
    """95% confidence interval for a Pearson r via Fisher's z."""
    if r is None or n < 4 or abs(r) >= 1.0:
        return None
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    return (math.tanh(z - z_alpha * se), math.tanh(z + z_alpha * se))


def _fisher_p(r: float | None, n: int) -> float | None:
    """Two-sided p-value for H0: r = 0 via Fisher's z (normal approx)."""
    if r is None or n < 4 or abs(r) >= 1.0:
        return None
    z = math.atanh(r) * math.sqrt(n - 3)
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


def _compare_rs(r1, n1, r2, n2) -> dict:
    """Fisher z test for the difference of two independent correlations."""
    if None in (r1, r2) or min(n1, n2) < 4:
        return {"z": None, "p": None}
    z = (math.atanh(r1) - math.atanh(r2)) / math.sqrt(
        1.0 / (n1 - 3) + 1.0 / (n2 - 3)
    )
    return {"z": z, "p": 2.0 * (1.0 - _norm_cdf(abs(z)))}


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _condition_stats(group: list[dict]) -> dict:
    xs = [t["estimate"] for t in group]
    ys = [t["true_entropy"] for t in group]
    r = _pearson(xs, ys)
    return {
        "n": len(group),
        "pearson_r": r,
        "pearson_ci95": _fisher_ci(r, len(group)),
        "pearson_p": _fisher_p(r, len(group)),
        "spearman_r": _spearman(xs, ys),
        "mean_estimate": sum(xs) / len(xs),
        "mean_true_entropy": sum(ys) / len(ys),
    }


def _question_level(group: list[dict]) -> list[tuple[float, float]]:
    """Aggregate to one point per question (mean over reps/generators).

    Trials within the same question are not independent, so the
    question-level correlation is the pseudoreplication-safe companion
    to the trial-level one.
    """
    by_q: dict[str, list[dict]] = {}
    for t in group:
        by_q.setdefault(t["question_id"], []).append(t)
    return [
        (
            sum(t["estimate"] for t in ts) / len(ts),
            sum(t["true_entropy"] for t in ts) / len(ts),
        )
        for ts in by_q.values()
    ]


def analyze_trials(trials: list[dict]) -> dict:
    """Correlation between verbalized estimates and true entropy.

    Primary endpoint per condition: Pearson r of (estimate,
    true_entropy) with a 95% Fisher CI and p-value. Decisive
    comparisons: self vs cross within each timing, tested with a Fisher
    z difference test. Question-level correlations guard against
    pseudoreplication within questions.
    """
    usable = [t for t in trials if t["estimate"] is not None]

    groups: dict[str, list[dict]] = {}
    for trial in usable:
        key = f"{trial['generator']}|{trial['relation']}|{trial['timing']}"
        groups.setdefault(key, []).append(trial)
    conditions = {key: _condition_stats(g) for key, g in groups.items()}

    pooled: dict[str, list[dict]] = {}
    for trial in usable:
        key = f"{trial['relation']}|{trial['timing']}"
        pooled.setdefault(key, []).append(trial)
    pooled_summary = {}
    for key, group in pooled.items():
        stats = _condition_stats(group)
        q_points = _question_level(group)
        qx = [p[0] for p in q_points]
        qy = [p[1] for p in q_points]
        stats["question_level"] = {
            "n": len(q_points),
            "pearson_r": _pearson(qx, qy),
            "pearson_p": _fisher_p(_pearson(qx, qy), len(q_points)),
        }
        pooled_summary[key] = stats

    comparisons = {}
    for timing in ("pre", "post"):
        self_stats = pooled_summary.get(f"self|{timing}", {})
        cross_stats = pooled_summary.get(f"cross|{timing}", {})
        comparisons[f"self_vs_cross|{timing}"] = _compare_rs(
            self_stats.get("pearson_r"), self_stats.get("n", 0),
            cross_stats.get("pearson_r"), cross_stats.get("n", 0),
        )

    return {
        "by_condition": conditions,
        "pooled": pooled_summary,
        "comparisons": comparisons,
    }


def run_pairwise_trial(
    question: dict,
    generator_key: str,
    relation: str,
    gen_low: dict,
    gen_high: dict,
    high_is_a: bool,
    reveal_identity: bool,
) -> dict:
    """One comparative judgment: which answer was more unpredictable?

    ``gen_low``/``gen_high`` are pre-computed generations (low vs high
    temperature) for the same question and generator. ``high_is_a``
    randomizes presentation order so position bias can be measured and
    cancelled out. The judge is the generator itself (self) or the other
    model (cross), with identity hidden by default.
    """
    estimator_key = (
        generator_key if relation == "self" else _other(generator_key)
    )
    answer_a = gen_high["answer"] if high_is_a else gen_low["answer"]
    answer_b = gen_low["answer"] if high_is_a else gen_high["answer"]

    prompt = build_pairwise_prompt(
        relation=relation,
        question=question["text"],
        answer_a=answer_a,
        answer_b=answer_b,
        generator_key=generator_key,
        reveal_identity=reveal_identity,
    )
    raw = _judge_text(estimator_key, prompt)
    choice = parse_choice(raw)

    correct_letter = "A" if high_is_a else "B"
    entropy_gap = gen_high["mean_entropy"] - gen_low["mean_entropy"]

    return {
        "question_id": question["id"],
        "question": question["text"],
        "lang": question["lang"],
        "category": question["category"],
        "generator": generator_key,
        "estimator": estimator_key,
        "relation": relation,
        "reveal_identity": reveal_identity,
        "temp_low": gen_low["temperature"],
        "temp_high": gen_high["temperature"],
        "entropy_low": gen_low["mean_entropy"],
        "entropy_high": gen_high["mean_entropy"],
        "entropy_gap": entropy_gap,
        "high_is_a": high_is_a,
        "choice": choice,
        "choice_raw": raw,
        "correct": (choice == correct_letter) if choice else None,
    }


def _binom_p_two_sided(k: int, n: int, p0: float = 0.5) -> float | None:
    """Exact two-sided binomial test for accuracy against chance."""
    if n == 0:
        return None
    from math import comb

    def prob(i: int) -> float:
        return comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))

    p_obs = prob(k)
    return min(1.0, sum(prob(i) for i in range(n + 1) if prob(i) <= p_obs + 1e-12))


def _compare_proportions(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Two-proportion z test (self vs cross accuracy)."""
    if min(n1, n2) == 0:
        return {"z": None, "p": None}
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": None, "p": None}
    z = (p1 - p2) / se
    return {"z": z, "p": 2.0 * (1.0 - _norm_cdf(abs(z)))}


def analyze_pairwise(trials: list[dict]) -> dict:
    """Accuracy of comparative judgments against 50% chance.

    Primary endpoint per relation: exact binomial test of accuracy, plus
    a two-proportion z test for the self vs cross difference. Accuracy
    is also reported conditional on presentation order (position-bias
    check) and binned by the size of the true entropy gap.
    """
    usable = [t for t in trials if t["correct"] is not None]
    by_relation: dict[str, list[dict]] = {}
    for t in usable:
        by_relation.setdefault(t["relation"], []).append(t)

    relations = {}
    for relation, group in by_relation.items():
        n = len(group)
        k = sum(1 for t in group if t["correct"])
        by_order = {"high_A": [], "high_B": []}
        for t in group:
            by_order["high_A" if t["high_is_a"] else "high_B"].append(t)
        order_stats = {
            key: {
                "n": len(ts),
                "accuracy": (sum(1 for t in ts if t["correct"]) / len(ts)) if ts else None,
            }
            for key, ts in by_order.items()
        }
        # Accuracy binned by true entropy gap size.
        bins: dict[str, list[dict]] = {"small": [], "large": []}
        gaps = sorted(abs(t["entropy_gap"]) for t in group)
        median_gap = gaps[len(gaps) // 2] if gaps else 0
        for t in group:
            bins["large" if abs(t["entropy_gap"]) >= median_gap else "small"].append(t)
        gap_stats = {
            key: {
                "n": len(ts),
                "accuracy": (sum(1 for t in ts if t["correct"]) / len(ts)) if ts else None,
            }
            for key, ts in bins.items()
        }
        relations[relation] = {
            "n": n,
            "correct": k,
            "accuracy": k / n if n else None,
            "binom_p": _binom_p_two_sided(k, n),
            "by_order": order_stats,
            "by_gap": gap_stats,
            "median_abs_gap": median_gap,
        }

    comparison = _compare_proportions(
        relations.get("self", {}).get("correct", 0),
        relations.get("self", {}).get("n", 0),
        relations.get("cross", {}).get("correct", 0),
        relations.get("cross", {}).get("n", 0),
    )
    return {"by_relation": relations, "self_vs_cross": comparison}


def _persist(run: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"run_{run['id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(run, fh, ensure_ascii=False, indent=2, default=str)
