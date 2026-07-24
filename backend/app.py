"""FastAPI backend for the entropy self-estimation lab.

Serves the experiment engine and the static frontend. Run with:

    uvicorn app:app --host 0.0.0.0 --port 8000

from this directory (see requirements.txt).
"""

import json
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engines
import experiment
from questions import QUESTION_POOL
from experiment import RUNS, start_run

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))
RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "results"))

# Resolve the active engine (LAB_ENGINE: local | gguf | api) once at
# startup and point the experiment runner at its model registry.
ACTIVE_ENGINE = engines.active_engine()
MODEL_REGISTRY = engines.get_registry()
experiment.ESTIMATOR_KEYS = list(MODEL_REGISTRY.keys())

app = FastAPI(title="Entropy Self-Estimation Lab")


class RunRequest(BaseModel):
    question_ids: list[str] | None = None
    use_selected: bool = True
    generators: list[str] = list(MODEL_REGISTRY.keys())
    timings: list[str] = ["pre", "post"]
    relations: list[str] = ["self", "cross"]
    reps: int = 1
    temp_min: float = 0.6
    temp_max: float = 1.2
    max_new_tokens: int = 48
    reveal_identity: bool = False
    master_seed: int = 7


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    selected = _load_selected()
    engine_info = {"name": ACTIVE_ENGINE}
    if ACTIVE_ENGINE == "api":
        cfg = engines.get_config()
        engine_info["api_base"] = cfg["LAB_API_BASE"]
    return {
        "engine": engine_info,
        "models": {k: v["label"] for k, v in MODEL_REGISTRY.items()},
        "questions": QUESTION_POOL,
        "selected_question_ids": [q["id"] for q in selected.get("questions", [])],
        "selection_report": selected,
    }


@app.post("/api/runs")
def create_run(req: RunRequest):
    questions = QUESTION_POOL
    if req.use_selected:
        selected_ids = {q["id"] for q in _load_selected().get("questions", [])}
        if selected_ids:
            questions = [q for q in QUESTION_POOL if q["id"] in selected_ids]
    if req.question_ids:
        wanted = set(req.question_ids)
        questions = [q for q in questions if q["id"] in wanted]
    if not questions:
        raise HTTPException(status_code=400, detail="No questions selected")

    config = {
        "questions": questions,
        "generators": req.generators,
        "timings": req.timings,
        "relations": req.relations,
        "reps": max(1, min(req.reps, 5)),
        "temp_min": req.temp_min,
        "temp_max": req.temp_max,
        "max_new_tokens": req.max_new_tokens,
        "reveal_identity": req.reveal_identity,
        "master_seed": req.master_seed,
    }
    run_id = start_run(config)
    return {"run_id": run_id}


@app.get("/api/pairwise_results")
def pairwise_results():
    """First available pairwise results file, in preference order.

    Resolved server-side so the frontend does not probe missing files
    (which logs noisy 404s). Returns 204 when no pairwise run exists.
    """
    from fastapi import Response

    for name in (
        "pairwise_api_results.json",
        "pairwise_3b_results.json",
        "pairwise_results.json",
    ):
        path = os.path.join(FRONTEND_DIR, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return Response(status_code=204)


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    run = RUNS.get(run_id) or _load_run_from_disk(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": run["id"],
        "status": run["status"],
        "progress": run.get("progress"),
        "error": run.get("error"),
    }


@app.get("/api/runs/{run_id}/results")
def run_results(run_id: str):
    run = RUNS.get(run_id) or _load_run_from_disk(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "done":
        raise HTTPException(status_code=202, detail="Run still in progress")
    return {
        "id": run["id"],
        "config": _public_config(run),
        "trials": run["trials"],
        "analysis": run.get("analysis"),
    }


def _public_config(run: dict) -> dict:
    cfg = dict(run.get("config", {}))
    cfg["question_ids"] = [q["id"] for q in cfg.pop("questions", [])]
    return cfg


def _load_run_from_disk(run_id: str):
    path = os.path.join(RESULTS_DIR, f"run_{run_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_selected() -> dict:
    path = os.path.join(RESULTS_DIR, "selected_questions.json")
    if not os.path.exists(path):
        return {"questions": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
