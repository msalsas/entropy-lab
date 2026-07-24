"""End-to-end integration test: web app + api engine against a mock
OpenAI-compatible server (stands in for LM Studio).

Verifies the whole remote path a user deploys: .env-selected api engine,
config endpoint, run creation, background execution, polling, results —
without torch, llama.cpp or a real server.
"""

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import api_engine
import experiment

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------- mock server

TOP_LOGPROBS = [
    {"token": f"tok{i}", "logprob": math.log(p)}
    for i, p in enumerate([0.5, 0.2, 0.1] + [0.2 / 17] * 17)
]


class _Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence test output
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"data": [{"id": "remote-a"}, {"id": "remote-b"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        if payload.get("logprobs"):
            tokens = ["alpha", " beta", " gamma"]
            content = [
                {"token": t, "logprob": TOP_LOGPROBS[0]["logprob"],
                 "top_logprobs": TOP_LOGPROBS}
                for t in tokens
            ]
            self._json({
                "choices": [{
                    "message": {"content": "alpha beta gamma"},
                    "logprobs": {"content": content},
                }]
            })
        else:  # deterministic judge call
            self._json({"choices": [{"message": {"content": "7"}}]})


@pytest.fixture(scope="module")
def mock_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


# ------------------------------------------------------------- app under test

@pytest.fixture(scope="module")
def client(mock_server):
    """Import the app with LAB_ENGINE=api pointed at the mock server."""
    mp = pytest.MonkeyPatch()
    mp.setenv("LAB_ENGINE", "api")
    mp.setenv("LAB_API_BASE", mock_server)
    mp.setenv("LAB_MODEL_A", "remote-a")
    mp.setenv("LAB_MODEL_A_LABEL", "Remote A")
    mp.setenv("LAB_MODEL_B", "remote-b")
    mp.setenv("LAB_MODEL_B_LABEL", "Remote B")
    api_engine._config = None
    saved_keys = list(experiment.ESTIMATOR_KEYS)

    import app as app_module

    yield TestClient(app_module.app)

    experiment.ESTIMATOR_KEYS = saved_keys
    api_engine._config = None
    mp.undo()


def test_config_reports_remote_engine(client, mock_server):
    cfg = client.get("/api/config").json()
    assert cfg["engine"]["name"] == "api"
    assert cfg["engine"]["api_base"] == mock_server
    assert cfg["models"] == {"model_a": "Remote A", "model_b": "Remote B"}
    assert experiment.ESTIMATOR_KEYS == ["model_a", "model_b"]


def test_pairwise_results_endpoint_no_404(client):
    resp = client.get("/api/pairwise_results")
    assert resp.status_code == 200  # bundled 3B results in frontend/
    assert "analysis" in resp.json()


def test_full_run_against_mock_lm_studio(client):
    from questions import QUESTION_POOL

    resp = client.post("/api/runs", json={
        "use_selected": False,
        "question_ids": [QUESTION_POOL[0]["id"]],
        "generators": ["model_a", "model_b"],
        "timings": ["post"],
        "relations": ["self", "cross"],
        "reps": 1,
        "max_new_tokens": 4,
    })
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    deadline = time.time() + 30
    status = "running"
    while time.time() < deadline:
        status = client.get(f"/api/runs/{run_id}").json()["status"]
        if status != "running":
            break
        time.sleep(0.2)
    assert status == "done", client.get(f"/api/runs/{run_id}").json()

    results = client.get(f"/api/runs/{run_id}/results").json()
    trials = results["trials"]
    assert len(trials) == 4  # 2 generators x 2 relations

    by_relation = {(t["generator"], t["relation"]): t for t in trials}
    self_a = by_relation[("model_a", "self")]
    assert self_a["estimator"] == "model_a"
    assert self_a["answer"] == "alpha beta gamma"
    assert self_a["estimate"] == 7.0
    assert self_a["true_entropy"] > 0
    assert self_a["topk_mass_mean"] == pytest.approx(1.0)
    cross_a = by_relation[("model_a", "cross")]
    assert cross_a["estimator"] == "model_b"

    analysis = results["analysis"]
    assert "pooled" in analysis or "by_condition" in analysis
