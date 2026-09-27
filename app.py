"""
Flask web app: 3D traffic-signal simulator + a small JSON API over the trained agents.

    python app.py          # http://localhost:5000

On Vercel this file is the entrypoint (zero config); everything in public/ is
served from the CDN.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
NS_MAX, EW_MAX = 3, 5
ACTIONS = ("EW_GREEN", "NS_GREEN")

POLICY_FILES = {
    "q_learning": "q_learning.json",
    "monte_carlo": "monte_carlo.json",
    "optimal": "optimal.json",
    "optimal_overflow_aware": "optimal_overflow_aware.json",
}

app = Flask(__name__, static_folder=str(ROOT / "public" / "static"), static_url_path="/static",
            template_folder=str(ROOT / "templates"))


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    return json.loads((MODELS / POLICY_FILES[name]).read_text())


@lru_cache(maxsize=None)
def load_results() -> dict:
    p = ROOT / "results" / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _int_arg(data, key, lo, hi):
    raw = data.get(key)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"'{key}' is required")
    try:
        v = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"'{key}' must be a whole number") from None
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi} (got {v})")
    return v


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/api/policies")
def policies():
    """Every Q-table in one payload, used by the browser simulator."""
    out = {}
    for name in POLICY_FILES:
        m = load(name)
        out[name] = {k: v for k, v in m.items() if k != "state"}
    return jsonify(ns_max=NS_MAX, ew_max=EW_MAX, actions=ACTIONS, policies=out)


@app.get("/api/results")
def results():
    return jsonify(load_results())


@app.route("/api/decide", methods=["GET", "POST"])
def decide():
    """Greedy decision for a queue state.

    GET  /api/decide?ns=2&ew=4&policy=q_learning
    POST /api/decide   {"ns": 2, "ew": 4, "policy": "monte_carlo"}
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        if not isinstance(data, dict) and not hasattr(data, "get"):
            return jsonify(error="send a JSON object"), 400
    else:
        data = request.args
    try:
        ns = _int_arg(data, "ns", 0, NS_MAX)
        ew = _int_arg(data, "ew", 0, EW_MAX)
        policy = (data.get("policy") or "q_learning").strip()
        if policy not in POLICY_FILES:
            raise ValueError(f"'policy' must be one of: {', '.join(POLICY_FILES)}")
    except ValueError as e:
        return jsonify(error=str(e)), 400

    q = load(policy)["Q"][ns][ew]
    q_opt = load("optimal")["Q"][ns][ew]
    action = 0 if q[0] >= q[1] else 1
    optimal = 0 if q_opt[0] >= q_opt[1] else 1
    return jsonify(
        state={"ns": ns, "ew": ew},
        policy=policy,
        action=ACTIONS[action],
        q_values={"EW_GREEN": round(q[0], 4), "NS_GREEN": round(q[1], 4)},
        optimal_action=ACTIONS[optimal],
        matches_optimal=action == optimal,
    )


@app.after_request
def headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
