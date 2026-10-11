"""HTTP API, built for Google Cloud Run (stateless container, $PORT)."""

from __future__ import annotations

import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Gaffer
from .backtest import load_history
from .fpl import FPLClient, build_snapshot
from .store import Store
from .team import TeamState, load_team
from .tools import Toolbox
from .xp import ModelState, fit_model

MODEL_TTL_SECONDS = 15 * 60

app = FastAPI(title="Gaffer", description="An AI Fantasy Premier League manager with guardrails.")
_client = FPLClient(cache_dir="/tmp/gaffer-cache")
_store = Store()
_lock = threading.Lock()
_cache: dict = {"model": None, "at": 0.0}


def current_model() -> ModelState:
    with _lock:
        if _cache["model"] is None or time.time() - _cache["at"] > MODEL_TTL_SECONDS:
            snap = build_snapshot(_client)
            _cache["model"] = fit_model(snap, hist=load_history(_client, snap))
            _cache["at"] = time.time()
        return _cache["model"]


class AdviseRequest(BaseModel):
    question: str = Field("What transfers should I make this week, and who should I captain?", max_length=1000)
    team_id: int | None = Field(None, description="FPL team id; omit to draft a squad from scratch")
    free_transfers: int | None = Field(None, ge=0, le=5)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/players/top")
def top_players(position: str | None = None, n: int = 15):
    t = current_model().table
    if position:
        t = t[t.pos == position.upper()]
    cols = ["name", "team_short", "pos", "price", "avail", "xp_next", "xp_horizon"]
    return t.sort_values("xp_next", ascending=False).head(min(n, 100))[cols].round(2).reset_index(names="id").to_dict("records")


@app.post("/advise")
def advise(req: AdviseRequest):
    model = current_model()
    team = (
        load_team(_client, model.snap, req.team_id, free_transfers=req.free_transfers) if req.team_id else TeamState(team_id=None)
    )
    result = Gaffer(store=_store).run(req.question, Toolbox(model, team))
    if result.status == "error":
        raise HTTPException(502, result.answer)
    return {
        "run_id": result.run_id,
        "status": result.status,
        "gameweek": model.next_gw,
        "answer": result.answer,
        "recommendation": result.recommendation,
        "guardrail_attempts": result.guardrail_attempts,
        "usage": result.usage,
    }


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    run = _store.get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.get("/stats")
def stats():
    return _store.stats()
