"""Thin, cached client for the public Fantasy Premier League API.

The FPL API is unauthenticated but rate-limit sensitive, so every response is
cached on disk with a TTL. Tests inject a fake `fetch` callable instead.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api"
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_SHAPE = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

Fetch = Callable[[str], dict | list]


class FPLClient:
    def __init__(
        self,
        cache_dir: Path | str = ".cache/fpl",
        ttl_seconds: int = 1800,
        fetch: Fetch | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl_seconds
        self._fetch = fetch or self._http_get
        self._http: httpx.Client | None = None

    def _http_get(self, path: str) -> dict | list:
        if self._http is None:
            self._http = httpx.Client(
                base_url=BASE_URL,
                timeout=20.0,
                headers={"User-Agent": "gaffer/0.1 (FPL research project)"},
            )
        resp = self._http.get(path)
        resp.raise_for_status()
        return resp.json()

    def get(self, path: str, ttl: int | None = None) -> dict | list:
        ttl = self.ttl if ttl is None else ttl
        key = path.strip("/").replace("/", "__") or "root"
        cached = self.cache_dir / f"{key}.json"
        if cached.exists() and time.time() - cached.stat().st_mtime < ttl:
            return json.loads(cached.read_text())
        data = self._fetch(path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data))
        return data

    def bootstrap(self) -> dict:
        return self.get("/bootstrap-static/")

    def fixtures(self) -> list:
        return self.get("/fixtures/")

    def element_summary(self, player_id: int, ttl: int | None = None) -> dict:
        return self.get(f"/element-summary/{player_id}/", ttl=ttl)

    def entry(self, team_id: int) -> dict:
        return self.get(f"/entry/{team_id}/", ttl=300)

    def entry_history(self, team_id: int) -> dict:
        return self.get(f"/entry/{team_id}/history/", ttl=300)

    def entry_picks(self, team_id: int, gw: int) -> dict:
        return self.get(f"/entry/{team_id}/event/{gw}/picks/", ttl=300)


@dataclass
class Snapshot:
    """Everything the model needs, normalised into DataFrames."""

    players: pd.DataFrame  # indexed by player id
    teams: pd.DataFrame  # indexed by team id
    fixtures: pd.DataFrame  # one row per fixture
    scoring: dict
    current_gw: int | None
    next_gw: int
    deadline: str | None
    rules: dict = field(default_factory=dict)

    def team_games_played(self, before_gw: int | None = None) -> pd.Series:
        fx = self.fixtures[self.fixtures.finished]
        if before_gw is not None:
            fx = fx[fx.event < before_gw]
        counts = pd.concat([fx.team_h, fx.team_a]).value_counts()
        return counts.reindex(self.teams.index, fill_value=0)

    def name(self, player_id: int) -> str:
        return str(self.players.at[player_id, "name"])


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def build_snapshot(client: FPLClient) -> Snapshot:
    boot = client.bootstrap()
    fixtures_raw = client.fixtures()
    return snapshot_from_raw(boot, fixtures_raw)


def snapshot_from_raw(boot: dict, fixtures_raw: list) -> Snapshot:
    teams = pd.DataFrame([{"id": t["id"], "team_name": t["name"], "short": t["short_name"]} for t in boot["teams"]]).set_index(
        "id"
    )

    rows = []
    for e in boot["elements"]:
        rows.append(
            {
                "id": e["id"],
                "name": e["web_name"],
                "full_name": f"{e['first_name']} {e['second_name']}",
                "team": e["team"],
                "pos": POSITIONS[e["element_type"]],
                "price": e["now_cost"] / 10,
                "status": e["status"],
                "chance": e["chance_of_playing_next_round"],
                "news": e["news"],
                "minutes": e["minutes"],
                "starts": e["starts"],
                "xg": _f(e["expected_goals"]),
                "xa": _f(e["expected_assists"]),
                "dc": _f(e.get("defensive_contribution", 0)),
                "saves": _f(e["saves"]),
                "bonus": _f(e["bonus"]),
                "yellow": _f(e["yellow_cards"]),
                "total_points": e["total_points"],
                "form": _f(e["form"]),
                "selected_by": _f(e["selected_by_percent"]),
                "penalties_order": e.get("penalties_order"),
            }
        )
    players = pd.DataFrame(rows).set_index("id")
    players["team_short"] = players.team.map(teams.short)

    fixtures = pd.DataFrame(
        [
            {
                "id": f["id"],
                "event": f["event"],
                "team_h": f["team_h"],
                "team_a": f["team_a"],
                "h_score": f["team_h_score"],
                "a_score": f["team_a_score"],
                "finished": bool(f["finished"]),
                "kickoff": f["kickoff_time"],
            }
            for f in fixtures_raw
        ]
    )

    current = next((ev["id"] for ev in boot["events"] if ev["is_current"]), None)
    nxt_ev = next((ev for ev in boot["events"] if ev["is_next"]), None)
    next_gw = nxt_ev["id"] if nxt_ev else (current or 1)
    deadline = nxt_ev["deadline_time"] if nxt_ev else None

    return Snapshot(
        players=players,
        teams=teams,
        fixtures=fixtures,
        scoring=boot["game_config"]["scoring"],
        current_gw=current,
        next_gw=next_gw,
        deadline=deadline,
        rules=boot["game_config"]["rules"],
    )
