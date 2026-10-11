"""Completed past seasons from the open vaastav/Fantasy-Premier-League dataset.

Each season becomes the same `Snapshot` + per-fixture history table that the
live backtest uses, so the model code under test is identical. The dataset also
records FPL's own expected points (`xP`) per player and gameweek, but
`fpl_xp_leakage` shows those values were captured after team news, so they are
not used as a baseline.
"""

from __future__ import annotations

import copy
from pathlib import Path

import httpx
import pandas as pd

from .fpl import POSITIONS, Snapshot

DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{path}"
FILES = {"gw": "gws/merged_gw.csv", "fixtures": "fixtures.csv", "teams": "teams.csv", "players": "players_raw.csv"}
SEASONS = ["2024-25", "2025-26"]

# 2025/26 scoring, matching the live game config. Seasons before 2025/26 had
# no defensive contribution points, so those are zeroed when the column is absent.
DEFAULT_SCORING = {
    "long_play": 2,
    "short_play": 1,
    "saves": 1,
    "assists": 3,
    "bonus": 1,
    "yellow_cards": -1,
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
}
HIST_FEATURES = {
    "minutes": "minutes",
    "starts": "starts",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "defensive_contribution": "dc",
    "saves": "saves",
    "bonus": "bonus",
    "yellow_cards": "yellow",
}


def _download(season: str, key: str, cache_dir: Path) -> Path:
    path = cache_dir / season / Path(FILES[key]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(DATA_URL.format(season=season, path=FILES[key]), timeout=60, follow_redirects=True)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


def load_season(season: str, cache_dir: Path | str = ".cache/history") -> tuple[Snapshot, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    files = {k: _download(season, k, cache_dir) for k in FILES}

    teams_raw = pd.read_csv(files["teams"])
    teams = teams_raw.rename(columns={"name": "team_name", "short_name": "short"}).set_index("id")[["team_name", "short"]]

    pr = pd.read_csv(files["players"])
    pr = pr[pr.element_type.isin(POSITIONS)]
    players = pd.DataFrame(
        {
            "name": pr.web_name.to_numpy(),
            "team": pr.team.to_numpy(),
            "pos": pr.element_type.map(POSITIONS).to_numpy(),
            "price": (pr.now_cost / 10).to_numpy(),
        },
        index=pd.Index(pr.id.to_numpy(), name="id"),
    )
    players["team_short"] = players.team.map(teams.short)

    fx = pd.read_csv(files["fixtures"])
    fixtures = pd.DataFrame(
        {
            "id": fx.id,
            "event": fx.event,
            "team_h": fx.team_h,
            "team_a": fx.team_a,
            "h_score": fx.team_h_score,
            "a_score": fx.team_a_score,
            "finished": fx.finished.astype(str).str.lower().eq("true"),
            "kickoff": fx.kickoff_time,
        }
    ).dropna(subset=["event"])
    fixtures["event"] = fixtures.event.astype(int)

    gw = pd.read_csv(files["gw"], encoding_errors="replace")
    gw = gw[gw.element.isin(players.index)]  # drops assistant managers and unregistered rows
    gw = gw.drop_duplicates(subset=["element", "GW", "fixture"])
    name_to_id = dict(zip(teams.team_name, teams.index))
    hist = pd.DataFrame(
        {
            "player_id": gw.element.astype(int),
            "round": gw.GW.astype(int),
            "fixture": gw.fixture.astype(int),
            "team": gw.team.map(name_to_id),
            "points": gw.total_points.astype(float),
            "value": gw.value / 10,
            "fpl_xp": pd.to_numeric(gw.xP, errors="coerce"),
        }
    )
    for src, dst in HIST_FEATURES.items():
        hist[dst] = pd.to_numeric(gw[src], errors="coerce").fillna(0) if src in gw else 0.0
    # Fall back to the fixture to find the club if the name lookup missed.
    missing = hist.team.isna()
    if missing.any():
        fx_teams = fixtures.set_index("id")[["team_h", "team_a"]]
        home = gw.loc[missing, "was_home"].astype(str).str.lower().eq("true")
        rows = fx_teams.reindex(hist.loc[missing, "fixture"])
        hist.loc[missing, "team"] = rows.team_h.where(home.to_numpy(), rows.team_a).to_numpy()
    hist["team"] = hist.team.astype(int)

    scoring = copy.deepcopy(DEFAULT_SCORING)
    if "defensive_contribution" not in gw:
        scoring["defensive_contribution"] = {p: 0 for p in scoring["defensive_contribution"]}

    snap = Snapshot(
        players=players,
        teams=teams,
        fixtures=fixtures,
        scoring=scoring,
        current_gw=int(fixtures.event.max()),
        next_gw=int(fixtures.event.max()),
        deadline=None,
    )
    return snap, hist.reset_index(drop=True)


def fpl_xp_leakage(hist: pd.DataFrame) -> dict:
    """Evidence that the archived FPL xP knows who played.

    Take players who started the previous gameweek. A genuine pre-deadline
    prediction can't separate the ones who then sat out from the ones who
    played anywhere near as cleanly as this does.
    """
    has_xp = hist.groupby("round").fpl_xp.transform(lambda x: x.fillna(0).ne(0).any())
    h = hist[has_xp]
    g = h.groupby(["player_id", "round"]).agg(xp=("fpl_xp", "first"), mins=("minutes", "sum"), pts=("points", "sum"))
    prev = h.groupby(["player_id", "round"]).starts.sum().rename("started_prev").reset_index()
    prev["round"] += 1
    g = g.join(prev.set_index(["player_id", "round"]), how="inner")
    reg = g[g.started_prev > 0]
    played = reg[reg.mins >= 60]
    return {
        "gameweeks_with_xp": int(h["round"].nunique()),
        "xp_when_previous_starter_then_did_not_play": round(float(reg[reg.mins == 0].xp.mean()), 2),
        "xp_when_previous_starter_then_played_60": round(float(played.xp.mean()), 2),
        "corr_xp_points_given_60_mins": round(float(played[["xp", "pts"]].corr().iloc[0, 1]), 3),
    }
