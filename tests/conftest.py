"""A synthetic 20-club league so tests are offline and deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from gaffer.fpl import snapshot_from_raw
from gaffer.xp import fit_model

SCORING = {
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
PER_CLUB = {1: 2, 2: 6, 3: 6, 4: 4}  # GKP, DEF, MID, FWD
N_TEAMS = 20
PLAYED_GWS = 3
TOTAL_GWS = 8


def round_robin(n):
    teams = list(range(1, n + 1))
    rounds = []
    for _ in range(n - 1):
        rounds.append([(teams[i], teams[n - 1 - i]) for i in range(n // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def make_raw(seed: int = 7):
    rng = np.random.default_rng(seed)
    teams = [{"id": i, "name": f"Club {i}", "short_name": f"C{i:02d}"} for i in range(1, N_TEAMS + 1)]
    att = rng.normal(0, 0.3, N_TEAMS)
    elements, pid = [], 1
    for t in range(1, N_TEAMS + 1):
        for et, n in PER_CLUB.items():
            for k in range(n):
                regular = k < {1: 1, 2: 4, 3: 4, 4: 2}[et]
                mins = 270 if regular else int(rng.integers(0, 60))
                xg90 = {1: 0, 2: 0.05, 3: 0.2, 4: 0.45}[et] * np.exp(att[t - 1]) * rng.uniform(0.5, 1.5)
                elements.append(
                    {
                        "id": pid,
                        "web_name": f"P{pid}",
                        "first_name": "Test",
                        "second_name": f"Player{pid}",
                        "team": t,
                        "element_type": et,
                        "now_cost": int({1: 45, 2: 45, 3: 60, 4: 65}[et] + rng.integers(0, 40) * (1 if regular else 0.2)),
                        "status": "a",
                        "chance_of_playing_next_round": None,
                        "news": "",
                        "minutes": mins,
                        "starts": 3 if regular else 0,
                        "expected_goals": f"{xg90 * mins / 90:.2f}",
                        "expected_assists": f"{xg90 * 0.6 * mins / 90:.2f}",
                        "defensive_contribution": int(rng.integers(0, 30)) if et == 2 else 0,
                        "saves": 9 if et == 1 and regular else 0,
                        "bonus": int(rng.integers(0, 3)),
                        "yellow_cards": 0,
                        "total_points": 10,
                        "form": "3.0",
                        "selected_by_percent": "5.0",
                    }
                )
                pid += 1
    rounds = round_robin(N_TEAMS)
    fixtures, fid = [], 1
    for gw in range(1, TOTAL_GWS + 1):
        for h, a in rounds[(gw - 1) % len(rounds)]:
            played = gw <= PLAYED_GWS
            fixtures.append(
                {
                    "id": fid,
                    "event": gw,
                    "team_h": h,
                    "team_a": a,
                    "finished": played,
                    "team_h_score": int(rng.poisson(1.5 * np.exp(att[h - 1]))) if played else None,
                    "team_a_score": int(rng.poisson(1.2 * np.exp(att[a - 1]))) if played else None,
                    "kickoff_time": None,
                }
            )
            fid += 1
    events = [
        {"id": g, "is_current": g == PLAYED_GWS, "is_next": g == PLAYED_GWS + 1, "deadline_time": "2026-10-17T10:00:00Z"}
        for g in range(1, TOTAL_GWS + 1)
    ]
    boot = {"teams": teams, "elements": elements, "events": events, "game_config": {"scoring": SCORING, "rules": {}}}
    return boot, fixtures


@pytest.fixture
def raw():
    return make_raw()


@pytest.fixture
def snap(raw):
    return snapshot_from_raw(*raw)


@pytest.fixture
def model(snap):
    return fit_model(snap, horizon=3)
