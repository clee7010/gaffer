import numpy as np
import pandas as pd

from gaffer.backtest import load_history, run_backtest
from gaffer.fpl import FPLClient
from gaffer.xp import features_from_history


def hist_rows(snap, rounds, minutes_by_round):
    """One player per club-row history: player 1 plays the given minutes each round."""
    fx = snap.fixtures
    rows = []
    for r in rounds:
        f = fx[(fx.event == r) & ((fx.team_h == 1) | (fx.team_a == 1))].iloc[0]
        rows.append(
            {
                "player_id": 1,
                "round": r,
                "fixture": f.id,
                "team": 1,
                "points": 2.0,
                "value": 5.0,
                "minutes": minutes_by_round[r],
                "starts": float(minutes_by_round[r] > 0),
                "xg": 0.3,
                "xa": 0.1,
                "dc": 5.0,
                "saves": 3.0,
                "bonus": 0.0,
                "yellow": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_decay_weights_recent_minutes(snap):
    # Benched in the most recent gameweek after two full games.
    hist = hist_rows(snap, [1, 2, 3], {1: 90, 2: 90, 3: 0})
    hist = pd.concat([hist, hist_rows(snap, [4], {4: 0})])
    flat = features_from_history(hist, snap, 4, half_life=None)
    decayed = features_from_history(hist, snap, 4, half_life=1.0)
    assert flat.at[1, "minutes"] / flat.at[1, "gp"] == 60  # 180 mins over 3 games
    assert decayed.at[1, "minutes"] / decayed.at[1, "gp"] < 60 * 0.75  # last game dominates
    # Per-90 output is not decayed by default.
    assert decayed.at[1, "rate_minutes"] == 180


def test_backtest_population_requires_registration(snap):
    hist = hist_rows(snap, [1, 2, 3], {1: 90, 2: 90, 3: 90})  # no row in GW4: left the league
    assert features_from_history(hist, snap, 4).empty
    assert not features_from_history(hist, snap, 4, registered_only=False).empty


def test_live_history_skips_unplayed_fixtures(snap, tmp_path):
    """FPL lists kicked-off-but-unfinished gameweeks with 0 minutes; they must not count."""
    played = snap.fixtures[snap.fixtures.finished].iloc[0]
    pending = snap.fixtures[~snap.fixtures.finished].iloc[0]
    snap.players.loc[:, "minutes"] = 0
    snap.players.loc[1, "minutes"] = 90

    def fetch(path):
        assert path == "/element-summary/1/"
        row = {"round": 1, "total_points": 6, "value": 50, "was_home": True, "minutes": 90, "starts": 1}
        return {
            "history": [
                {**row, "fixture": int(played.id)},
                {**row, "fixture": int(pending.id), "round": int(pending.event), "minutes": 0},
            ]
        }

    hist = load_history(FPLClient(cache_dir=tmp_path, fetch=fetch), snap)
    assert hist.fixture.tolist() == [played.id]


def test_backtest_runs_on_synthetic_season(snap):
    rng = np.random.default_rng(0)
    rows = []
    for gw in range(1, 4):
        for pid, p in snap.players.iterrows():
            rows.append(
                {
                    "player_id": pid,
                    "round": gw,
                    "fixture": 0,
                    "team": p.team,
                    "points": float(rng.integers(0, 6)),
                    "value": p.price,
                    "minutes": float(p.minutes / 3),
                    "starts": float(p.starts / 3),
                    "xg": p.xg / 3,
                    "xa": p.xa / 3,
                    "dc": p.dc / 3,
                    "saves": p.saves / 3,
                    "bonus": 0.0,
                    "yellow": 0.0,
                }
            )
    res = run_backtest(None, snap, start_gw=3, hist=pd.DataFrame(rows), decisions=False)
    assert set(res.predictor) == {"model", "form", "ppg"} and res.gw.unique().tolist() == [3]
