"""Walk-forward backtest of the xP model against simple baselines.

For each target gameweek t, everything (team ratings, player rates, team
baselines) is refit on gameweeks < t only, then scored against what actually
happened in t. Baselines are what a typical manager looks at:

* form: points per team game over the previous 4 gameweeks (FPL's "form").
* ppg: season points per team game so far.

Beyond error metrics we score decisions: the actual points of the best £100m
squad's XI (captain doubled) when each predictor is fed into the same optimiser.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .fpl import FPLClient, Snapshot
from .optimizer import optimise
from .strength import fit_ratings
from .xp import FEATURE_COLS, project_long

HISTORY_COLS = {
    "minutes": "minutes",
    "starts": "starts",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "defensive_contribution": "dc",
    "saves": "saves",
    "bonus": "bonus",
    "yellow_cards": "yellow",
}


def load_history(client: FPLClient, snap: Snapshot, workers: int = 8) -> pd.DataFrame:
    ids = list(snap.players.index[snap.players.minutes > 0])

    def fetch(pid):
        return pid, client.element_summary(pid, ttl=6 * 3600)["history"]

    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for pid, hist in pool.map(fetch, ids):
            for h in hist:
                row = {"player_id": pid, "round": h["round"], "points": h["total_points"], "value": h["value"] / 10}
                row.update({new: float(h.get(old) or 0) for old, new in HISTORY_COLS.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def _features_before(hist: pd.DataFrame, snap: Snapshot, t: int) -> pd.DataFrame:
    past = hist[hist["round"] < t]
    agg = past.groupby("player_id")[FEATURE_COLS].sum()
    feat = snap.players.loc[agg.index, ["pos", "team"]].join(agg)
    feat["gp"] = feat.team.map(snap.team_games_played(before_gw=t)).fillna(0)
    feat["avail"] = 1.0  # historical injury news isn't available, so the backtest is conservative
    return feat[feat.minutes > 0]


def _xi_points(pred: pd.Series, prices: pd.Series, snap: Snapshot, actual: pd.Series) -> float:
    table = snap.players.loc[pred.index, ["pos", "team"]].copy()
    table["price"] = prices.reindex(pred.index).fillna(snap.players.price)
    table["xp_next"] = pred
    table["xp_horizon"] = pred
    table["avail"] = 1.0
    plan = optimise(table, bank=100.0)
    pts = actual.reindex(plan.starting_xi).fillna(0)
    return float(pts.sum() + actual.get(plan.captain, 0))


def run_backtest(client: FPLClient, snap: Snapshot, start_gw: int = 4, hist: pd.DataFrame | None = None) -> pd.DataFrame:
    hist = load_history(client, snap) if hist is None else hist
    done = snap.fixtures.groupby("event").finished.all()
    last = int(done[done].index.max())  # only score gameweeks that are fully played
    teams = list(snap.teams.index)
    rows = []
    for t in range(start_gw, last + 1):
        feat = _features_before(hist, snap, t)
        ratings = fit_ratings(snap.fixtures, teams, before_gw=t)
        long = project_long(feat, ratings, snap.fixtures, [t], snap.scoring, before_gw=t)
        model = long.groupby("player_id").xp.sum().reindex(feat.index).fillna(0)

        gp_all = feat.gp.replace(0, np.nan)
        recent = hist[(hist["round"] >= t - 4) & (hist["round"] < t)].groupby("player_id").points.sum()
        gp_recent = feat.team.map(snap.team_games_played(before_gw=t) - snap.team_games_played(before_gw=t - 4))
        form = (recent.reindex(feat.index).fillna(0) / gp_recent.replace(0, np.nan)).fillna(0)
        season = hist[hist["round"] < t].groupby("player_id").points.sum()
        ppg = (season.reindex(feat.index).fillna(0) / gp_all).fillna(0)

        actual = hist[hist["round"] == t].groupby("player_id").points.sum().reindex(feat.index).fillna(0)
        prices = hist[hist["round"] == t].groupby("player_id").value.first()

        for name, pred in [("model", model), ("form", form), ("ppg", ppg)]:
            top20 = pred.sort_values(ascending=False).head(20).index
            rows.append(
                {
                    "gw": t,
                    "predictor": name,
                    "n_players": len(pred),
                    "mae": float((pred - actual).abs().mean()),
                    "spearman": float(spearmanr(pred, actual).statistic),
                    "top20_actual_mean": float(actual[top20].mean()),
                    "optimal_xi_actual_pts": _xi_points(pred, prices, snap, actual),
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    s = results.groupby("predictor")[["mae", "spearman", "top20_actual_mean", "optimal_xi_actual_pts"]].mean()
    return s.sort_values("spearman", ascending=False)
