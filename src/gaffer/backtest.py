"""Walk-forward backtest of the xP model against baselines.

For each target gameweek t, everything (team ratings, player rates, team
baselines) is refit on gameweeks < t only, then scored against what actually
happened in t. Baselines:

* form: points per team game over the previous 4 gameweeks.
* ppg: season points per team game so far.

FPL's own archived xP is deliberately not a baseline: `history.fpl_xp_leakage`
shows it was captured after team news, so it would be an unfair comparison.

Beyond error metrics we score decisions: the actual points of the best £100m
squad's XI (captain doubled) when each predictor is fed into the same optimiser.

The population each week is every player who has played this season and is
registered for gameweek t, with their club as of t (so January transfers are
handled correctly).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .fpl import FPLClient, Snapshot
from .history import HIST_FEATURES
from .optimizer import optimise
from .strength import fit_ratings
from .xp import HALF_LIFE, RATE_HALF_LIFE, features_from_history, project_long

METRICS = ["mae", "spearman", "top20_actual_mean", "optimal_xi_actual_pts"]
HIGHER_IS_BETTER = {"mae": False, "spearman": True, "top20_actual_mean": True, "optimal_xi_actual_pts": True}


def load_history(client: FPLClient, snap: Snapshot, workers: int = 8) -> pd.DataFrame:
    """Per-fixture history for the current season from the live API."""
    ids = list(snap.players.index[snap.players.minutes > 0])
    fx = snap.fixtures.set_index("id")

    def fetch(pid):
        return pid, client.element_summary(pid, ttl=6 * 3600)["history"]

    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for pid, hist in pool.map(fetch, ids):
            for h in hist:
                f = fx.loc[h["fixture"]]
                if not f.finished:  # the API lists in-progress gameweeks with 0 minutes
                    continue
                row = {
                    "player_id": pid,
                    "round": h["round"],
                    "fixture": h["fixture"],
                    "team": int(f.team_h if h["was_home"] else f.team_a),
                    "points": h["total_points"],
                    "value": h["value"] / 10,
                }
                row.update({new: float(h.get(old) or 0) for old, new in HIST_FEATURES.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def _xi_points(pred: pd.Series, feat: pd.DataFrame, prices: pd.Series, actual: pd.Series) -> float:
    table = feat[["pos", "team"]].copy()
    table["price"] = prices.reindex(pred.index)
    table = table.dropna(subset=["price"])
    table["xp_next"] = pred
    table["xp_horizon"] = pred
    table["avail"] = 1.0
    plan = optimise(table, bank=100.0)
    pts = actual.reindex(plan.starting_xi).fillna(0)
    return float(pts.sum() + actual.get(plan.captain, 0))


def run_backtest(
    client: FPLClient | None,
    snap: Snapshot,
    start_gw: int = 4,
    hist: pd.DataFrame | None = None,
    end_gw: int | None = None,
    half_life: float | None = HALF_LIFE,
    rate_half_life: float | None = RATE_HALF_LIFE,
    decisions: bool = True,
) -> pd.DataFrame:
    hist = load_history(client, snap) if hist is None else hist
    done = snap.fixtures.groupby("event").finished.all()
    last = int(done[done].index.max())  # only score gameweeks that are fully played
    last = min(last, end_gw) if end_gw else last
    teams = list(snap.teams.index)
    rows = []
    for t in range(start_gw, last + 1):
        feat = features_from_history(hist, snap, t, half_life, rate_half_life)
        if feat.empty:
            continue
        now = hist[hist["round"] == t].groupby("player_id")
        actual = now.points.sum().reindex(feat.index).fillna(0)
        prices = now.value.first()

        ratings = fit_ratings(snap.fixtures, teams, before_gw=t)
        long = project_long(feat, ratings, snap.fixtures, [t], snap.scoring, before_gw=t)
        preds = {"model": long.groupby("player_id").xp.sum().reindex(feat.index).fillna(0)}

        gp_recent = feat.team.map(snap.team_games_played(before_gw=t) - snap.team_games_played(before_gw=t - 4))
        recent = hist[(hist["round"] >= t - 4) & (hist["round"] < t)].groupby("player_id").points.sum()
        preds["form"] = (recent.reindex(feat.index).fillna(0) / gp_recent.replace(0, np.nan)).fillna(0)
        season = hist[hist["round"] < t].groupby("player_id").points.sum()
        games = (
            hist[hist["round"] < t].groupby("player_id").size().reindex(feat.index)
        )  # unweighted, so baselines don't move with half_life
        preds["ppg"] = (season.reindex(feat.index).fillna(0) / games.replace(0, np.nan)).fillna(0)

        for name, pred in preds.items():
            top20 = pred.sort_values(ascending=False).head(20).index
            rows.append(
                {
                    "gw": t,
                    "predictor": name,
                    "n_players": len(pred),
                    "mae": float((pred - actual).abs().mean()),
                    "spearman": float(spearmanr(pred, actual).statistic),
                    "top20_actual_mean": float(actual[top20].mean()),
                    "optimal_xi_actual_pts": _xi_points(pred, feat, prices, actual) if decisions else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    s = results.groupby("predictor")[METRICS].mean()
    s["gameweeks"] = results.groupby("predictor").gw.nunique()
    return s.sort_values("spearman", ascending=False)


def head_to_head(results: pd.DataFrame, baseline: str) -> dict:
    """Relative improvement of the model over a baseline, and weekly win rate, per metric."""
    wide = results.pivot(index="gw", columns="predictor", values=METRICS)
    wide = wide[wide["mae"][baseline].notna()]  # only weeks where the baseline exists
    out = {}
    for m in METRICS:
        mod, base = wide[m]["model"], wide[m][baseline]
        wins = (mod < base) if not HIGHER_IS_BETTER[m] else (mod > base)
        change = (base.mean() - mod.mean()) / base.mean() if not HIGHER_IS_BETTER[m] else mod.mean() / base.mean() - 1
        out[m] = {"improvement_pct": round(100 * change, 1), "weeks_won": int(wins.sum()), "weeks": len(wins)}
    return out
