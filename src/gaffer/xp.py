"""Expected FPL points (xP) per player per fixture.

Each scoring event is modelled as a probability, not a heuristic score:

* minutes: start-rate and minutes-per-team-game from the season so far,
  scaled by the injury-news availability flag.
* goals / assists: per-90 xG and xA, Bayesian-shrunk toward the positional
  mean (prior worth PRIOR_MINUTES), then scaled by how much more or less the
  team is expected to score in this fixture than in its average match so far.
* clean sheets / goals conceded / saves: Poisson in the opponent's expected
  goals from the team-strength model (`strength.py`).
* defensive contribution: P(Poisson(dc per 90) >= threshold).

Points per event come from the live FPL scoring config, so a rule change does
not need a code change. The same function serves the live model and the
walk-forward backtest, which is what keeps the backtest honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .fpl import Snapshot
from .strength import TeamRatings, fit_ratings

PRIOR_MINUTES = 270.0
P60_GIVEN_START = 0.88
SUB_MINUTES = 20.0
# Defensive contribution thresholds (CBIT for DEF, CBIRT for MID/FWD). Not in
# the API's scoring config, which only carries the points value.
DC_THRESHOLD = {"GKP": np.inf, "DEF": 10, "MID": 12, "FWD": 12}
FEATURE_COLS = ["minutes", "starts", "xg", "xa", "dc", "saves", "bonus", "yellow"]


def availability(players: pd.DataFrame) -> pd.Series:
    """Probability of being available, from FPL's status/news flags."""
    chance = players["chance"].astype(float) / 100.0
    unknown = chance.isna()
    by_status = players["status"].map({"a": 1.0, "d": 0.5}).fillna(0.0)
    return chance.where(~unknown, by_status).clip(0, 1)


def features_from_snapshot(snap: Snapshot) -> pd.DataFrame:
    f = snap.players[["pos", "team", *FEATURE_COLS]].copy()
    f["gp"] = f.team.map(snap.team_games_played()).fillna(0)
    f["avail"] = availability(snap.players)
    return f


def player_rates(feat: pd.DataFrame) -> pd.DataFrame:
    """Per-90 rates (shrunk), expected minutes and playing probabilities."""
    out = pd.DataFrame(index=feat.index)
    mins = feat.minutes.astype(float)
    gp = feat.gp.astype(float).replace(0, np.nan)

    regulars = feat[mins >= 450]
    for col in ["xg", "xa", "dc", "saves", "bonus", "yellow"]:
        raw = feat[col] / mins.clip(lower=1) * 90
        reg_raw = regulars[col] / regulars.minutes * 90
        prior = reg_raw.groupby(regulars.pos).mean().reindex(feat.pos).to_numpy()
        prior = np.nan_to_num(prior, nan=float(np.nan_to_num(reg_raw.mean())))
        out[f"{col}90"] = (raw * mins + prior * PRIOR_MINUTES) / (mins + PRIOR_MINUTES)

    start_rate = (feat.starts / gp).clip(0, 1).fillna(0)
    mins_pg = (mins / gp).clip(0, 90).fillna(0)
    sub_part = ((mins_pg - 80 * start_rate).clip(lower=0) / SUB_MINUTES).clip(0, 1)
    avail = feat.avail.astype(float)

    out["p_play"] = (start_rate + (1 - start_rate) * sub_part).clip(0, 1) * avail
    out["p60"] = start_rate * P60_GIVEN_START * avail
    out["exp_min"] = mins_pg * avail
    return out


def team_fixtures(fixtures: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    """Long table of (team, gw, opponent, is_home); doubles appear twice, blanks not at all."""
    fx = fixtures[fixtures.event.isin(gws)]
    home = pd.DataFrame({"team": fx.team_h, "opp": fx.team_a, "gw": fx.event, "home": True})
    away = pd.DataFrame({"team": fx.team_a, "opp": fx.team_h, "gw": fx.event, "home": False})
    return pd.concat([home, away], ignore_index=True)


def team_baselines(ratings: TeamRatings, fixtures: pd.DataFrame, before_gw: int | None) -> pd.DataFrame:
    """Mean expected goals for/against in the matches each team has already played.

    Player per-90s were accrued against exactly these opponents, so a fixture's
    attacking multiplier is lambda_for / baseline_for rather than raw lambda.
    """
    played = fixtures[fixtures.finished]
    if before_gw is not None:
        played = played[played.event < before_gw]
    rows = []
    for h, a in zip(played.team_h, played.team_a):
        lh, la = ratings.expected_goals(h, a)
        rows += [(h, lh, la), (a, la, lh)]
    if not rows:
        lh, la = np.exp(ratings.mu + ratings.home / 2), np.exp(ratings.mu + ratings.home / 2)
        return pd.DataFrame({"base_for": lh, "base_against": la}, index=ratings.team_ids)
    df = pd.DataFrame(rows, columns=["team", "base_for", "base_against"])
    return df.groupby("team").mean()


def _expected_floor_div(lam: np.ndarray, k: int, terms: int = 12) -> np.ndarray:
    """E[floor(X / k)] for X ~ Poisson(lam) = sum_j P(X >= j*k)."""
    lam = np.asarray(lam, float)
    return sum(poisson.sf(j * k - 1, lam) for j in range(1, terms + 1))


def project_long(
    feat: pd.DataFrame,
    ratings: TeamRatings,
    fixtures: pd.DataFrame,
    gws: list[int],
    scoring: dict,
    before_gw: int | None = None,
) -> pd.DataFrame:
    """One row per (player, fixture) with every xP component broken out."""
    rates = player_rates(feat)
    base = team_baselines(ratings, fixtures, before_gw)
    tf = team_fixtures(fixtures, gws)
    tf["lam_for"], tf["lam_against"] = (
        zip(
            *[
                ratings.expected_goals(t, o) if h else ratings.expected_goals(o, t)[::-1]
                for t, o, h in zip(tf.team, tf.opp, tf.home)
            ]
        )
        if len(tf)
        else ([], [])
    )

    df = feat[["pos", "team"]].join(rates).reset_index(names="player_id")
    df = df.merge(tf, on="team", how="inner")
    df = df.join(base, on="team")

    pos = df.pos.to_numpy()
    m = df.exp_min.to_numpy() / 90.0
    att_factor = (df.lam_for / df.base_for).to_numpy()
    def_factor = (df.lam_against / df.base_against).to_numpy()
    lam_ag = df.lam_against.to_numpy()

    gs = df.pos.map(scoring["goals_scored"]).to_numpy(float)
    cs_pts = df.pos.map(scoring["clean_sheets"]).to_numpy(float)
    gc_pts = df.pos.map(scoring["goals_conceded"]).to_numpy(float)
    dc_pts = df.pos.map(scoring["defensive_contribution"]).to_numpy(float)

    df["pts_appearance"] = df.p60 * scoring["long_play"] + (df.p_play - df.p60) * scoring["short_play"]
    df["exp_goals"] = df.xg90 * m * att_factor
    df["pts_goals"] = df.exp_goals * gs
    df["exp_assists"] = df.xa90 * m * att_factor
    df["pts_assists"] = df.exp_assists * scoring["assists"]
    df["p_clean_sheet"] = df.p60 * np.exp(-lam_ag)
    df["pts_clean_sheet"] = df.p_clean_sheet * cs_pts
    df["pts_conceded"] = gc_pts * _expected_floor_div(lam_ag * m, 2)
    is_gk = pos == "GKP"
    df["pts_saves"] = np.where(is_gk, scoring["saves"] * _expected_floor_div(df.saves90 * m * def_factor, 3), 0.0)
    thresh = df.pos.map(DC_THRESHOLD).to_numpy(float)
    p_dc = np.where(np.isfinite(thresh), poisson.sf(np.nan_to_num(thresh, posinf=0) - 1, df.dc90 * m), 0.0)
    df["pts_defcon"] = p_dc * dc_pts
    df["pts_bonus"] = df.bonus90 * m * scoring["bonus"]
    df["pts_cards"] = df.yellow90 * m * scoring["yellow_cards"]

    comp = [c for c in df.columns if c.startswith("pts_")]
    df["xp"] = df[comp].sum(axis=1)
    return df


@dataclass
class ModelState:
    """A fitted model for one gameweek horizon; the agent's tools read from this."""

    snap: Snapshot
    ratings: TeamRatings
    long: pd.DataFrame
    table: pd.DataFrame
    gws: list[int]

    @property
    def next_gw(self) -> int:
        return self.gws[0]


def fit_model(snap: Snapshot, horizon: int = 5) -> ModelState:
    gws = [g for g in range(snap.next_gw, snap.next_gw + horizon) if g <= 38]
    ratings = fit_ratings(snap.fixtures, list(snap.teams.index))
    feat = features_from_snapshot(snap)
    long = project_long(feat, ratings, snap.fixtures, gws, snap.scoring)

    per_gw = long.pivot_table(index="player_id", columns="gw", values="xp", aggfunc="sum")
    per_gw = per_gw.reindex(index=snap.players.index, columns=gws).fillna(0.0)
    per_gw.columns = [f"gw{g}" for g in gws]

    p = snap.players
    table = p[["name", "team_short", "team", "pos", "price", "status", "news", "selected_by"]].join(per_gw)
    table["avail"] = availability(p)
    table["xp_next"] = per_gw.iloc[:, 0]
    table["xp_horizon"] = per_gw.sum(axis=1)
    table["value"] = table.xp_horizon / table.price
    return ModelState(snap=snap, ratings=ratings, long=long, table=table, gws=gws)
