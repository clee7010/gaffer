"""Team attack/defence ratings from match results.

Model (Maher / Dixon-Coles without the low-score correction):

    goals_home ~ Poisson(exp(mu + home + att[h] - def[a]))
    goals_away ~ Poisson(exp(mu +        att[a] - def[h]))

fitted by L2-penalised maximum likelihood. Early in a season there are only a
handful of matches per team, so the ridge penalty shrinks every team toward
league average instead of trusting one 5-0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class TeamRatings:
    team_ids: list[int]
    mu: float
    home: float
    att: dict[int, float]
    dfn: dict[int, float]

    def expected_goals(self, home_team: int, away_team: int) -> tuple[float, float]:
        lam_h = np.exp(self.mu + self.home + self.att[home_team] - self.dfn[away_team])
        lam_a = np.exp(self.mu + self.att[away_team] - self.dfn[home_team])
        return float(lam_h), float(lam_a)

    def table(self, teams: pd.DataFrame | None = None) -> pd.DataFrame:
        df = pd.DataFrame({"att": self.att, "def": self.dfn})
        df["net"] = df["att"] + df["def"]
        if teams is not None:
            df.insert(0, "short", teams.short.reindex(df.index))
        return df.sort_values("net", ascending=False)


def fit_ratings(
    fixtures: pd.DataFrame,
    team_ids: list[int],
    l2: float = 3.0,
    before_gw: int | None = None,
) -> TeamRatings:
    played = fixtures[fixtures.finished]
    if before_gw is not None:
        played = played[played.event < before_gw]

    idx = {t: i for i, t in enumerate(team_ids)}
    n = len(team_ids)
    if played.empty:
        zeros = {t: 0.0 for t in team_ids}
        return TeamRatings(team_ids, float(np.log(1.35)), 0.2, zeros, dict(zeros))

    h = played.team_h.map(idx).to_numpy()
    a = played.team_a.map(idx).to_numpy()
    # Stack home-goals and away-goals rows into one long design.
    scorer = np.concatenate([h, a])
    conceder = np.concatenate([a, h])
    is_home = np.concatenate([np.ones(len(h)), np.zeros(len(a))])
    y = np.concatenate([played.h_score.to_numpy(float), played.a_score.to_numpy(float)])

    def unpack(theta):
        return theta[0], theta[1], theta[2 : 2 + n], theta[2 + n :]

    def objective(theta):
        mu, home, att, dfn = unpack(theta)
        eta = mu + home * is_home + att[scorer] - dfn[conceder]
        lam = np.exp(eta)
        nll = np.sum(lam - y * eta) + 0.5 * l2 * (att @ att + dfn @ dfn)
        r = lam - y  # d nll / d eta
        g_att = np.bincount(scorer, weights=r, minlength=n) + l2 * att
        g_dfn = -np.bincount(conceder, weights=r, minlength=n) + l2 * dfn
        grad = np.concatenate([[r.sum(), (r * is_home).sum()], g_att, g_dfn])
        return nll, grad

    theta0 = np.zeros(2 + 2 * n)
    theta0[0] = np.log(max(y.mean(), 0.1))
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    mu, home, att, dfn = unpack(res.x)
    return TeamRatings(
        team_ids=list(team_ids),
        mu=float(mu),
        home=float(home),
        att={t: float(att[i]) for t, i in idx.items()},
        dfn={t: float(dfn[i]) for t, i in idx.items()},
    )
