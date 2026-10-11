import numpy as np
import pandas as pd

from gaffer.optimizer import optimise
from gaffer.strength import fit_ratings
from gaffer.team import estimate_free_transfers
from gaffer.xp import _expected_floor_div, fit_model


def test_ratings_recover_known_strengths():
    rng = np.random.default_rng(0)
    n = 10
    att, dfn = rng.normal(0, 0.35, n), rng.normal(0, 0.35, n)
    rows = []
    for _ in range(12):  # 12 double round-robins = plenty of data
        for h in range(n):
            for a in range(n):
                if h != a:
                    rows.append(
                        {
                            "team_h": h + 1,
                            "team_a": a + 1,
                            "event": 1,
                            "finished": True,
                            "h_score": rng.poisson(np.exp(0.2 + 0.25 + att[h] - dfn[a])),
                            "a_score": rng.poisson(np.exp(0.2 + att[a] - dfn[h])),
                        }
                    )
    r = fit_ratings(pd.DataFrame(rows), list(range(1, n + 1)), l2=0.5)
    assert np.corrcoef(att, [r.att[i + 1] for i in range(n)])[0, 1] > 0.9
    assert np.corrcoef(dfn, [r.dfn[i + 1] for i in range(n)])[0, 1] > 0.9
    assert 0.1 < r.home < 0.4


def test_ratings_shrink_with_no_data():
    r = fit_ratings(pd.DataFrame(columns=["team_h", "team_a", "event", "finished", "h_score", "a_score"]), [1, 2])
    assert r.att == {1: 0.0, 2: 0.0}


def test_expected_floor_div_matches_simulation():
    rng = np.random.default_rng(1)
    for lam in [0.3, 1.4, 3.0]:
        sim = (rng.poisson(lam, 200_000) // 2).mean()
        assert abs(_expected_floor_div(np.array([lam]), 2)[0] - sim) < 0.01


def test_injured_player_projects_zero(snap):
    pid = snap.players.index[snap.players.minutes > 200][0]
    snap.players.loc[pid, ["status", "chance"]] = ["i", 0]
    model = fit_model(snap, horizon=3)
    assert model.table.at[pid, "xp_horizon"] == 0
    assert model.table.at[pid, "avail"] == 0


def test_blank_gameweek_projects_zero_and_double_adds_up(snap):
    gw = snap.next_gw
    fx = snap.fixtures
    blank = fx[(fx.event == gw)].iloc[0]
    # Move one fixture out of the next gameweek: both clubs blank.
    snap.fixtures.loc[blank.name, "event"] = 99
    model = fit_model(snap, horizon=2)
    blanked = model.table[model.table.team == blank.team_h]
    assert (blanked[f"gw{gw}"] == 0).all()

    # Put it in the following gameweek: those clubs now have a double.
    snap.fixtures.loc[blank.name, "event"] = gw + 1
    doubled = fit_model(snap, horizon=2).table
    star = blanked.xp_horizon.idxmax()
    assert doubled.at[star, f"gw{gw + 1}"] > model.table.at[star, f"gw{gw + 1}"] * 1.5


def test_only_goalkeepers_get_save_points(model):
    long = model.long
    assert (long.loc[long.pos != "GKP", "pts_saves"] == 0).all()
    assert (long.loc[long.pos == "GKP", "pts_saves"] >= 0).all()
    assert long.loc[long.pos == "GKP", "pts_saves"].max() > 0


def test_draft_optimiser_obeys_rules(model):
    plan = optimise(model.table, bank=100.0)
    t = model.table.loc[plan.squad]
    assert plan.status == "Optimal"
    assert len(plan.squad) == 15 and len(plan.starting_xi) == 11 and len(plan.bench) == 4
    assert t.pos.value_counts().to_dict() == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert t.team.value_counts().max() <= 3
    assert t.price.sum() <= 100.0 + 1e-6
    assert plan.captain in plan.starting_xi and plan.captain != plan.vice_captain
    assert model.table.at[plan.bench[0], "pos"] == "GKP"


def test_optimiser_only_takes_hits_that_pay(model):
    squad = optimise(model.table, bank=100.0).squad
    # Re-optimising the optimal squad with nothing changed should make no transfers.
    plan = optimise(model.table, current_squad=squad, bank=0.0, free_transfers=1)
    assert plan.transfers_in == [] and plan.hits == 0


def test_free_transfer_rollover():
    hist = {"current": [{"event": g, "event_transfers": n} for g, n in [(1, 0), (2, 0), (3, 2), (4, 0)]], "chips": []}
    # Enter GW2 with 1, roll to 2, use 2 -> 0 +1 = 1, roll -> 2 entering GW5.
    assert estimate_free_transfers(hist, started_event=1, upto_gw=5) == 2
    hist["current"] = [{"event": g, "event_transfers": 0} for g in range(1, 10)]
    assert estimate_free_transfers(hist, started_event=1, upto_gw=10) == 5  # capped
