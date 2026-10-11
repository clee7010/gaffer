import pytest

from gaffer.guardrails import validate
from gaffer.optimizer import optimise
from gaffer.team import TeamState


def rec_from_plan(plan, model, transfers=(), hit=0, chip="none"):
    t = model.table
    xp = sum(t.at[i, "xp_next"] for i in plan.starting_xi) + t.at[plan.captain, "xp_next"]
    return {
        "transfers": list(transfers),
        "chip": chip,
        "starting_xi": list(plan.starting_xi),
        "bench": list(plan.bench),
        "captain": plan.captain,
        "vice_captain": plan.vice_captain,
        "points_hit": hit,
        "expected_points_next_gw": round(float(xp) - hit, 2),
        "key_claims": [{"player_id": plan.captain, "metric": "xp_next", "value": round(float(t.at[plan.captain, "xp_next"]), 2)}],
        "summary": "test",
    }


@pytest.fixture
def drafted(model):
    plan = optimise(model.table, bank=100.0)
    return plan, TeamState(team_id=None)


@pytest.fixture
def managed(model):
    """An existing squad with £0.5m in the bank and 1 free transfer."""
    plan = optimise(model.table, bank=95.0)
    t = model.table
    team = TeamState(
        team_id=1, squad=plan.squad, bank=0.5, free_transfers=1, sell_prices={i: float(t.at[i, "price"]) for i in plan.squad}
    )
    return plan, team


def codes(report):
    return {e.code for e in report.errors}


def test_optimiser_output_passes(model, drafted):
    plan, team = drafted
    report = validate(rec_from_plan(plan, model), team, model)
    assert report.passed, report.errors


def test_club_limit(model, drafted):
    plan, team = drafted
    t = model.table
    rec = rec_from_plan(plan, model)
    # Fill the outfield bench with like-for-like players from the most-owned club.
    club = t.loc[plan.squad].team.value_counts().idxmax()
    for slot, victim in enumerate(rec["bench"][1:], start=1):
        same = t[(t.team == club) & (t.pos == t.at[victim, "pos"]) & ~t.index.isin(rec["starting_xi"] + rec["bench"])]
        rec["bench"][slot] = int(same.index[0])
    assert (t.loc[rec["starting_xi"] + rec["bench"]].team == club).sum() > 3
    assert "CLUB_LIMIT" in codes(validate(rec, team, model))


def test_captain_must_start(model, drafted):
    plan, team = drafted
    rec = rec_from_plan(plan, model)
    rec["captain"] = plan.bench[1]
    assert "CAPTAIN_NOT_IN_XI" in codes(validate(rec, team, model))


def test_unknown_player_short_circuits(model, drafted):
    plan, team = drafted
    rec = rec_from_plan(plan, model)
    rec["starting_xi"][0] = 999_999
    assert codes(validate(rec, team, model)) == {"UNKNOWN_PLAYER"}


def test_invented_numbers_are_caught(model, drafted):
    plan, team = drafted
    rec = rec_from_plan(plan, model)
    rec["key_claims"][0]["value"] += 3.0
    rec["expected_points_next_gw"] += 10
    assert {"CLAIM_MISMATCH", "XP_TOTAL_MISMATCH"} <= codes(validate(rec, team, model))


def test_injured_starter_blocked(model, drafted):
    plan, team = drafted
    model.table.loc[plan.starting_xi[3], ["avail", "news"]] = [0.0, "Hamstring - out for 4 weeks"]
    assert "UNAVAILABLE_STARTER" in codes(validate(rec_from_plan(plan, model), team, model))


def test_doubt_is_warning_not_error(model, drafted):
    plan, team = drafted
    model.table.loc[plan.starting_xi[3], ["avail", "news"]] = [0.75, "Knock - 75% chance of playing"]
    report = validate(rec_from_plan(plan, model), team, model)
    assert report.passed and report.warnings[0].code == "DOUBTFUL_STARTER"


def _two_transfers(plan, model, team):
    """Swap two bench outfielders for the cheapest legal like-for-like replacements."""
    t = model.table
    club_counts = t.loc[team.squad].team.value_counts()
    pairs, taken = [], set(team.squad)
    for out in plan.bench[1:3]:
        cand = t[(t.pos == t.at[out, "pos"]) & ~t.index.isin(taken)]
        cand = cand[cand.team.map(club_counts).fillna(0) == 0].sort_values("price")
        pick = int(cand.index[0])
        taken.add(pick)
        club_counts[t.at[pick, "team"]] = 1
        pairs.append({"out": out, "in": pick})
    return pairs


def _apply_to_bench(rec, transfers):
    swap = {p["out"]: p["in"] for p in transfers}
    rec["bench"] = [swap.get(i, i) for i in rec["bench"]]
    return rec


def test_hit_must_be_declared(model, managed):
    plan, team = managed
    transfers = _two_transfers(plan, model, team)
    squad = [i for i in plan.squad if i not in {p["out"] for p in transfers}] + [p["in"] for p in transfers]
    rec = _apply_to_bench(rec_from_plan(plan, model, transfers=transfers, hit=0), transfers)
    assert set(rec["starting_xi"] + rec["bench"]) == set(squad)
    report = validate(rec, team, model)
    assert "HIT_MISCOUNT" in codes(report) and report.computed["points_hit"] == 4

    rec["points_hit"] = 4
    rec["expected_points_next_gw"] -= 4
    assert validate(rec, team, model).passed


def test_wildcard_makes_transfers_free(model, managed):
    plan, team = managed
    transfers = _two_transfers(plan, model, team)
    rec = _apply_to_bench(rec_from_plan(plan, model, transfers=transfers, chip="wildcard"), transfers)
    assert validate(rec, team, model).passed


def test_over_budget(model, managed):
    plan, team = managed
    t = model.table
    out = plan.bench[1]
    pricey = int(t[(t.pos == t.at[out, "pos"]) & ~t.index.isin(team.squad)].price.idxmax())
    assert t.at[pricey, "price"] > t.at[out, "price"] + team.bank
    transfers = [{"out": out, "in": pricey}]
    rec = _apply_to_bench(rec_from_plan(plan, model, transfers=transfers), transfers)
    assert "OVER_BUDGET" in codes(validate(rec, team, model))


def test_position_swap_rejected(model, managed):
    plan, team = managed
    t = model.table
    out = plan.bench[0]  # the reserve keeper
    fwd = int(t[(t.pos == "FWD") & ~t.index.isin(team.squad)].index[0])
    rec = rec_from_plan(plan, model, transfers=[{"out": out, "in": fwd}])
    assert "POSITION_SWAP" in codes(validate(rec, team, model))


def test_chip_already_used(model, managed):
    plan, team = managed
    team.chips_used = ["bboost@GW2"]
    rec = rec_from_plan(plan, model, chip="bench_boost")
    rec["expected_points_next_gw"] += sum(model.table.at[i, "xp_next"] for i in plan.bench)
    assert codes(validate(rec, team, model)) == {"CHIP_UNAVAILABLE"}
