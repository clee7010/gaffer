"""Tools the agent can call. Each returns JSON-serialisable data."""

from __future__ import annotations

import math

import pandas as pd

from .guardrails import GuardrailReport, validate
from .optimizer import optimise
from .team import TeamState
from .xp import ModelState, team_fixtures

POSITION_ENUM = ["GKP", "DEF", "MID", "FWD"]

TOOL_SCHEMAS = [
    {
        "name": "get_team_overview",
        "description": (
            "The manager's current squad with model projections, bank, estimated free transfers, chips used, "
            "the upcoming gameweek and its deadline. Call this first. If there is no squad, you are building "
            "one from scratch with a £100.0m budget."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "search_players",
        "description": (
            "Search and rank players by projected points. Every filter is optional. Returns id, name, club, "
            "position, price, availability, xp_next (next gameweek), xp_horizon (sum over the horizon) and "
            "value (xp_horizon per £m)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {"type": "string", "enum": POSITION_ENUM},
                "max_price": {"type": "number", "description": "Maximum price in £m, e.g. 7.5"},
                "club": {"type": "string", "description": "Three-letter club code, e.g. ARS"},
                "name": {"type": "string", "description": "Case-insensitive substring of the player's name"},
                "sort_by": {"type": "string", "enum": ["xp_next", "xp_horizon", "value", "selected_by"]},
                "limit": {"type": "integer", "description": "Max rows (default 10, max 40)"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_player",
        "description": (
            "Full detail for one player: news, availability, and the per-fixture expected points breakdown "
            "(appearance, goals, assists, clean sheet, saves, defensive contribution, bonus) with the opponent "
            "and the team-strength model's expected goals for and against."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"player_id": {"type": "integer"}},
            "required": ["player_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fixture_outlook",
        "description": (
            "Fixture difficulty from the Poisson team-strength model: for each club and gameweek in the "
            "horizon, the opponent, expected goals for/against and clean-sheet probability. Ranked by the "
            "chosen metric over the horizon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rank_by": {"type": "string", "enum": ["attack", "defence"]},
                "clubs": {"type": "array", "items": {"type": "string"}, "description": "Optional club codes"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "optimise_transfers",
        "description": (
            "Solve the integer program for the best squad under every FPL rule (budget, 2/5/5/3, max 3 per club, "
            "formation), trading off horizon xP against -4 hits. Returns transfers, XI, bench, captain and "
            "projected points. Use it to check or improve your own ideas; you can lock or ban players."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_transfers": {"type": "integer", "description": "Cap on transfers (omit for no cap)"},
                "locked_ids": {"type": "array", "items": {"type": "integer"}, "description": "Must be in squad"},
                "banned_ids": {"type": "array", "items": {"type": "integer"}, "description": "Must not be bought"},
                "wildcard": {"type": "boolean", "description": "Unlimited free transfers this week"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_recommendation",
        "description": (
            "Submit the final recommendation. It is checked by deterministic guardrails: FPL squad rules, "
            "budget, the points hit, availability of starters, and that every number you quote matches the "
            "model. If it fails you get the violations back. Fix them and submit again. Nothing reaches the "
            "manager until this passes."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "transfers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"out": {"type": "integer"}, "in": {"type": "integer"}},
                        "required": ["out", "in"],
                        "additionalProperties": False,
                    },
                },
                "chip": {"type": "string", "enum": ["none", "wildcard", "free_hit", "bench_boost", "triple_captain"]},
                "starting_xi": {"type": "array", "items": {"type": "integer"}},
                "bench": {"type": "array", "items": {"type": "integer"}, "description": "Bench order; reserve GK first"},
                "captain": {"type": "integer"},
                "vice_captain": {"type": "integer"},
                "points_hit": {"type": "integer", "description": "Points deducted for extra transfers (0, 4, 8, ...)"},
                "expected_points_next_gw": {
                    "type": "number",
                    "description": "Model xP of the XI next gameweek, captain counted twice, minus the hit",
                },
                "key_claims": {
                    "type": "array",
                    "description": "Every number you cite in the summary, so it can be fact-checked",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_id": {"type": "integer"},
                            "metric": {"type": "string", "enum": ["xp_next", "xp_horizon", "price"]},
                            "value": {"type": "number"},
                        },
                        "required": ["player_id", "metric", "value"],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string", "description": "The advice for the manager, in your voice"},
            },
            "required": [
                "transfers",
                "chip",
                "starting_xi",
                "bench",
                "captain",
                "vice_captain",
                "points_hit",
                "expected_points_next_gw",
                "key_claims",
                "summary",
            ],
            "additionalProperties": False,
        },
    },
]


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = df.reset_index(names="id")[["id", *cols]].copy()
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(2)
    return out.to_dict(orient="records")


class Toolbox:
    def __init__(self, model: ModelState, team: TeamState):
        self.model = model
        self.team = team
        self.last_report: GuardrailReport | None = None
        self.accepted: dict | None = None

    @property
    def t(self) -> pd.DataFrame:
        return self.model.table

    def call(self, name: str, args: dict):
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            raise KeyError(f"Unknown tool {name}")
        return fn(**args)

    # --- tools --------------------------------------------------------------
    def tool_get_team_overview(self):
        snap, team = self.model.snap, self.team
        info = {
            "next_gameweek": self.model.next_gw,
            "deadline": snap.deadline,
            "horizon_gameweeks": self.model.gws,
            "mode": "draft (no squad yet, £100.0m budget)" if team.drafting else "manage existing squad",
        }
        if team.drafting:
            return info
        squad = self.t.loc[team.squad]
        cols = ["name", "team_short", "pos", "price", "avail", "news", "xp_next", "xp_horizon"]
        return {
            **info,
            "team_name": team.team_name,
            "overall_rank": team.overall_rank,
            "bank": team.bank,
            "free_transfers": team.free_transfers,
            "chips_used": team.chips_used,
            "squad": _records(squad.sort_values(["pos", "xp_next"], ascending=[True, False]), cols),
            "squad_xp_next_sum": round(float(squad.xp_next.sum()), 2),
            "data_notes": team.notes,
        }

    def tool_search_players(self, position=None, max_price=None, club=None, name=None, sort_by="xp_next", limit=10):
        df = self.t
        if position:
            df = df[df.pos == position]
        if max_price is not None:
            df = df[df.price <= max_price + 1e-9]
        if club:
            df = df[df.team_short.str.upper() == club.upper()]
        if name:
            df = df[df.name.str.contains(name, case=False, regex=False)]
        limit = max(1, min(int(limit or 10), 40))
        df = df.sort_values(sort_by or "xp_next", ascending=False).head(limit)
        cols = ["name", "team_short", "pos", "price", "avail", "xp_next", "xp_horizon", "value", "selected_by"]
        rows = _records(df, cols)
        owned = set(self.team.squad)
        for r in rows:
            r["owned"] = r["id"] in owned
        return rows

    def tool_get_player(self, player_id: int):
        if player_id not in self.t.index:
            raise ValueError(f"No player with id {player_id}")
        p = self.t.loc[player_id]
        teams = self.model.snap.teams.short
        fx = self.model.long[self.model.long.player_id == player_id].copy()
        fx["opponent"] = fx.opp.map(teams) + fx.home.map({True: " (H)", False: " (A)"})
        comp = ["gw", "opponent", "lam_for", "lam_against", "exp_goals", "exp_assists", "p_clean_sheet", "xp"]
        comp += [c for c in fx.columns if c.startswith("pts_")]
        fixtures = fx[comp].round(2).to_dict(orient="records")
        return {
            "id": player_id,
            "name": p["name"],
            "club": p.team_short,
            "position": p.pos,
            "price": p.price,
            "status": p.status,
            "availability": round(float(p.avail), 2),
            "news": p.news,
            "selected_by_pct": p.selected_by,
            "xp_next": round(float(p.xp_next), 2),
            "xp_horizon": round(float(p.xp_horizon), 2),
            "owned": player_id in self.team.squad,
            "fixtures": fixtures,
        }

    def tool_fixture_outlook(self, rank_by="attack", clubs=None):
        teams = self.model.snap.teams.short
        tf = team_fixtures(self.model.snap.fixtures, self.model.gws)
        r = self.model.ratings
        rows = []
        for t, o, gw, home in zip(tf.team, tf.opp, tf.gw, tf.home):
            lf, la = r.expected_goals(t, o) if home else r.expected_goals(o, t)[::-1]
            rows.append(
                {
                    "club": teams[t],
                    "gw": int(gw),
                    "vs": f"{teams[o]} ({'H' if home else 'A'})",
                    "xg_for": round(lf, 2),
                    "xg_against": round(la, 2),
                    "p_clean_sheet": round(math.exp(-la), 2),
                }
            )
        df = pd.DataFrame(rows)
        if clubs:
            df = df[df.club.isin([c.upper() for c in clubs])]
        key = "xg_for" if rank_by == "attack" else "p_clean_sheet"
        order = df.groupby("club")[key].sum().sort_values(ascending=False)
        return [
            {
                "club": club,
                f"total_{key}": round(float(total), 2),
                "fixtures": df[df.club == club].drop(columns="club").to_dict(orient="records"),
            }
            for club, total in order.items()
        ]

    def tool_optimise_transfers(self, max_transfers=None, locked_ids=(), banned_ids=(), wildcard=False):
        team = self.team
        plan = optimise(
            self.t,
            current_squad=team.squad,
            bank=team.bank,
            sell_prices=team.sell_prices,
            free_transfers=15 if wildcard else team.free_transfers,
            max_transfers=max_transfers,
            locked=list(locked_ids),
            banned=list(banned_ids),
        )
        if plan.status != "Optimal":
            return {"status": plan.status, "message": "No feasible squad under those constraints."}
        nm = lambda ids: [f"{self.t.at[i, 'name']} ({i}, {self.t.at[i, 'pos']}, £{self.t.at[i, 'price']}m)" for i in ids]
        return {
            "status": plan.status,
            "transfers_out": nm(plan.transfers_out),
            "transfers_in": nm(plan.transfers_in),
            "points_hit": 4 * plan.hits,
            "starting_xi": nm(plan.starting_xi),
            "bench_in_order": nm(plan.bench),
            "captain": nm([plan.captain])[0],
            "vice_captain": nm([plan.vice_captain])[0],
            "xp_next_gw_lineup_incl_captain": round(plan.xp_next_xi, 2),
            "objective_horizon": round(plan.objective, 2),
            "bank_after": plan.bank_after,
            "ids": {
                "starting_xi": plan.starting_xi,
                "bench": plan.bench,
                "captain": plan.captain,
                "vice_captain": plan.vice_captain,
                "transfers": plan.transfer_pairs(self.t),
            },
        }

    def tool_submit_recommendation(self, **rec):
        report = validate(rec, self.team, self.model)
        self.last_report = report
        if report.passed:
            self.accepted = {**rec, "guardrails": report.to_dict()}
        return report.to_dict()
