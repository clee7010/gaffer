"""Deterministic guardrails on the agent's final recommendation.

The LLM never gets the last word on anything checkable. Every recommendation
goes through `validate` before it is accepted or stored:

* legality: FPL squad rules (15 players, 2/5/5/3, max 3 per club, budget,
  valid formation, captaincy, bench order, chip availability);
* arithmetic: the points hit it quotes must match the transfer count;
* grounding: every number it quotes (xP, prices) must match the model to a
  tolerance, so it cannot invent stats to justify a pick;
* safety: no injured or suspended starters.

Errors bounce back to the agent as a tool error so it can fix them; warnings
are attached to the accepted recommendation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from .fpl import SQUAD_SHAPE
from .optimizer import HIT_COST, MAX_XI, MIN_XI
from .team import TeamState
from .xp import ModelState

XP_ABS_TOL = 0.25
XP_REL_TOL = 0.05
CHIP_CODES = {"wildcard": "wildcard", "free_hit": "freehit", "bench_boost": "bboost", "triple_captain": "3xc"}


@dataclass
class Issue:
    code: str
    message: str


@dataclass
class GuardrailReport:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    computed: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"passed": self.passed, **asdict(self)}

    def error(self, code: str, msg: str):
        self.errors.append(Issue(code, msg))

    def warn(self, code: str, msg: str):
        self.warnings.append(Issue(code, msg))


def _close(claimed: float, actual: float) -> bool:
    return abs(claimed - actual) <= max(XP_ABS_TOL, XP_REL_TOL * abs(actual))


def validate(rec: dict, team: TeamState, model: ModelState) -> GuardrailReport:
    r = GuardrailReport()
    t = model.table
    name = lambda i: f"{t.at[i, 'name']} ({i})" if i in t.index else str(i)

    xi, bench = list(rec.get("starting_xi", [])), list(rec.get("bench", []))
    transfers = rec.get("transfers", [])
    outs = [tr["out"] for tr in transfers]
    ins = [tr["in"] for tr in transfers]
    chip = rec.get("chip", "none")

    referenced = set(xi + bench + outs + ins + [rec.get("captain"), rec.get("vice_captain")])
    referenced |= {c["player_id"] for c in rec.get("key_claims", [])}
    unknown = sorted((i for i in referenced if i not in t.index), key=str)
    if unknown:
        r.error("UNKNOWN_PLAYER", f"Player ids not in the FPL database: {unknown}")
        return r  # nothing else is checkable

    # --- transfers -> resulting squad -------------------------------------
    current = set(team.squad)
    if team.drafting:
        if transfers:
            r.error("DRAFT_HAS_TRANSFERS", "No existing squad: leave transfers empty and pick 15 players.")
        squad = set(xi + bench)
    else:
        for i in outs:
            if i not in current:
                r.error("TRANSFER_OUT_NOT_OWNED", f"{name(i)} is not in the current squad.")
        for i in ins:
            if i in current:
                r.error("TRANSFER_IN_ALREADY_OWNED", f"{name(i)} is already in the squad.")
        dupes = [i for i, n in Counter(outs + ins).items() if n > 1]
        if dupes:
            r.error("DUPLICATE_TRANSFER", f"Players appear in more than one transfer: {dupes}")
        for tr in transfers:
            if tr["out"] in t.index and tr["in"] in t.index and t.at[tr["out"], "pos"] != t.at[tr["in"], "pos"]:
                r.error(
                    "POSITION_SWAP",
                    f"Transfers must be like-for-like: {name(tr['out'])} is {t.at[tr['out'], 'pos']}, "
                    f"{name(tr['in'])} is {t.at[tr['in'], 'pos']}.",
                )
        squad = (current - set(outs)) | set(ins)

    lineup = xi + bench
    if len(xi) != 11 or len(bench) != 4 or len(set(lineup)) != 15:
        r.error("LINEUP_SIZE", f"Need 11 distinct starters and 4 bench players; got {len(xi)} + {len(bench)}.")
    if set(lineup) != squad:
        missing, extra = squad - set(lineup), set(lineup) - squad
        r.error(
            "LINEUP_MISMATCH",
            f"XI + bench must equal the post-transfer squad. Missing: {[name(i) for i in missing]}; "
            f"not in squad: {[name(i) for i in extra]}.",
        )

    # --- squad legality ---------------------------------------------------
    pos_counts = Counter(t.loc[list(squad), "pos"])
    for pos, n in SQUAD_SHAPE.items():
        if pos_counts.get(pos, 0) != n:
            r.error("SQUAD_SHAPE", f"Squad needs {n} {pos}, has {pos_counts.get(pos, 0)}.")
    for club, n in Counter(t.loc[list(squad), "team_short"]).items():
        if n > 3:
            r.error("CLUB_LIMIT", f"{n} players from {club}; the limit is 3.")

    budget = team.bank + sum(team.sell_prices.get(i, t.at[i, "price"]) for i in current)
    cost = sum(team.sell_prices.get(i, t.at[i, "price"]) if i in current else t.at[i, "price"] for i in squad)
    bank_after = round(budget - cost, 1)
    r.computed["bank_after"] = bank_after
    if bank_after < -0.05:
        r.error("OVER_BUDGET", f"Squad costs £{cost:.1f}m against £{budget:.1f}m available (short £{-bank_after:.1f}m).")

    xi_pos = Counter(t.loc[xi, "pos"]) if xi else Counter()
    for pos in SQUAD_SHAPE:
        if not MIN_XI[pos] <= xi_pos.get(pos, 0) <= MAX_XI[pos]:
            r.error("FORMATION", f"XI has {xi_pos.get(pos, 0)} {pos}; allowed {MIN_XI[pos]}-{MAX_XI[pos]}.")
    if bench and t.at[bench[0], "pos"] != "GKP":
        r.error("BENCH_ORDER", "The first bench slot must be the reserve goalkeeper.")

    cap, vice = rec.get("captain"), rec.get("vice_captain")
    if cap not in xi:
        r.error("CAPTAIN_NOT_IN_XI", f"Captain {name(cap)} is not in the starting XI.")
    if vice not in xi:
        r.error("VICE_NOT_IN_XI", f"Vice-captain {name(vice)} is not in the starting XI.")
    if cap == vice:
        r.error("CAPTAIN_EQUALS_VICE", "Captain and vice-captain must be different players.")

    # --- chips and hits ---------------------------------------------------
    if chip != "none":
        half = 1 if model.next_gw <= 19 else 2
        used = [u for u in team.chips_used if u.startswith(CHIP_CODES[chip] + "@")]
        used_this_half = [u for u in used if (int(u.split("@GW")[1]) <= 19) == (half == 1)]
        if used_this_half:
            r.error("CHIP_UNAVAILABLE", f"{chip} already used this half of the season ({used_this_half[0]}).")
        if team.drafting:
            r.error("CHIP_WHILE_DRAFTING", "Chips are not applicable when building a squad from scratch.")

    n_transfers = len(transfers)
    free = n_transfers if chip in ("wildcard", "free_hit") or team.drafting else team.free_transfers
    hit = HIT_COST * max(0, n_transfers - free)
    r.computed["points_hit"] = hit
    if rec.get("points_hit") != hit:
        r.error(
            "HIT_MISCOUNT",
            f"{n_transfers} transfer(s) with {free} free means a -{hit} hit, but the recommendation says -{rec.get('points_hit')}.",
        )

    # --- availability -----------------------------------------------------
    for i in xi:
        a = t.at[i, "avail"]
        if a <= 0.25:
            r.error("UNAVAILABLE_STARTER", f"{name(i)} is unlikely to play ({a:.0%}): {t.at[i, 'news'] or t.at[i, 'status']}")
        elif a < 1:
            r.warn("DOUBTFUL_STARTER", f"{name(i)} is a doubt ({a:.0%}): {t.at[i, 'news']}")
        if t.at[i, "xp_next"] == 0 and a > 0.25:
            r.warn("BLANK_STARTER", f"{name(i)} has no fixture in GW{model.next_gw}.")

    # --- grounding: numbers must come from the model -----------------------
    cap_mult = 3 if chip == "triple_captain" else 2
    xi_xp = sum(t.at[i, "xp_next"] for i in xi)
    if cap in t.index:
        xi_xp += (cap_mult - 1) * t.at[cap, "xp_next"]
    if chip == "bench_boost":
        xi_xp += sum(t.at[i, "xp_next"] for i in bench)
    xi_xp = round(float(xi_xp), 2)
    r.computed["expected_points_next_gw"] = xi_xp
    claimed = rec.get("expected_points_next_gw")
    if claimed is None or not _close(claimed, xi_xp - hit):
        r.error(
            "XP_TOTAL_MISMATCH",
            f"Claimed GW{model.next_gw} total {claimed} but the model gives {xi_xp - hit:.2f} "
            f"({xi_xp:.2f} from the lineup minus the {hit}-point hit).",
        )

    for claim in rec.get("key_claims", []):
        i, metric, val = claim["player_id"], claim["metric"], claim["value"]
        actual = float(t.at[i, metric])
        ok = abs(val - actual) < 0.05 if metric == "price" else _close(val, actual)
        if not ok:
            r.error("CLAIM_MISMATCH", f"Claimed {metric}={val} for {name(i)}, model has {actual:.2f}.")

    return r
