"""Squad / transfer optimisation as a mixed-integer linear program (PuLP + HiGHS).

Decision variables per player: in squad, in starting XI, captain. The objective
is horizon xP for starters, a small weight on bench xP (auto-sub insurance),
the captain's next-gameweek xP counted twice, minus 4 points per transfer
beyond the free ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from .fpl import SQUAD_SHAPE

BENCH_WEIGHT = 0.15
HIT_COST = 4
MIN_XI = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_XI = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}


@dataclass
class Plan:
    squad: list[int]
    starting_xi: list[int]
    bench: list[int]
    captain: int
    vice_captain: int
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)
    hits: int = 0
    objective: float = 0.0
    xp_next_xi: float = 0.0
    bank_after: float = 0.0
    status: str = "Optimal"

    def transfer_pairs(self, table: pd.DataFrame) -> list[dict]:
        """Pair outs with ins like-for-like by position, as FPL requires."""
        ins = list(self.transfers_in)
        pairs = []
        for o in self.transfers_out:
            i = next(i for i in ins if table.at[i, "pos"] == table.at[o, "pos"])
            ins.remove(i)
            pairs.append({"out": o, "in": i})
        return pairs


def optimise(
    table: pd.DataFrame,
    *,
    current_squad: list[int] | None = None,
    bank: float = 100.0,
    sell_prices: dict[int, float] | None = None,
    free_transfers: int = 1,
    max_transfers: int | None = None,
    locked: list[int] = (),
    banned: list[int] = (),
    team_limit: int = 3,
    min_avail: float = 0.5,
) -> Plan:
    """`table` needs columns pos, team, price, xp_next, xp_horizon, avail.

    With no current squad this builds a squad from scratch with `bank` as the
    full budget (unlimited free transfers).
    """
    current = set(current_squad or [])
    sell = sell_prices or {}
    drafting = not current

    pool = table[(table.avail >= min_avail) | table.index.isin(current)]
    pool = pool[~pool.index.isin(banned) | pool.index.isin(current)]
    ids = list(pool.index)

    prob = pulp.LpProblem("gaffer", pulp.LpMaximize)
    x = prob.add_variable_dicts("squad", ids, cat="Binary")
    s = prob.add_variable_dicts("start", ids, cat="Binary")
    c = prob.add_variable_dicts("cap", ids, cat="Binary")
    hits = prob.add_variable("hits", lowBound=0, cat="Integer")

    xp_h = pool.xp_horizon.to_dict()
    xp_1 = pool.xp_next.to_dict()
    prob += pulp.lpSum(xp_h[i] * s[i] + BENCH_WEIGHT * xp_h[i] * (x[i] - s[i]) + xp_1[i] * c[i] for i in ids) - HIT_COST * hits

    prob += pulp.lpSum(x.values()) == 15
    prob += pulp.lpSum(s.values()) == 11
    prob += pulp.lpSum(c.values()) == 1
    for i in ids:
        prob += s[i] <= x[i]
        prob += c[i] <= s[i]
    for pos, n in SQUAD_SHAPE.items():
        members = [i for i in ids if pool.at[i, "pos"] == pos]
        prob += pulp.lpSum(x[i] for i in members) == n
        prob += pulp.lpSum(s[i] for i in members) >= MIN_XI[pos]
        prob += pulp.lpSum(s[i] for i in members) <= MAX_XI[pos]
    for members in pool.groupby("team").groups.values():
        prob += pulp.lpSum(x[i] for i in members) <= team_limit

    def cost(i):
        return sell.get(i, pool.at[i, "price"]) if i in current else pool.at[i, "price"]

    budget = bank + sum(sell.get(i, table.at[i, "price"]) for i in current)
    prob += pulp.lpSum(cost(i) * x[i] for i in ids) <= budget + 1e-6

    if not drafting:
        n_in = pulp.lpSum(x[i] for i in ids if i not in current)
        prob += hits >= n_in - free_transfers
        if max_transfers is not None:
            prob += n_in <= max_transfers
    else:
        prob += hits == 0
    for i in locked:
        if i in x:
            prob += x[i] == 1

    prob.solve(pulp.HiGHS(msg=False, timeLimit=30))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return Plan([], [], [], -1, -1, status=status)

    squad = [i for i in ids if x[i].value() > 0.5]
    xi = [i for i in squad if s[i].value() > 0.5]
    cap = next(i for i in xi if c[i].value() > 0.5)
    vice = max((i for i in xi if i != cap), key=lambda i: xp_1[i])
    bench_out = [i for i in squad if i not in xi]
    bench_gk = [i for i in bench_out if pool.at[i, "pos"] == "GKP"]
    bench_rest = sorted((i for i in bench_out if i not in bench_gk), key=lambda i: -xp_1[i])
    spent = sum(cost(i) for i in squad)

    return Plan(
        squad=squad,
        starting_xi=xi,
        bench=bench_gk + bench_rest,
        captain=cap,
        vice_captain=vice,
        transfers_in=[i for i in squad if i not in current] if not drafting else [],
        transfers_out=[i for i in current if i not in squad],
        hits=round(hits.value() or 0),
        objective=float(pulp.value(prob.objective)),
        xp_next_xi=float(sum(xp_1[i] for i in xi) + xp_1[cap]),
        bank_after=round(budget - spent, 1),
        status=status,
    )
