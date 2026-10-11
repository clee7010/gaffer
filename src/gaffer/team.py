"""A manager's current FPL team, reconstructed from public endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

from .fpl import FPLClient, Snapshot

MAX_FREE_TRANSFERS = 5


@dataclass
class TeamState:
    team_id: int | None
    squad: list[int] = field(default_factory=list)
    bank: float = 100.0
    sell_prices: dict[int, float] = field(default_factory=dict)
    free_transfers: int = 1
    manager: str | None = None
    team_name: str | None = None
    overall_rank: int | None = None
    chips_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def drafting(self) -> bool:
        return not self.squad


def estimate_free_transfers(history: dict, started_event: int, upto_gw: int) -> int:
    """Roll free transfers forward: +1 per deadline (cap 5), minus transfers made.

    Wildcard / Free Hit weeks keep the banked count. The public API does not
    expose this number directly, so callers can override it.
    """
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    used = {row["event"]: row["event_transfers"] for row in history.get("current", [])}
    ft = 1
    for gw in range(started_event + 1, upto_gw):  # deadlines already passed
        if chips.get(gw) in ("wildcard", "freehit"):
            continue
        ft = min(MAX_FREE_TRANSFERS, max(0, ft - used.get(gw, 0)) + 1)
    return ft


def load_team(client: FPLClient, snap: Snapshot, team_id: int, free_transfers: int | None = None) -> TeamState:
    entry = client.entry(team_id)
    history = client.entry_history(team_id)
    gw = snap.current_gw or snap.next_gw
    picks = client.entry_picks(team_id, gw)

    squad = [p["element"] for p in picks["picks"]]
    if picks.get("active_chip") == "freehit" and gw > 1:
        # A Free Hit squad reverts, so the real squad is last week's.
        squad = [p["element"] for p in client.entry_picks(team_id, gw - 1)["picks"]]

    bank = picks["entry_history"]["bank"] / 10
    notes = [
        (
            "Sell prices use current prices: the public API does not expose purchase prices, "
            "so real sell prices may be up to £0.1m-£0.3m lower on risen players."
        ),
    ]
    if free_transfers is None:
        free_transfers = estimate_free_transfers(history, entry.get("started_event", 1), snap.next_gw)
        notes.append("Free transfers are estimated from public transfer history.")

    return TeamState(
        team_id=team_id,
        squad=squad,
        bank=bank,
        sell_prices={i: float(snap.players.at[i, "price"]) for i in squad},
        free_transfers=free_transfers,
        manager=f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        team_name=entry.get("name"),
        overall_rank=entry.get("summary_overall_rank"),
        chips_used=[f"{c['name']}@GW{c['event']}" for c in history.get("chips", [])],
        notes=notes,
    )
