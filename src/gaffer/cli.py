"""`gaffer` command line."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .fpl import FPLClient, build_snapshot
from .team import TeamState, load_team
from .xp import fit_model

console = Console()


def _model(horizon: int):
    client = FPLClient()
    from .backtest import load_history

    with console.status("Fetching FPL data and fitting the model (first run fetches every player's history)..."):
        snap = build_snapshot(client)
        model = fit_model(snap, horizon=horizon, hist=load_history(client, snap))
    return client, snap, model


def _team(client, snap, team_id, ft) -> TeamState:
    if team_id is None:
        return TeamState(team_id=None)
    return load_team(client, snap, team_id, free_transfers=ft)


def _player_table(df, title: str) -> Table:
    tbl = Table(title=title)
    for col in ["name", "club", "pos", "£m", "avail", "xP next", "xP horizon"]:
        tbl.add_column(col, justify="right" if col not in ("name", "club", "pos") else "left")
    for _, r in df.iterrows():
        tbl.add_row(r["name"], r.team_short, r.pos, f"{r.price:.1f}", f"{r.avail:.0%}", f"{r.xp_next:.2f}", f"{r.xp_horizon:.1f}")
    return tbl


def cmd_top(a):
    _, _, model = _model(a.horizon)
    df = model.table
    if a.pos:
        df = df[df.pos == a.pos.upper()]
    if a.max_price:
        df = df[df.price <= a.max_price]
    console.print(_player_table(df.sort_values(a.sort, ascending=False).head(a.n), f"Top players, GW{model.next_gw}"))


def cmd_ratings(a):
    _, snap, model = _model(a.horizon)
    tbl = Table(title="Team strength (log-scale, Poisson MLE with ridge shrinkage)")
    for c in ["club", "attack", "defence", "net"]:
        tbl.add_column(c)
    for _, r in model.ratings.table(snap.teams).iterrows():
        tbl.add_row(r["short"], f"{r.att:+.2f}", f"{r['def']:+.2f}", f"{r.net:+.2f}")
    console.print(tbl)
    console.print(f"home advantage: x{2.718281828**model.ratings.home:.2f} goals")


def cmd_optimise(a):
    from .tools import Toolbox

    client, snap, model = _model(a.horizon)
    team = _team(client, snap, a.team, a.ft)
    out = Toolbox(model, team).tool_optimise_transfers(max_transfers=a.max_transfers, wildcard=a.wildcard)
    out.pop("ids", None)
    console.print_json(json.dumps(out))


def cmd_advise(a):
    from .agent import Gaffer
    from .tools import Toolbox

    client, snap, model = _model(a.horizon)
    team = _team(client, snap, a.team, a.ft)
    question = a.question or (
        "Build me the best squad for the next few gameweeks."
        if team.drafting
        else "What transfers should I make this week, and who should I captain?"
    )

    def on_tool(name, args, out, is_error):
        if name == "submit_recommendation":
            verdict = "[green]PASSED[/]" if not is_error else "[red]REJECTED[/]"
            console.print(f"  🛡  guardrails {verdict}")
            if is_error:
                for e in out["errors"]:
                    console.print(f"     [red]{e['code']}[/]: {e['message']}")
        else:
            shown = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
            console.print(f"  🔧 {name}({shown}){' [red]error[/]' if is_error else ''}")

    console.print(Panel(question, title=f"GW{model.next_gw} · {team.team_name or 'new squad'}"))
    agent = Gaffer(provider="ollama" if a.local else "anthropic", model=a.model)
    result = agent.run(question, Toolbox(model, team), on_tool=on_tool)

    rec = result.recommendation
    if rec:
        t = model.table
        if rec["transfers"]:
            for tr in rec["transfers"]:
                console.print(
                    f"  [red]OUT[/] {t.at[tr['out'], 'name']}  →  [green]IN[/] {t.at[tr['in'], 'name']} (£{t.at[tr['in'], 'price']}m)"
                )
        xi = t.loc[rec["starting_xi"]].sort_values("pos", key=lambda s: s.map({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}))
        console.print(
            _player_table(xi, f"Starting XI  (C: {t.at[rec['captain'], 'name']}, VC: {t.at[rec['vice_captain'], 'name']})")
        )
        console.print("Bench: " + ", ".join(t.at[i, "name"] for i in rec["bench"]))
        for w in rec["guardrails"]["warnings"]:
            console.print(f"  [yellow]⚠ {w['message']}[/]")
    console.print(Panel(result.answer, title=f"The Gaffer says ({result.status})"))
    u = result.usage
    console.print(
        f"[dim]run #{result.run_id} · {result.tool_calls} tool calls · {result.guardrail_attempts} guardrail attempt(s) · "
        f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens ({u['cache_read_input_tokens']:,} cached)[/]"
    )


def cmd_backtest(a):
    from .backtest import head_to_head, run_backtest, summarise

    if a.season == "live":
        client = FPLClient()
        snap, hist = build_snapshot(client), None
    else:
        from .history import load_season

        client = None
        snap, hist = load_season(a.season)
    with console.status(f"Walk-forward backtest ({a.season})..."):
        res = run_backtest(client, snap, start_gw=a.start, hist=hist)
    out = Path(a.out) / a.season
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "backtest_by_gw.csv", index=False)
    summary = summarise(res)
    summary.round(3).to_csv(out / "backtest_summary.csv")
    h2h = {b: head_to_head(res, b) for b in summary.index if b != "model"}
    (out / "head_to_head.json").write_text(json.dumps(h2h, indent=2))
    console.print(summary.round(3).to_string())
    if hist is not None and "fpl_xp" in hist:
        from .history import fpl_xp_leakage

        leak = fpl_xp_leakage(hist)
        (out / "fpl_xp_leakage.json").write_text(json.dumps(leak, indent=2))
        console.print(f"FPL archived xP leakage check: {leak}")
    for base, metrics in h2h.items():
        console.print(f"\n[bold]model vs {base}[/]")
        for m, v in metrics.items():
            console.print(f"  {m:24s} {v['improvement_pct']:+6.1f}%   better in {v['weeks_won']}/{v['weeks']} gameweeks")


def cmd_stats(a):
    from .store import Store

    console.print_json(json.dumps(Store().stats(model=a.model)))


EVAL_QUESTIONS = [
    "What transfers should I make this week, and who should I captain?",
    "Should I take a -4 hit this week or roll my transfer?",
    "Who is the best captain this week and why?",
    "My worst player needs replacing. Who should go and who comes in?",
    "Plan my team for the next few gameweeks.",
]


def cmd_eval(a):
    """Run the agent over real managers' teams from the overall league."""
    from .agent import Gaffer
    from .store import Store
    from .tools import Toolbox

    client, snap, model = _model(a.horizon)
    standings = client.get(f"/leagues-classic/314/standings/?page_standings={a.page}", ttl=86400)
    entries = [r["entry"] for r in standings["standings"]["results"]][: a.n]
    agent = Gaffer(provider="ollama" if a.local else "anthropic", model=a.model)
    for k, entry in enumerate(entries):
        question = EVAL_QUESTIONS[k % len(EVAL_QUESTIONS)]
        try:
            team = load_team(client, snap, entry)
        except Exception as e:  # noqa: BLE001 - a deleted or private team shouldn't stop the batch
            console.print(f"[yellow]skip team {entry}: {e}[/]")
            continue
        t0 = time.perf_counter()
        res = agent.run(question, Toolbox(model, team))
        dt = time.perf_counter() - t0
        console.print(
            f"[{k + 1}/{len(entries)}] team {entry}: {res.status} after {res.guardrail_attempts} submission(s), "
            f"{res.tool_calls} tool calls, {dt:.0f}s"
        )
    console.print_json(json.dumps(Store().stats(model=agent.model)))


def cmd_serve(a):
    import uvicorn

    uvicorn.run("gaffer.api:app", host="0.0.0.0", port=a.port)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="gaffer", description="An AI Fantasy Premier League manager.")
    p.add_argument("--horizon", type=int, default=5, help="Gameweeks to project (default 5)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("advise", help="Ask the agent for advice")
    s.add_argument("question", nargs="?")
    s.add_argument("--team", type=int, help="Your FPL team id (from the URL of your points page)")
    s.add_argument("--ft", type=int, help="Override free transfers")
    s.add_argument("--local", action="store_true", help="Use a free local model through Ollama instead of Claude")
    s.add_argument("--model", help="Model name (default claude-opus-5-5, or qwen3:4b with --local)")
    s.set_defaults(fn=cmd_advise)

    s = sub.add_parser("top", help="Top projected players")
    s.add_argument("--pos", choices=["GKP", "DEF", "MID", "FWD", "gkp", "def", "mid", "fwd"])
    s.add_argument("--max-price", type=float)
    s.add_argument("--sort", default="xp_next", choices=["xp_next", "xp_horizon", "value"])
    s.add_argument("-n", type=int, default=15)
    s.set_defaults(fn=cmd_top)

    sub.add_parser("ratings", help="Team strength ratings").set_defaults(fn=cmd_ratings)

    s = sub.add_parser("optimise", help="Run the optimiser without the agent")
    s.add_argument("--team", type=int)
    s.add_argument("--ft", type=int)
    s.add_argument("--max-transfers", type=int)
    s.add_argument("--wildcard", action="store_true")
    s.set_defaults(fn=cmd_optimise)

    s = sub.add_parser("backtest", help="Walk-forward backtest of the xP model")
    s.add_argument("--season", default="live", help="live (current season) or a past season, e.g. 2024-25")
    s.add_argument("--start", type=int, default=4)
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("stats", help="Guardrail and run statistics from the database")
    s.add_argument("--model", help="Only runs from this model")
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("eval", help="Run the agent on real teams from the overall league and report guardrail stats")
    s.add_argument("-n", type=int, default=10, help="Number of teams")
    s.add_argument("--page", type=int, default=20, help="Overall-league standings page to sample (50 teams per page)")
    s.add_argument("--local", action="store_true", help="Use a free local model through Ollama")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("serve", help="Run the HTTP API")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
