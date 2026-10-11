"""Gaffer dashboard: the model, optimiser and backtest, no API key needed.

uv run --extra app streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gaffer.backtest import load_history
from gaffer.fpl import FPLClient, build_snapshot
from gaffer.team import TeamState, load_team
from gaffer.tools import Toolbox
from gaffer.xp import fit_model, team_fixtures

REPORTS = Path(__file__).parent / "reports"
# Validated categorical slots 1-2 and the blue sequential ramp (see the dataviz palette).
MODEL_BLUE, BASELINE_ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

st.set_page_config(page_title="Gaffer", page_icon="⚽", layout="wide")


@st.cache_resource(ttl=3600, show_spinner="Fetching live FPL data and fitting the model...")
def get_model():
    client = FPLClient(cache_dir="/tmp/gaffer-cache")
    snap = build_snapshot(client)
    try:
        hist = load_history(client, snap)
    except Exception:  # noqa: BLE001 - fall back to season totals if per-player history is unavailable
        hist = None
    return client, fit_model(snap, hist=hist)


def chart_layout(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
        **kw,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


st.title("⚽ Gaffer")
st.caption(
    "An AI Fantasy Premier League manager. A Poisson expected-points model, an integer-programming optimiser, "
    "and guardrails that check every recommendation the agent makes. [Code on GitHub](https://github.com/clee7010/gaffer)"
)

tab_bt, tab_players, tab_fix, tab_opt, tab_agent = st.tabs(
    ["📈 Does it work?", "🔮 Projections", "🗓️ Fixtures", "🧮 Optimiser", "🤖 The agent"]
)

# --- Backtest -----------------------------------------------------------------
with tab_bt:
    st.subheader("Walk-forward backtest")
    st.write(
        "For every gameweek, the model is refit using only data from before that week, then scored against what "
        "actually happened. The comparison is FPL's **form** stat. The decay settings were tuned on 2024/25 only; "
        "**2025/26 is an untouched holdout.**"
    )
    season = st.radio("Season", ["2025-26", "2024-25"], horizontal=True, captions=["holdout", "tuning"])
    by_gw = pd.read_csv(REPORTS / season / "backtest_by_gw.csv")
    h2h = json.loads((REPORTS / season / "head_to_head.json").read_text())["form"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Prediction error vs form",
        f"−{h2h['mae']['improvement_pct']:.1f}%",
        f"better in {h2h['mae']['weeks_won']}/{h2h['mae']['weeks']} gameweeks",
        delta_color="off",
    )
    c2.metric(
        "Rank correlation vs form",
        f"+{h2h['spearman']['improvement_pct']:.1f}%",
        f"better in {h2h['spearman']['weeks_won']}/{h2h['spearman']['weeks']} gameweeks",
        delta_color="off",
    )
    c3.metric(
        "Actual points of the picked XI",
        f"+{h2h['optimal_xi_actual_pts']['improvement_pct']:.1f}%",
        f"better in {h2h['optimal_xi_actual_pts']['weeks_won']}/{h2h['optimal_xi_actual_pts']['weeks']} gameweeks",
        delta_color="off",
    )

    def weekly(metric: str, title: str, fmt: str):
        wide = by_gw.pivot(index="gw", columns="predictor", values=metric)
        fig = go.Figure()
        for name, color, label in [("model", MODEL_BLUE, "Gaffer model"), ("form", BASELINE_ORANGE, "Form")]:
            fig.add_scatter(
                x=wide.index,
                y=wide[name],
                name=label,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate=f"%{{y:{fmt}}}",
            )
        chart_layout(fig, height=300, xaxis_title="Gameweek", yaxis_title=title)
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Mean absolute error per player** (lower is better)")
        weekly("mae", "MAE (points)", ".2f")
    with right:
        st.markdown("**Actual points scored by each predictor's optimised XI**")
        weekly("optimal_xi_actual_pts", "Points", ".0f")
    with st.expander("Data leakage found in FPL's own archived xP"):
        leak = json.loads((REPORTS / season / "fpl_xp_leakage.json").read_text())
        st.write(
            "The public dataset also archives FPL's own expected points. It looks unbeatable, but among players who "
            f"started the previous week it averages **{leak['xp_when_previous_starter_then_did_not_play']}** for "
            f"those who then didn't play vs **{leak['xp_when_previous_starter_then_played_60']}** for those who "
            "did, which means it was recorded after team news. It's excluded as a baseline."
        )
    with st.expander("Weekly results table"):
        st.dataframe(by_gw.round(3), width="stretch", hide_index=True)

# --- Live model -----------------------------------------------------------------
try:
    client, model = get_model()
except Exception as e:  # noqa: BLE001
    client = model = None
    live_error = str(e)

if model is None:
    for tab in (tab_players, tab_fix, tab_opt):
        with tab:
            st.warning(f"Couldn't reach the FPL API right now ({live_error}). The backtest tab still works.")
else:
    t = model.table
    with tab_players:
        st.subheader(f"Expected points, GW{model.next_gw}–{model.gws[-1]}")
        f1, f2, f3 = st.columns([1, 1, 2])
        pos = f1.selectbox("Position", ["All", "GKP", "DEF", "MID", "FWD"])
        max_price = f2.slider("Max price (£m)", 4.0, 16.0, 16.0, 0.5)
        name = f3.text_input("Search name")
        df = t[t.price <= max_price]
        if pos != "All":
            df = df[df.pos == pos]
        if name:
            df = df[df.name.str.contains(name, case=False, regex=False)]
        show = df.sort_values("xp_next", ascending=False).head(60)
        st.dataframe(
            show[["name", "team_short", "pos", "price", "avail", "xp_next", "xp_horizon", "value", "news"]].rename(
                columns={
                    "team_short": "club",
                    "xp_next": f"xP GW{model.next_gw}",
                    "xp_horizon": "xP horizon",
                    "value": "xP per £m",
                    "avail": "available",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "price": st.column_config.NumberColumn(format="£%.1fm"),
                "available": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                f"xP GW{model.next_gw}": st.column_config.NumberColumn(format="%.2f"),
                "xP horizon": st.column_config.NumberColumn(format="%.1f"),
                "xP per £m": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with tab_fix:
        st.subheader("Fixture ticker")
        metric = st.radio("Show", ["Expected goals for", "Clean-sheet probability"], horizontal=True)
        tf = team_fixtures(model.snap.fixtures, model.gws)
        short = model.snap.teams.short
        rows = []
        for team, opp, gw, home in zip(tf.team, tf.opp, tf.gw, tf.home):
            lf, la = model.ratings.expected_goals(team, opp) if home else model.ratings.expected_goals(opp, team)[::-1]
            rows.append(
                {
                    "club": short[team],
                    "gw": int(gw),
                    "label": f"{short[opp]} ({'H' if home else 'A'})",
                    "value": lf if metric.startswith("Expected") else math.exp(-la),
                }
            )
        fx = pd.DataFrame(rows).groupby(["club", "gw"]).agg(value=("value", "sum"), label=("label", " + ".join)).reset_index()
        z = fx.pivot(index="club", columns="gw", values="value")
        labels = fx.pivot(index="club", columns="gw", values="label").reindex_like(z).fillna("blank")
        order = z.sum(axis=1).sort_values().index
        z, labels = z.loc[order], labels.loc[order]
        fig = go.Figure(
            go.Heatmap(
                z=z.values,
                x=[f"GW{g}" for g in z.columns],
                y=z.index,
                text=labels.values,
                texttemplate="%{text}",
                colorscale=[[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
                xgap=2,
                ygap=2,
                hovertemplate="%{y} vs %{text}<br>%{z:.2f}<extra></extra>",
                colorbar={"title": {"text": "xG for" if metric.startswith("Expected") else "P(CS)", "side": "right"}},
            )
        )
        chart_layout(fig, height=640)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, width="stretch")
        st.caption("Easiest runs at the top. Values come from the Poisson team-strength model, not FPL's difficulty ratings.")

    with tab_opt:
        st.subheader("Squad optimiser")
        st.write(
            "Solves the exact integer program: £100m budget, 2/5/5/3, max 3 per club, valid formation, −4 per extra transfer."
        )
        o1, o2, o3 = st.columns([2, 1, 1])
        team_id = o1.text_input("Your FPL team id (optional; leave blank to build from scratch)")
        ft = o2.number_input("Free transfers", 0, 5, 1)
        max_tr = o3.number_input("Max transfers", 0, 15, 3)
        if st.button("Optimise", type="primary"):
            try:
                team = (
                    load_team(client, model.snap, int(team_id), free_transfers=ft) if team_id.strip() else TeamState(team_id=None)
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't load team {team_id}: {e}")
                st.stop()
            plan = Toolbox(model, team).tool_optimise_transfers(max_transfers=None if team.drafting else int(max_tr))
            if plan["status"] != "Optimal":
                st.error(plan.get("message", plan["status"]))
            else:
                ids = plan["ids"]
                if not team.drafting:
                    st.markdown(
                        "**Transfers:** "
                        + (
                            ", ".join(f"{t.at[tr['out'], 'name']} → **{t.at[tr['in'], 'name']}**" for tr in ids["transfers"])
                            or "roll the transfer"
                        )
                    )
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    f"Projected GW{model.next_gw} points", f"{plan['xp_next_gw_lineup_incl_captain'] - plan['points_hit']:.1f}"
                )
                m2.metric("Points hit", f"−{plan['points_hit']}")
                m3.metric("Bank after", f"£{plan['bank_after']:.1f}m")
                xi = t.loc[ids["starting_xi"]].copy()
                xi["role"] = ["(C)" if i == ids["captain"] else "(VC)" if i == ids["vice_captain"] else "" for i in xi.index]
                xi = xi.sort_values("pos", key=lambda s: s.map({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}))
                st.dataframe(
                    xi[["name", "role", "team_short", "pos", "price", "xp_next", "xp_horizon"]], hide_index=True, width="stretch"
                )
                st.caption("Bench: " + ", ".join(t.at[i, "name"] for i in ids["bench"]))

# --- Agent ----------------------------------------------------------------------
with tab_agent:
    st.subheader("How the agent works")
    st.write(
        "A language model (Claude, or a free local model through Ollama) plans the advice using six tools. It can only "
        "finish by calling `submit_recommendation`, which **deterministic guardrails** check for FPL legality, budget, "
        "the points hit, injured starters, and whether every number it quotes matches the model. Any violation is sent "
        "back as an error to fix. The agent runs locally (`gaffer advise`); this hosted demo has no LLM attached."
    )
    sample = REPORTS / "agent" / "sample_run.json"
    stats = REPORTS / "agent" / "eval_stats.json"
    if stats.exists():
        st.markdown("**Guardrail results from real runs** (`gaffer eval`)")
        st.json(json.loads(stats.read_text()))
    if sample.exists():
        run = json.loads(sample.read_text())
        st.markdown(f"**A real run:** *{run['question']}*")
        for step in run["steps"]:
            icon = "🛡️" if step["name"] == "submit_recommendation" else "🔧"
            verdict = "" if step["name"] != "submit_recommendation" else (" ❌ rejected" if step["is_error"] else " ✅ passed")
            with st.expander(f"{icon} {step['name']}{verdict}"):
                st.code(json.dumps(step["detail"], indent=2)[:3000], language="json")
        st.info(run["answer"])
