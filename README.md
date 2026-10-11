# Gaffer ⚽🤖

**An AI Fantasy Premier League manager that isn't allowed to make things up.**

Gaffer is a Claude agent that manages your FPL team. It pulls live data from the Fantasy Premier League API and projects every player's points with a probabilistic model built from scratch. It searches transfers with an integer-programming solver. Before any recommendation reaches you, deterministic guardrails check it against the FPL rules and against the model's own numbers.

An illustrative run:

```
$ gaffer advise --team 1234567 "Is it time to sell Palmer?"

  🔧 get_team_overview()
  🔧 get_player(player_id=154)
  🔧 fixture_outlook(rank_by=attack)
  🔧 optimise_transfers(max_transfers=1)
  🛡  guardrails REJECTED
     XP_TOTAL_MISMATCH: Claimed GW7 total 61.4 but the model gives 57.12 ...
  🛡  guardrails PASSED
  OUT Palmer  →  IN Mbeumo (£7.9m)
```

LLMs are good at reasoning about trade-offs ("is a −4 hit worth it for a double gameweek?") and bad at arithmetic, rules, and remembering this week's injuries. So Gaffer splits the work:

| Job | Who does it |
|---|---|
| Live player, fixture, and injury data | FPL API (tool) |
| How many points will each player score? | Poisson / Bayesian xP model (tool) |
| Best legal squad under a £100m budget | Mixed-integer program, solved by HiGHS (tool) |
| Weighing risk, explaining the plan, judgement calls | Claude |
| Is the plan legal, and are its numbers true? | Deterministic guardrails |
| Audit trail of every run, tool call, and verdict | SQLite |

## Architecture

```mermaid
flowchart LR
    U[Manager] -->|question + team id| A[Gaffer agent<br/>Claude Opus 5.5]
    A -->|function calls| T{Tools}
    T --> F[FPL API client<br/>cached]
    T --> M[xP model<br/>Poisson team strength<br/>+ shrunk player rates]
    T --> O[ILP optimiser<br/>PuLP + HiGHS]
    A -->|submit_recommendation| G[Guardrails]
    G -->|violations as tool errors| A
    G -->|pass| DB[(SQLite audit log)]
    DB --> R[Recommendation]
    subgraph Cloud Run
      A
      T
      G
      DB
    end
```

The agent loop ([`agent.py`](src/gaffer/agent.py)) is written by hand instead of using an SDK tool runner, because the guardrails are part of the control flow:

1. Claude explores with read-only tools (`get_team_overview`, `search_players`, `get_player`, `fixture_outlook`, `optimise_transfers`).
2. It must finish by calling `submit_recommendation` (a strict JSON schema). A prose answer alone can't end a run.
3. [`guardrails.py`](src/gaffer/guardrails.py) validates the submission. Any violation goes back to Claude as a tool error with a machine-readable code, and Claude fixes it and resubmits (at most 4 attempts).
4. Every request, tool call (with latency), and guardrail verdict is written to SQLite ([`store.py`](src/gaffer/store.py)). `gaffer stats` aggregates pass rates and the most common violations.

## The guardrails

| Code | Catches |
|---|---|
| `UNKNOWN_PLAYER` | Hallucinated player ids |
| `SQUAD_SHAPE`, `CLUB_LIMIT`, `FORMATION`, `BENCH_ORDER` | Illegal squads and line-ups (2/5/5/3, max 3 per club, valid XI, reserve keeper first on the bench) |
| `OVER_BUDGET` | Plans that cost more than bank + sell value |
| `TRANSFER_OUT_NOT_OWNED`, `POSITION_SWAP`, `LINEUP_MISMATCH` | Transfers that don't line up with the actual squad |
| `HIT_MISCOUNT` | Quoting the wrong points hit for the number of transfers |
| `XP_TOTAL_MISMATCH`, `CLAIM_MISMATCH` | **Invented numbers**: every xP or price quoted in the advice must match the model within tolerance |
| `UNAVAILABLE_STARTER` | Starting an injured or suspended player (doubts are only warnings) |
| `CHIP_UNAVAILABLE` | Playing a chip already used in this half of the season |
| `CAPTAIN_NOT_IN_XI`, `CAPTAIN_EQUALS_VICE` | Captaincy mistakes |

The grounding checks matter most. Claude has to list every number it cites in `key_claims`, and each one is checked against the model. If it can't back a claim with the tools, it can't make it.

## The model

All of it lives in [`strength.py`](src/gaffer/strength.py) and [`xp.py`](src/gaffer/xp.py).

**Team strength.** A Maher / Dixon–Coles style Poisson model fitted by penalised maximum likelihood on this season's results:

$$\text{goals}_{home} \sim \text{Poisson}\big(e^{\mu + h + \alpha_{home} - \delta_{away}}\big), \qquad \text{goals}_{away} \sim \text{Poisson}\big(e^{\mu + \alpha_{away} - \delta_{home}}\big)$$

An L2 penalty on attack α and defence δ pulls every team toward league average. Early in the season one 5–0 shouldn't make a team elite. The analytic gradient goes to L-BFGS, so a fit takes milliseconds. (The FPL API's own team-strength fields are all zero this season, so the model has to estimate them.)

**Player expected points.** Each FPL scoring event is modelled as a probability, not a heuristic:

- **Minutes:** start rate and minutes per team game, multiplied by the availability flag from injury news.
- **Goals and assists:** xG/90 and xA/90, **Bayesian-shrunk** toward the positional mean (a prior worth 270 minutes, so a 90-minute cameo can't produce a superstar). They're then scaled by λ_for for this fixture ÷ the team's average λ_for in matches already played. A player's per-90 rates were earned against specific opponents, and this ratio adjusts for that.
- **Clean sheets:** P(60+ minutes) · e^(−λ_against).
- **Goals conceded:** E[⌊G/2⌋] under Poisson, computed exactly from the survival function. The same function handles save points (E[⌊S/3⌋]).
- **Defensive contribution:** P(Poisson(DC/90 · minutes) ≥ 10 or 12), following the 2025/26 rule.
- **Points per event:** read from the API's live `game_config.scoring`, so a rule change needs no code change.
- **Blank and double gameweeks:** handled naturally, because projections are per fixture and then summed.

**Optimiser.** [`optimizer.py`](src/gaffer/optimizer.py) uses binary variables for squad, XI, and captain. It maximises horizon xP for starters, plus a small weight on the bench and the captain's next-gameweek xP, minus 4 × hits. Constraints cover budget (sell prices), 2/5/5/3, 3 per club, and formation. It solves the full 667-player problem in about 0.1 seconds.

## Does the model work? A two-season walk-forward backtest

`gaffer backtest` refits *everything* for each gameweek *t* using only data from before *t*: team ratings, player rates, and team baselines. It then predicts *t* and scores the predictions against what actually happened. The baselines are the stats managers usually use: **form** (points per game over the last 4 gameweeks) and **ppg** (season points per game). Each predictor also feeds the same optimiser, and the resulting XI is scored on its actual points, which measures decision quality and not just error.

Two complete seasons come from the open [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) dataset, with 370–560 players scored per gameweek. Each player's club is taken as of each gameweek, so January transfers are handled. **The model's two decay settings were tuned on 2024/25 only. 2025/26 was never looked at during development, so it's an honest holdout:**

| 2025/26 holdout (GW4–38) | Gaffer model | Form | Model vs form |
|---|---|---|---|
| Mean absolute error per player ↓ | **1.52** | 1.70 | **−10.5%**, better in **35/35** gameweeks |
| Spearman rank correlation ↑ | **0.609** | 0.546 | **+11.6%**, better in **35/35** gameweeks |
| Avg actual points of top-20 picks ↑ | **4.43** | 3.82 | +15.8%, better in 24/35 |
| Actual points of the optimised XI ↑ | **57.6** | 48.4 | **+19.1%**, better in 26/35 |

The tuning season (2024/25) shows the same pattern: −8.5% MAE (35/35 weeks), +8.6% rank correlation (33/35), and +22.7% XI points (28/35). Full per-gameweek results are in [`reports/`](reports/).

**What the tuning found.** The original model weighted every match equally and lost to form on ranking. Adding an exponential time decay fixed that, but the two parts of the model want different settings. *Who plays* changes week to week (half-life of 1 gameweek), while *how good a player is per 90 minutes* is best estimated from the whole season (no decay). The grid search is reproducible from `features_from_history` in [`xp.py`](src/gaffer/xp.py).

**A data leakage catch.** The archive also includes FPL's own expected points (`xP`), which at first look unbeatable: rank correlation 0.74, and its XI averaging 97 points. That's too good. Among players who started the previous gameweek, FPL's archived xP averages **0.93 for those who then didn't play** and **3.28 for those who did** ([`fpl_xp_leakage`](src/gaffer/history.py)). The values were evidently captured after team news, so the comparison would be unfair, and it's excluded.

> **Caveats:** the backtest assumes every player is available, because historical injury news isn't in the archive. That understates the live model, which does use the injury flags. The optimised-XI metric is noisy week to week (26/35 wins), so treat the error and ranking metrics as the headline.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd gaffer
uv sync
export ANTHROPIC_API_KEY=sk-ant-...

# No API key needed for these:
uv run gaffer top --pos MID -n 10        # best projected midfielders
uv run gaffer ratings                    # team attack/defence ratings
uv run gaffer optimise --team 1234567    # pure-maths transfer plan
uv run gaffer backtest --season 2025-26  # walk-forward evaluation (or 2024-25, or live)
uv run --extra app streamlit run streamlit_app.py   # the dashboard

# The agent, with Claude:
uv run gaffer advise --team 1234567                    # your team (id is in the URL of your Points page)
uv run gaffer advise "Build me a wildcard squad"       # no team: drafts from scratch with £100m
uv run gaffer advise --team 1234567 --ft 2 "Should I take a hit for Haaland?"

# The agent, free and offline, on a local model:
brew install ollama && ollama pull qwen3:4b
OLLAMA_CONTEXT_LENGTH=16384 ollama serve &
uv run gaffer advise --local --team 1234567

uv run gaffer eval --local -n 10         # run on 10 real managers' teams, report guardrail stats
uv run gaffer stats                      # pass rates and violations caught, from the database
```

Configuration: `GAFFER_PROVIDER` (`anthropic` or `ollama`), `GAFFER_MODEL` (default `claude-opus-5-5`, or `qwen3:4b` locally), `GAFFER_EFFORT` (default `medium`), `GAFFER_DB` (SQLite path).

### Two backends, one loop

The agent loop, tools, and guardrails don't depend on which model drives them. `--local` points the official Anthropic SDK at [Ollama](https://ollama.com)'s Anthropic-compatible endpoint, so a free 4-billion-parameter model on a laptop runs the same code path as Claude. Small models make far more mistakes, which makes them a good stress test for the guardrails (results below).

## Dashboard

[`streamlit_app.py`](streamlit_app.py) is an interactive front end that needs no API key. It has five tabs: the backtest results with weekly charts, live projections with filters, a fixture-difficulty heatmap from the team-strength model, the squad optimiser (enter any FPL team id), and a trace of a real agent run. To host it free, deploy the repo on [Streamlit Community Cloud](https://streamlit.io/cloud) with `streamlit_app.py` as the entry point (it installs from `requirements.txt`).

## HTTP API and deployment (Google Cloud Run)

```bash
uv run gaffer serve   # http://localhost:8080/docs
```

| Endpoint | |
|---|---|
| `POST /advise` | `{"team_id": 1234567, "question": "..."}` → recommendation, guardrail report, token usage |
| `GET /players/top?position=FWD` | Model projections |
| `GET /runs/{id}` | Full audit trail of a run: tool calls, latencies, every guardrail attempt |
| `GET /stats` | Aggregate guardrail metrics |

[`deploy/cloudrun.sh`](deploy/cloudrun.sh) builds the [`Dockerfile`](Dockerfile) with Cloud Build and deploys it with the API key mounted from Secret Manager. The service requires authentication, since every call spends API credits. Setup steps are in the script header.

## Engineering notes

- **Tests (30, offline, under 15 seconds):** they run on a synthetic 20-club league. Coverage includes recovering known team strengths from simulated seasons, checking the closed-form Poisson expectations against Monte Carlo, blank and double gameweek handling, time-decayed features, skipping in-progress fixtures, optimiser legality, every guardrail code, both model backends, and the agent loop against a **scripted fake Claude** that submits a hallucinated total, gets bounced, and corrects itself. The fake checks the request is append-only, which keeps prompt caching and thinking-block replay valid.
- **CI:** ruff lint and format, plus pytest, on every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- **Cost control:** prompt caching on the static system prompt and tools, adaptive thinking with configurable effort, and token usage logged per run.
- **Resilience:** FPL responses are cached on disk with TTLs. Tool exceptions go back to the model as `is_error` results instead of crashing the run. On a safety decline, the server-side model fallback takes over.

## Known limitations

- The public API doesn't expose purchase prices, so sell prices are approximated by current prices. Free transfers are estimated from transfer history and can be overridden with `--ft`.
- Minutes modelling is the weakest part of the model: it doesn't know about rotation risk or new signings.
- The optimiser treats the horizon as one block (no week-by-week transfer planning) and ignores price changes.
- On Cloud Run the SQLite file lives on ephemeral disk. For durable history, mount a volume or point `GAFFER_DB` at persistent storage.

## Acknowledgements

Developed with the help of Claude (Anthropic) as an AI pair programmer. Data comes from the public Fantasy Premier League API. This project isn't affiliated with the Premier League.
