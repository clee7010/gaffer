"""The agent loop against a scripted fake Claude: no network, no API key."""

import json
from types import SimpleNamespace as NS

from gaffer.agent import Gaffer
from gaffer.store import Store
from gaffer.team import TeamState
from gaffer.tools import Toolbox


def usage():
    return NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=80)


def tool_call(name, args, id_):
    return NS(type="tool_use", name=name, input=args, id=id_)


def reply(blocks, stop="tool_use"):
    return NS(content=blocks, stop_reason=stop, usage=usage())


class FakeClient:
    """Plays a script; each step sees the conversation so far."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot, the loop appends
        return self.script.pop(0)(kwargs["messages"])


def last_tool_result(messages):
    return json.loads(messages[-1]["content"][-1]["content"])


def make_script(bad_first: bool):
    state = {}

    def overview(_):
        return reply([NS(type="text", text="Let me look at the squad."), tool_call("get_team_overview", {}, "t1")])

    def plan(_):
        return reply([tool_call("optimise_transfers", {}, "t2")])

    def submit(messages, wrong=False):
        if "ids" in (res := last_tool_result(messages)):
            state["plan"] = res
        ids = state["plan"]["ids"]
        xp = state["plan"]["xp_next_gw_lineup_incl_captain"] - state["plan"]["points_hit"]
        rec = {
            "transfers": ids["transfers"],
            "chip": "none",
            "starting_xi": ids["starting_xi"],
            "bench": ids["bench"],
            "captain": ids["captain"],
            "vice_captain": ids["vice_captain"],
            "points_hit": state["plan"]["points_hit"],
            "expected_points_next_gw": xp + (15 if wrong else 0),  # hallucinated total on the first try
            "key_claims": [],
            "summary": "Back the model, captain the striker.",
        }
        return reply([tool_call("submit_recommendation", rec, f"t-sub-{len(messages)}")])

    def done(_):
        return reply([NS(type="text", text="Locked in. Good luck.")], stop="end_turn")

    steps = [overview, plan]
    if bad_first:
        steps.append(lambda m: submit(m, wrong=True))
    return steps + [submit, done]


def test_agent_self_corrects_after_guardrail_rejection(model, tmp_path):
    store = Store(tmp_path / "g.db")
    fake = FakeClient(make_script(bad_first=True))
    toolbox = Toolbox(model, TeamState(team_id=None))
    result = Gaffer(client=fake, store=store).run("Build me a squad", toolbox)

    assert result.status == "accepted"
    assert result.guardrail_attempts == 2
    assert result.recommendation["guardrails"]["passed"]
    assert result.usage["input_tokens"] == 100 * len(fake.requests)

    # The rejection was sent back to the model as a tool error naming the problem.
    bounced = fake.requests[3]["messages"][-1]["content"][0]
    assert bounced["is_error"] and "XP_TOTAL_MISMATCH" in bounced["content"]

    run = store.get_run(result.run_id)
    assert run["status"] == "accepted"
    assert [g["passed"] for g in run["guardrail_events"]] == [0, 1]
    assert {t["name"] for t in run["tool_calls"]} >= {"get_team_overview", "optimise_transfers", "submit_recommendation"}
    assert store.stats()["violations_caught"] == {"XP_TOTAL_MISMATCH": 1}


def test_request_shape(model, tmp_path):
    fake = FakeClient(make_script(bad_first=False))
    Gaffer(client=fake, store=Store(tmp_path / "g.db")).run("hi", Toolbox(model, TeamState(team_id=None)))
    req = fake.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert {t["name"] for t in req["tools"]} >= {"submit_recommendation", "optimise_transfers"}
    # Messages are append-only: every request extends the previous one.
    for prev, nxt in zip(fake.requests, fake.requests[1:]):
        assert nxt["messages"][: len(prev["messages"])] == prev["messages"]


def test_agent_cannot_finish_without_passing_guardrails(model, tmp_path):
    chatty = [lambda _: reply([NS(type="text", text="Just pick Haaland, trust me.")], stop="end_turn")] * 2
    result = Gaffer(client=FakeClient(chatty), store=Store(tmp_path / "g.db")).run("hi", Toolbox(model, TeamState(team_id=None)))
    assert result.status == "rejected" and result.recommendation is None


def test_tool_exceptions_go_back_to_the_model(model, tmp_path):
    script = [
        lambda _: reply([tool_call("get_player", {"player_id": 424242}, "x")]),
        lambda m: reply([NS(type="text", text=m[-1]["content"][0]["content"])], stop="end_turn"),
        lambda _: reply([NS(type="text", text="ok")], stop="end_turn"),
    ]
    fake = FakeClient(script)
    Gaffer(client=fake, store=Store(tmp_path / "g.db")).run("hi", Toolbox(model, TeamState(team_id=None)))
    err = fake.requests[1]["messages"][-1]["content"][0]
    assert err["is_error"] and "No player with id 424242" in err["content"]
