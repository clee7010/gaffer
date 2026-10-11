"""The agent loop: Claude + tools + guardrails, with every step audited.

A hand-written loop rather than the SDK's tool runner because the guardrail
is part of the control flow: it counts submission attempts, bounces failed
recommendations back as tool errors, and refuses to end the run on prose alone.

Two backends share this loop. `anthropic` is Claude via the API. `ollama` is a
free local model (default Qwen3 4B) through Ollama's Anthropic-compatible
endpoint, so the same SDK, tools and guardrails run with no API key or cost.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import anthropic

from .store import Store
from .tools import TOOL_SCHEMAS, Toolbox

PROVIDER = os.getenv("GAFFER_PROVIDER", "anthropic")  # anthropic | ollama
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "ollama": "qwen3:4b"}
EFFORT = os.getenv("GAFFER_EFFORT", "medium")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

SYSTEM_PROMPT = """\
You are the Gaffer: a sharp, dry-humoured Fantasy Premier League manager who \
trusts data over hunches. You advise one FPL manager on their team for the \
coming gameweek.

How you work:
- Start with get_team_overview. Use the tools for every fact. Your memory of \
players, prices, form or injuries is out of date; the tools are live.
- Projections come from a Poisson team-strength model and a per-player expected \
points model. Use get_player to see why a player projects well before you back \
them, and fixture_outlook for runs of fixtures.
- optimise_transfers solves the exact integer program. Treat it as a strong \
second opinion, not an oracle: it ignores things like price rises and how \
reliable the news is. If you overrule it, say why.
- A -4 hit has to pay for itself over the horizon. Rolling a free transfer is \
a legitimate answer.
- Finish by calling submit_recommendation. Put every number you quote in the \
summary into key_claims. If the guardrails reject it, fix exactly what they \
flag and resubmit. Never argue with them; they check the rules and the model.
- The summary is for the manager: under 200 words, with the reasoning, the \
risks, and one line of personality. No markdown tables.
"""

ToolEvent = Callable[[str, dict, dict | list | str, bool], None]


@dataclass
class AgentResult:
    run_id: int
    status: str  # accepted | rejected | error
    answer: str
    recommendation: dict | None
    guardrail_attempts: int
    usage: dict = field(default_factory=dict)
    tool_calls: int = 0


class Gaffer:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        store: Store | None = None,
        provider: str = PROVIDER,
        model: str | None = None,
        effort: str = EFFORT,
        max_turns: int = 24,
        max_guardrail_attempts: int = 4,
    ):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"provider must be one of {list(DEFAULT_MODELS)}")
        self.provider = provider
        if client is None:
            client = (
                anthropic.Anthropic(base_url=OLLAMA_URL, api_key="ollama", timeout=900)
                if provider == "ollama"
                else anthropic.Anthropic()
            )
        self.client = client
        self.store = store or Store()
        self.model = model or os.getenv("GAFFER_MODEL") or DEFAULT_MODELS[provider]
        self.effort = effort
        self.max_turns = max_turns
        self.max_guardrail_attempts = max_guardrail_attempts

    def _create(self, messages: list):
        if self.provider == "ollama":
            # Ollama implements the core Messages API, not Claude's beta features.
            return self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            # On a safety decline, retry server-side on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def run(self, question: str, toolbox: Toolbox, on_tool: ToolEvent | None = None) -> AgentResult:
        team = toolbox.team
        run_id = self.store.start_run(team.team_id, toolbox.model.next_gw, question, self.model)
        messages: list = [{"role": "user", "content": question}]
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        attempts = n_tools = nudges = 0
        status, answer = "error", ""

        for _ in range(self.max_turns):
            try:
                resp = self._create(messages)
            except anthropic.APIError as e:
                answer = f"API error: {e}"
                break
            except TypeError as e:  # raised by the SDK when no credentials resolve
                answer = f"No Anthropic credentials found; set ANTHROPIC_API_KEY. ({e})"
                break
            for k in usage:
                usage[k] += getattr(resp.usage, k, 0) or 0
            messages.append({"role": "assistant", "content": resp.content})
            text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
            if text:
                answer = text

            if resp.stop_reason == "pause_turn":
                continue
            if resp.stop_reason in ("refusal", "max_tokens"):
                answer = answer or f"Stopped: {resp.stop_reason}"
                break
            if resp.stop_reason == "end_turn":
                if toolbox.accepted:
                    status = "accepted"
                    break
                if nudges == 0:
                    nudges += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please finalise by calling submit_recommendation so the guardrails can check it.",
                        }
                    )
                    continue
                status = "rejected"
                break

            results = []
            for block in (b for b in resp.content if b.type == "tool_use"):
                n_tools += 1
                t0 = time.perf_counter()
                try:
                    out = toolbox.call(block.name, dict(block.input))
                    is_error = False
                    if block.name == "submit_recommendation":
                        attempts += 1
                        self.store.log_guardrail(run_id, attempts, out)
                        is_error = not out["passed"]
                except Exception as e:  # noqa: BLE001 - tool bugs go back to the model, not up the stack
                    out, is_error = f"{type(e).__name__}: {e}", True
                latency = (time.perf_counter() - t0) * 1000
                self.store.log_tool(run_id, block.name, dict(block.input), is_error, latency)
                if on_tool:
                    on_tool(block.name, dict(block.input), out, is_error)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": out if isinstance(out, str) else json.dumps(out, default=str),
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

            if attempts >= self.max_guardrail_attempts and not toolbox.accepted:
                status = "rejected"
                answer = "The recommendation failed the guardrails too many times; not publishing it."
                break

        if toolbox.accepted:
            status = "accepted"
            answer = toolbox.accepted["summary"]
        self.store.finish_run(run_id, status, answer, toolbox.accepted, usage)
        return AgentResult(
            run_id=run_id,
            status=status,
            answer=answer,
            recommendation=toolbox.accepted,
            guardrail_attempts=attempts,
            usage=usage,
            tool_calls=n_tools,
        )
