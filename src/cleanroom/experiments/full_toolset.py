"""Full-toolset agent runtime — the exploratory arm.

Every agent here gets the COMPLETE deepagents built-in toolset on a REAL, on-disk backend:

    planning    write_todos
    filesystem  ls, read_file, write_file, edit_file, glob, grep
    shell       execute

===========================  READ THIS BEFORE REUSING  ===========================
THIS ARM HAS NO CLEAN-ROOM ISOLATION. Every agent shares one ``LocalShellBackend`` rooted at
the run directory, so the Test Agent can read the Code Agent's implementation, the Code Agent
can read the test suite, and ``execute`` gives all of them a real shell on the host.

That is the POINT of this module — it exists to see what the agents do with a full toolset —
but it means results produced here are NOT evidence for the paper's independent-derivation
claim. A test that passes in this arm may pass because the test agent read the code.

It lives outside ``agents/deep/`` on purpose. ``tests/test_isolation.py`` fails the build if
``FilesystemBackend``, ``root_dir`` or ``subagents`` appears in any ``agents/deep/*.py``, and
that guard must keep protecting the clean-room drivers. Do not import this module from there,
and do not "unify" the two runtimes.
=================================================================================

The prompts are reused verbatim from the clean-room drivers so the only variable in an A/B is
the toolset and the backend. Their "you will never be given the implementation" paragraphs are
therefore now FALSE in this arm — noted in :func:`isolation_notes`, which the report prints.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.cleanroom.utils.llm_client import get_llm

# Where each agent's artifacts land inside the shared run root. Same names as the clean-room
# virtual roots so a diff between the two arms lines up, but these are REAL directories.
SPEC_DIR = "spec"
CODE_DIR = "code"
TEST_DIR = "tests"
PROOF_DIR = "proof"
SKILL_DIR = "skills"


def full_toolset_max_steps(default: int = 80) -> int:
    """Step budget per agent invocation (LangGraph ``recursion_limit``).

    Higher default than the clean-room arm's 60: a shell-enabled agent spends turns on
    ``execute`` that the state-backed one never had, and starving it would confound the
    comparison this arm exists to make. Tune with CLEANROOM_FULL_MAX_STEPS.
    """
    try:
        return max(4, int(os.getenv("CLEANROOM_FULL_MAX_STEPS", str(default))))
    except ValueError:
        return default


def build_backend(root: Path):
    """One shared on-disk backend for the whole run.

    ``LocalShellBackend`` is the only built-in backend that supports ``execute`` (verified
    against deepagents 0.6.12 — ``FilesystemBackend`` has no execute), so it is what the
    "full toolset" in this arm's name actually requires.

    ``inherit_env=False``: the agent gets a clean environment rather than the parent's, which
    keeps the pipeline's own API keys out of a shell the model drives. That is a credential
    boundary, not an isolation one — the filesystem stays fully shared by design.

    ``virtual_mode=True`` is set EXPLICITLY, for two reasons. It silences a deprecation warning
    (the default flips in a later deepagents), and with it False the backend's own docs say
    absolute paths and ``..`` bypass ``root_dir`` entirely — so the file tools could wander into
    the repository, or anywhere else on the machine.

    !! It does NOT sandbox ``execute``. Per deepagents' own note, ``virtual_mode`` applies path
    semantics to the FILE tools only; shell commands still run unrestricted on the host, as the
    user running the pipeline. A model that decides to `rm -rf` something, curl an endpoint, or
    read ~/.ssh can. Run this arm somewhere you are willing to have a model hold a shell.
    """
    from deepagents.backends import LocalShellBackend

    root.mkdir(parents=True, exist_ok=True)
    return LocalShellBackend(root_dir=root, virtual_mode=True, timeout=120, inherit_env=False)


def build_full_agent(tools: list, system_prompt: str, backend, *, name: str,
                     model: str | None = None, temperature: float = 0.0):
    """A deep agent with the full built-in toolset on ``backend``.

    ``create_deep_agent`` always installs write_todos and the filesystem tools; ``execute``
    comes from the backend supporting it. So "the full toolset" is a property of the backend
    choice plus the absence of a ``permissions=`` restriction — there is no positive flag to
    set, and passing ``tools=`` only ADDS the submit tools on top.
    """
    from deepagents import create_deep_agent

    llm = get_llm(model, temperature=temperature) if model else get_llm(temperature=temperature)
    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        backend=backend,
        name=name,
    )


def _tool_calls(state: dict) -> Counter:
    """How many times the agent called each tool, built-ins included.

    Counted off the assistant messages' ``tool_calls`` rather than the ToolMessages, so a call
    the backend rejected (a blocked ``execute``, a write outside the root) still counts as an
    attempt — which is exactly what this arm is trying to observe.
    """
    counts: Counter = Counter()
    for message in state.get("messages") or []:
        calls = getattr(message, "tool_calls", None)
        if not calls and isinstance(message, dict):
            calls = message.get("tool_calls")
        for call in calls or []:
            nm = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if nm:
                counts[nm] += 1
    return counts


def _assistant_turns(state: dict) -> int:
    return sum(1 for m in (state.get("messages") or [])
               if getattr(m, "type", None) == "ai"
               or (isinstance(m, dict) and m.get("role") == "assistant"))


class RunLog:
    """Per-agent tool-call and timing accounting for the whole run.

    The pipeline already tracks tokens/latency/cost through GLOBAL_HANDLER on the shared LLM
    client, and this arm changes nothing about that. What is NEW here is which TOOLS each
    agent reached for, which is the question this pass was set up to answer.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, agent: str, unit: str, state: dict, seconds: float,
               max_steps: int) -> dict:
        counts = _tool_calls(state)
        row = {
            "agent": agent,
            "unit": unit,                       # which feature/FR this invocation covered
            "assistant_turns": _assistant_turns(state),
            "max_steps": max_steps,
            "seconds": round(seconds, 2),
            "tool_calls_total": sum(counts.values()),
            "tool_calls": dict(sorted(counts.items())),
        }
        self.rows.append(row)
        return row

    def by_agent(self) -> dict[str, dict]:
        """``{agent: {invocations, turns, seconds, tools: {name: n}}}`` — the report table."""
        out: dict[str, dict] = {}
        for row in self.rows:
            agg = out.setdefault(row["agent"], {
                "invocations": 0, "assistant_turns": 0, "seconds": 0.0,
                "tool_calls_total": 0, "tools": Counter(),
            })
            agg["invocations"] += 1
            agg["assistant_turns"] += row["assistant_turns"]
            agg["seconds"] += row["seconds"]
            agg["tool_calls_total"] += row["tool_calls_total"]
            agg["tools"].update(row["tool_calls"])
        for agg in out.values():
            agg["tools"] = dict(sorted(agg["tools"].items(), key=lambda kv: -kv[1]))
            agg["seconds"] = round(agg["seconds"], 2)
        return out

    def as_dict(self) -> dict:
        return {"per_invocation": self.rows, "per_agent": self.by_agent(),
                "isolation": isolation_notes()}


def invoke_full(agent, prompt: str, *, max_steps: int | None = None) -> tuple[dict, float]:
    """Run one agent to completion. Returns ``(state, seconds)``.

    No ``files`` are passed: unlike the clean-room arm, this agent's filesystem IS the run
    directory on disk, seeded by writing real files before the call.
    """
    steps = max_steps or full_toolset_max_steps()
    started = time.time()
    state = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                         {"recursion_limit": steps})
    return state, time.time() - started


def seed_disk(root: Path, files: dict[str, str]) -> list[Path]:
    """Write ``{relative_path: content}`` under the run root for an agent to read."""
    written: list[Path] = []
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content or "")
        written.append(dest)
    return written


def read_disk(root: Path, rel: str) -> str:
    """Content of one file under the run root, or "" if the agent never created it."""
    path = root / rel
    try:
        return path.read_text()
    except OSError:
        return ""


def isolation_notes() -> list[str]:
    """What this arm gives up, printed into the run report so a result is never mistaken.

    Written down rather than left implicit because the artifacts this arm produces look
    exactly like the clean-room arm's, and only the report distinguishes them.
    """
    return [
        "NO clean-room isolation: all agents share one on-disk backend rooted at the run dir.",
        "The Test Agent can read generated code; the Code Agent can read the test suite.",
        "`execute` runs UNSANDBOXED on the host as the invoking user; virtual_mode guards the "
        "file tools' paths but explicitly does not restrict shell commands.",
        "Prompts still claim the other pool is invisible — that claim is FALSE in this arm.",
        "Results here are NOT evidence for independent derivation. Use the clean-room arm.",
    ]
