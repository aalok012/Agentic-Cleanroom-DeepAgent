"""Shared runtime for the deepagents-backed repair loops.

The pipeline's deterministic loops call the LLM once per broken artifact and re-run the
checker themselves. The drivers in this package hand that control to a LangChain
``deepagents`` agent instead: it plans with ``write_todos``, reads and edits the artifacts
through the built-in filesystem tools, and calls a checker tool itself until the check
passes or its step budget runs out.

==========================  WHY THE VIRTUAL FILESYSTEM  ==========================
``create_deep_agent`` ALWAYS installs its built-in file tools (``ls``/``read_file``/
``edit_file``/``glob``/``grep``) — passing ``tools=`` is additive and cannot remove them.
So the tool list is NOT where isolation is enforced. Isolation comes from the backend:
we use the default ``StateBackend``, a virtual filesystem held in LangGraph state that
contains EXACTLY the files we seed. The real disk is unreachable from inside the agent,
and ``execute`` fails closed because ``StateBackend`` is not a sandbox backend.

That is what lets the Java compile-repair driver keep the guarantee documented in
``agents/code/compile_repair.py``: the code-repair agent is seeded with generated CODE
files only, so it cannot read a test source even if it tries.

Do NOT express that split with deepagents *subagents* — a subagent shares the parent's
filesystem state, which would silently reunite the two pools. Use separate top-level
agent invocations, one per isolation pool, as the drivers here do.
=================================================================================
"""

from __future__ import annotations

import os
import re
from typing import Any

from src.cleanroom.utils.llm_client import get_llm

# Virtual-filesystem roots inside the agent's StateBackend. Distinct per pool so a stray
# glob in one driver can never name a path belonging to another.
CODE_ROOT = "/code"
TEST_ROOT = "/tests"


class DeepAgentUnavailable(RuntimeError):
    """Raised when the deepagents extra is not installed."""


def deepagents_available() -> bool:
    try:
        import deepagents  # noqa: F401
    except Exception:
        return False
    return True


def _require_deepagents():
    try:
        from deepagents import create_deep_agent
        from deepagents.backends.utils import create_file_data
    except Exception as exc:  # pragma: no cover - import guard
        raise DeepAgentUnavailable(
            "the deepagents driver needs the `deepagents` package: `uv add deepagents`"
        ) from exc
    return create_deep_agent, create_file_data


def deep_max_steps(default: int = 60) -> int:
    """Hard cap on agent<->tool turns per invocation (LangGraph ``recursion_limit``).

    This is the deep driver's cost ceiling — the agent decides how many repair rounds to
    take, so the budget is a step count rather than the deterministic loop's round count.
    Tune with CLEANROOM_DEEP_MAX_STEPS."""
    try:
        return max(4, int(os.getenv("CLEANROOM_DEEP_MAX_STEPS", str(default))))
    except ValueError:
        return default


def seed_files(mapping: dict[str, str]) -> dict[str, Any]:
    """Turn ``{virtual_path: content}`` into the ``files`` state the agent starts with."""
    _, create_file_data = _require_deepagents()
    return {path: create_file_data(content or "") for path, content in mapping.items()}


def read_virtual_files(paths: list[str]) -> dict[str, str]:
    """Current contents of ``paths`` in the agent's virtual filesystem.

    Call this from INSIDE a tool: ``StateBackend`` resolves LangGraph state through
    ``get_config()`` at call time, so a checker tool sees the edits the agent has made so
    far in this same run — which is what makes an agent-driven check-fix-recheck loop work.
    """
    from deepagents.backends import StateBackend

    backend = StateBackend()
    out: dict[str, str] = {}
    for path in paths:
        result = backend.read(path, limit=1_000_000)
        if result.error or not result.file_data:
            continue
        out[path] = result.file_data.get("content", "")
    return out


def build_agent(tools: list, system_prompt: str, *, temperature: float = 0.0,
                model: str | None = None, name: str | None = None):
    """A deep agent on the project's own LLM client.

    Passing our ``get_llm()`` instance keeps the whole tool loop on the configured
    OpenAI-compatible endpoint AND keeps GLOBAL_HANDLER attached, so every call the agent
    makes lands in the run's existing token/latency/cost accounting for free.
    """
    create_deep_agent, _ = _require_deepagents()
    llm = get_llm(model, temperature=temperature) if model else get_llm(temperature=temperature)
    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        name=name,
    )


def invoke_agent(agent, prompt: str, files: dict[str, Any], *, max_steps: int | None = None) -> dict:
    """Run the agent to completion and return its final state (``files`` included)."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}], "files": files},
        {"recursion_limit": max_steps or deep_max_steps()},
    )


def final_files(state: dict) -> dict[str, str]:
    """``{virtual_path: content}`` from a finished agent state."""
    return {path: (data or {}).get("content", "")
            for path, data in (state.get("files") or {}).items()}


def virtual_path(root: str, index: int, name: str) -> str:
    """Stable, collision-free virtual path for artifact ``index``.

    The index prefix is what the drivers map back to a position in ``generated_code`` /
    ``generated_tests``, so an agent that renames nothing stays trivially re-attachable.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name or f"file{index}")
    return f"{root}/{index:02d}_{safe}"


def index_from_virtual_path(path: str) -> int | None:
    """Inverse of :func:`virtual_path` — the artifact index, or None if unrecognized."""
    m = re.match(r"^/[^/]+/(\d+)_", path or "")
    return int(m.group(1)) if m else None


def agent_step_count(state: dict) -> int:
    """Number of assistant turns the agent took — recorded as the loop's effort metric."""
    return sum(1 for m in (state.get("messages") or [])
               if getattr(m, "type", None) == "ai" or (isinstance(m, dict) and m.get("role") == "assistant"))
