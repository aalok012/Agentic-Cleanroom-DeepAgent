"""Shared fixtures for the deepagents driver tests.

Every test here runs OFFLINE: the model is a scripted fake, so the assertions are about the
driver's wiring and isolation guarantees, not about model quality.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedModel(GenericFakeChatModel):
    """A fake chat model that replays a fixed list of AIMessages.

    ``GenericFakeChatModel`` does not implement ``bind_tools``, which the deep agent calls
    before every turn; returning ``self`` makes it usable as an agent model.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def script(*turns) -> ScriptedModel:
    """Build a ScriptedModel from ``(tool_name, args)`` tuples, ending with a plain reply.

    A trailing string turn becomes the agent's final text answer.
    """
    messages = []
    for i, turn in enumerate(turns):
        if isinstance(turn, str):
            messages.append(AIMessage(content=turn))
        else:
            name, args = turn
            messages.append(AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": f"call{i}"}]))
    return ScriptedModel(messages=iter(messages))


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a scripted model as the project's LLM for the duration of a test."""

    def install(model):
        for module in ("src.cleanroom.agents.deep.runtime",
                       "src.cleanroom.experiments.full_toolset"):
            monkeypatch.setattr(f"{module}.get_llm", lambda *a, **k: model)
        return model

    return install
