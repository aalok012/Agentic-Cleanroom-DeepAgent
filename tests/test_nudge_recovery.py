"""The agent stops one message short of submitting; the nudge must recover it."""
from src.cleanroom.agents.deep import generation as G


class FlakyAgent:
    """Narrates on the first invocation, submits when asked again — the observed failure."""
    def __init__(self, submit_fn, fr_id):
        self.submit_fn, self.fr_id, self.calls = submit_fn, fr_id, 0

    def invoke(self, state, config):
        self.calls += 1
        msgs = list(state.get("messages") or [])
        if self.calls == 1:
            msgs.append({"role": "assistant",
                         "content": "The implementation has been successfully submitted."})
        else:
            self.submit_fn.invoke({"fr_id": self.fr_id, "content": "def f():\n    return 1\n"})
            msgs.append({"role": "assistant", "content": "Submitted."})
        return {**state, "messages": msgs}


def test_nudge_recovers_unsubmitted_fr(monkeypatch):
    captured = {}

    def fake_build_agent(tools, prompt, **kw):
        submit = next(t for t in tools if t.name == "submit_implementation")
        agent = FlakyAgent(submit, "1.1")
        captured["agent"] = agent
        return agent

    monkeypatch.setattr(G, "build_agent", fake_build_agent)
    monkeypatch.setattr(G, "invoke_agent",
                        lambda a, p, f, max_steps=None: a.invoke({"messages": [], "files": f}, {}))
    monkeypatch.setattr(G, "nudge_agent",
                        lambda a, st, m, max_steps=None: a.invoke(
                            {**st, "messages": list(st.get("messages") or []) + [{"role": "user", "content": m}]}, {}))

    ir = {"features": [{"id": "1", "name": "F"}]}
    contracts = [{"fr_id": "1.1", "feature_id": "1", "signature": "def f() -> int",
                  "docstring": "d", "file_path": "x.py", "mvc_layer": "controller"}]

    files, metrics = G.deep_generate_code(ir, contracts, language="python", stack="fastapi")

    assert captured["agent"].calls == 2, "should have been nudged exactly once"
    assert metrics["frs_submitted"] == 1
    assert len(files) == 1
    print("PASS: narrated first, submitted after nudge, no exception raised")
