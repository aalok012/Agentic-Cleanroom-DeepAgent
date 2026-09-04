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


def test_driver_forces_proof_revision_when_agent_never_verifies(fake_llm):
    """The observed Qwen3 behaviour: submit once, never call dafny_verify.

    Measured over 37 foodsaver features it called the verifier zero times on every one, and
    28 of 32 submitted modules did not even parse. The driver must run the loop itself and
    feed Dafny's diagnostics back, rather than recording the first draft as unproved.
    """
    from src.cleanroom.agents.deep.generation import deep_generate_dafny
    from tests.test_deep_generation import IR, CONTRACT, probe

    seen: list[str] = []

    def verifier(source: str):
        seen.append(source)
        ok = "datatype Model" in source          # only the corrected form parses
        return ok, "" if ok else "F.dfy(2,14): Error: invalid SynonymTypeDecl"

    # The agent never calls dafny_verify; it submits a bad module, then a good one only
    # after the driver hands back the parse error.
    fake_llm(probe(
        ("submit_dafny", {"module_name": "F", "source": "module FDomain refines Domain { type Model = { a: bool } }"}),
        "done",
        ("submit_dafny", {"module_name": "F", "source": "module FDomain refines Domain { datatype Model = Model(a: bool) }"}),
        "fixed now"))

    out, metrics = deep_generate_dafny(IR, "1", [CONTRACT], module="F",
                                       verifier=verifier, max_rounds=6)

    assert len(seen) == 2, f"driver should have verified both submissions, saw {len(seen)}"
    assert metrics["verified"] is True, "the corrected module must be recorded as verified"
    assert "datatype Model" in out["source"]
    print("PASS: agent never verified; driver forced a revision and the module now verifies")


def test_verifier_closure_survives_a_failing_verification(tmp_path, monkeypatch):
    """The bug that made every feature score unproved.

    DafnyAgent's verifier closure did `"\n".join(res.messages)`, but messages is a list of
    dicts, so it raised TypeError on every FAILING verification -- while a PASSING one returns
    an empty list and joins fine. The agent's dafny_verify tool caught the TypeError and told
    the model "the verifier could not be run ... continue without it", so no proof agent ever
    saw a Dafny error. 0/37 features verified.
    """
    from src.cleanroom.utils.dafny_verify import DafnyResult, format_messages

    failing = DafnyResult(ok=False, verified=False, errors=1,
                          messages=[{"line": 4, "col": 15, "message": "invalid SynonymTypeDecl"}])
    passing = DafnyResult(ok=True, verified=True, errors=0, messages=[])

    # The old expression, kept here so the regression is unambiguous.
    import pytest
    with pytest.raises(TypeError):
        "\n".join(failing.messages)

    text = format_messages(failing.messages)
    assert "invalid SynonymTypeDecl" in text
    assert "4:15" in text, "the agent needs the line and column to fix the error"
    assert format_messages(passing.messages) == ""
    print("PASS: a failing verification now renders as actionable text instead of raising")
