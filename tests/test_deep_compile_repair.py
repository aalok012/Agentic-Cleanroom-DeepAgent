"""Deepagents compile-repair driver: wiring, and the isolation it must not break.

The stage's contract (see ``agents/code/compile_repair``) is that the Code Agent sees compiler
diagnostics and nothing else. Handing the loop to an agent with built-in ``ls``/``read_file``
tools is exactly where that could silently regress, so the isolation tests below drive the
agent through the file tools deliberately and assert it comes up empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cleanroom.agents.code.compile_repair import CompileCheck
from src.cleanroom.agents.deep import compile_repair as deep

CODE_SRC = "class Gen1 { int x = BROKEN; }"
TEST_SRC = "class Feature_1Test { void t() { assertEquals(42, Gen1.run()); } }"


def make_ir() -> dict:
    return {
        "project_name": "demo",
        "generated_code": {"files": [
            {"fr_id": "1.1", "feature_id": "1", "path": "models/gen1.py",
             "mvc_layer": "model", "content": CODE_SRC},
        ]},
        "generated_tests": {"features": [
            {"feature_id": "1", "test_source": TEST_SRC},
        ]},
    }


@pytest.fixture
def stub_build(monkeypatch, tmp_path):
    """Replace the real javac/maven machinery with a content-driven stub.

    The build "fails" while the generated code still contains BROKEN, so the agent's
    ``compile_check`` tool reports success only once it has actually repaired the source in
    the IR — the same feedback shape the real loop provides.
    """
    code_path = (tmp_path / "src" / "Gen1.java").resolve()
    test_path = (tmp_path / "src" / "Feature_1Test.java").resolve()
    calls: dict[str, int] = {"rebuild": 0, "check": 0}

    state = {"broken_tests": False}

    def rebuild(**kwargs):
        calls["rebuild"] += 1
        return tmp_path

    def check(project_dir, stack, timeout):
        calls["check"] += 1
        ir = state["ir"]
        bad_code = "BROKEN" in (ir["generated_code"]["files"][0]["content"] or "")
        bad_test = state["broken_tests"] and "BROKEN" in (
            ir["generated_tests"]["features"][0]["test_source"] or "")
        diags = []
        if bad_code:
            diags.append(f"{code_path}:1: error: cannot find symbol BROKEN")
        if bad_test:
            diags.append(f"{test_path}:1: error: cannot find symbol BROKEN_ASSERT")
        if not diags:
            return CompileCheck(True, False, "javac", "javac compile-check passed", "")
        return CompileCheck(False, False, "javac", "javac compile-check failed", "\n".join(diags))

    monkeypatch.setattr(deep, "_rebuild_project", rebuild)
    monkeypatch.setattr(deep, "_run_compile_check", check)
    monkeypatch.setattr(deep, "_generated_source_map", lambda *a, **k: {code_path: 0})
    monkeypatch.setattr(deep, "_generated_test_map", lambda *a, **k: {test_path: 0})
    return {"calls": calls, "state": state, "code_path": code_path, "test_path": test_path,
            "tmp_path": tmp_path}


def run(ir, model, fake_llm, stub_build, **kwargs):
    stub_build["state"]["ir"] = ir
    fake_llm(model)
    return deep.run_java_compile_repair_deep(
        code_agent=object(), ir=ir, code_dir=stub_build["tmp_path"], stack="java",
        generated_tests=ir.get("generated_tests"), test_agent=None, **kwargs)


# --- wiring ------------------------------------------------------------------------
def test_agent_edits_are_committed_to_the_ir(fake_llm, stub_build):
    """An edit the agent makes in its virtual filesystem must land in ``ir['generated_code']``."""
    from tests.conftest import script

    ir = make_ir()
    model = script(
        ("compile_check", {}),
        ("edit_file", {"file_path": "/code/00_1_1.java",
                       "old_string": "BROKEN", "new_string": "0"}),
        ("compile_check", {}),
        "repaired",
    )
    metrics = run(ir, model, fake_llm, stub_build)

    assert ir["generated_code"]["files"][0]["content"] == "class Gen1 { int x = 0; }"
    assert metrics["ok"] is True
    assert metrics["driver"] == "deepagent"
    assert metrics["agent"]["code"]["files_changed"] == 1


def test_final_edit_after_last_compile_check_is_still_committed(fake_llm, stub_build):
    """The agent may edit and then stop without re-checking; that edit must not be dropped."""
    from tests.conftest import script

    ir = make_ir()
    model = script(
        ("compile_check", {}),
        ("edit_file", {"file_path": "/code/00_1_1.java",
                       "old_string": "BROKEN", "new_string": "7"}),
        "done, did not re-check",
    )
    run(ir, model, fake_llm, stub_build)
    assert ir["generated_code"]["files"][0]["content"] == "class Gen1 { int x = 7; }"


def test_identity_fields_are_never_taken_from_the_agent(fake_llm, stub_build):
    """The agent authors content only — ids and paths stay as the planner set them."""
    from tests.conftest import script

    ir = make_ir()
    model = script(
        ("edit_file", {"file_path": "/code/00_1_1.java",
                       "old_string": "BROKEN", "new_string": "0"}),
        "done",
    )
    run(ir, model, fake_llm, stub_build)
    f = ir["generated_code"]["files"][0]
    assert (f["fr_id"], f["feature_id"], f["path"], f["mvc_layer"]) == (
        "1.1", "1", "models/gen1.py", "model")


# --- isolation ---------------------------------------------------------------------
def test_code_agent_filesystem_contains_no_test_sources(fake_llm, stub_build):
    """``ls`` over the whole virtual filesystem must not surface a test source.

    This is the structural half of the guarantee: the code agent's filesystem is seeded with
    generated code only, so there is no path for it to read even if it looks.
    """
    from tests.conftest import script

    ir = make_ir()
    model = script(("ls", {"path": "/"}), ("glob", {"pattern": "**/*"}), "looked around")
    run(ir, model, fake_llm, stub_build)

    from langchain_core.messages import ToolMessage  # noqa: PLC0415

    # Re-run capturing the tool output the agent received.
    ir2 = make_ir()
    stub_build["state"]["ir"] = ir2
    model2 = script(("ls", {"path": "/"}), ("glob", {"pattern": "**/*"}), "looked around")
    fake_llm(model2)
    seen: list[str] = []
    original = deep.invoke_agent

    def capture(agent, prompt, files, **kw):
        state = original(agent, prompt, files, **kw)
        seen.extend(m.content for m in state["messages"] if isinstance(m, ToolMessage))
        return state

    import src.cleanroom.agents.deep.compile_repair as mod
    mod.invoke_agent = capture
    try:
        mod.run_java_compile_repair_deep(
            code_agent=object(), ir=ir2, code_dir=stub_build["tmp_path"], stack="java",
            generated_tests=ir2.get("generated_tests"), test_agent=None)
    finally:
        mod.invoke_agent = original

    blob = "\n".join(seen)
    assert "assertEquals" not in blob, "a generated TEST source leaked into the code agent"
    assert "Feature_1Test" not in blob
    assert "/tests" not in blob


def test_code_agent_diagnostics_hide_test_error_text(fake_llm, stub_build):
    """Test-source compile errors reach the code agent as a count, never as content."""
    stub_build["state"]["broken_tests"] = True
    ir = make_ir()
    ir["generated_tests"]["features"][0]["test_source"] = "class Feature_1Test { BROKEN_ASSERT }"

    from src.cleanroom.agents.code.compile_repair import _parse_compile_errors

    chk = CompileCheck(
        False, False, "javac", "failed",
        f"{stub_build['code_path']}:1: error: cannot find symbol BROKEN\n"
        f"{stub_build['test_path']}:1: error: cannot find symbol BROKEN_ASSERT",
    )
    errors = _parse_compile_errors(chk.diagnostics, stub_build["tmp_path"])
    from src.cleanroom.agents.code.compile_repair import _map_errors

    mapped_code, mapped_tests, _ = _map_errors(
        errors, {stub_build["code_path"]: 0}, {stub_build["test_path"]: 0})
    text = deep._code_diagnostics(stub_build["tmp_path"], chk, mapped_code, mapped_tests)

    assert "BROKEN_ASSERT" not in text, "test diagnostics leaked to the code agent"
    assert "BROKEN" in text                      # its own error is present
    assert "deliberately not shown" in text      # and the omission is declared


# --- short circuits ----------------------------------------------------------------
def test_disabled_when_max_rounds_is_zero(stub_build):
    ir = make_ir()
    m = deep.run_java_compile_repair_deep(
        code_agent=object(), ir=ir, code_dir=Path("."), stack="java", max_rounds=0)
    assert m["skipped"] and m["reason"] == "disabled"


def test_skipped_without_generated_code(stub_build):
    m = deep.run_java_compile_repair_deep(
        code_agent=object(), ir={}, code_dir=Path("."), stack="java")
    assert m["skipped"] and m["reason"] == "no generated_code"
