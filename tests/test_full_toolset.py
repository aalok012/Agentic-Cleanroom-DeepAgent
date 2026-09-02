"""The exploratory full-toolset arm: backend, toolset and tool-call accounting.

Offline — the model is scripted. These assert that the arm really does what it claims (a real
on-disk backend, a working shell, per-agent tool counts including the built-ins) and, just as
importantly, that standing it up did NOT weaken the clean-room arm next door.
"""

from __future__ import annotations

import pytest

from src.cleanroom.agents.deep import runtime as deep
from src.cleanroom.experiments import full_toolset as ft
from tests.conftest import script

pytestmark = pytest.mark.skipif(
    not deep.deepagents_available(), reason="deepagents not installed")


def probe(*turns):
    return script(*turns, *["done"] * 8)


# --- the backend is real, and it really executes ------------------------------------
def test_backend_is_on_disk_and_supports_execute(tmp_path):
    """The whole point of this arm: a persistent backend whose files you can inspect, and a
    shell the agent can actually run commands in."""
    backend = ft.build_backend(tmp_path / "run")

    assert type(backend).__name__ == "LocalShellBackend"
    result = backend.execute("echo full-toolset-probe")
    assert result.exit_code == 0
    assert "full-toolset-probe" in result.output


def test_seeded_files_land_on_the_real_disk(tmp_path):
    """Unlike the clean-room arm's StateBackend, seeding writes inspectable files."""
    root = tmp_path / "run"
    ft.seed_disk(root, {"spec/1_1.md": "# FR 1.1\n"})

    assert (root / "spec" / "1_1.md").read_text() == "# FR 1.1\n"
    assert ft.read_disk(root, "spec/1_1.md") == "# FR 1.1\n"
    assert ft.read_disk(root, "spec/missing.md") == "", "a missing file must not raise"


# --- tool-call accounting: the new information this pass exists to gather ------------
def test_tool_calls_are_counted_including_built_ins(tmp_path, fake_llm):
    """The report's headline is WHICH tools each agent reached for, so the counter must see
    the deepagents built-ins (write_todos/ls/execute), not just our own submit_* tools."""
    model = probe(
        ("write_todos", {"todos": [{"content": "look around", "status": "pending"}]}),
        ("ls", {"path": "."}),
        ("ls", {"path": "spec"}),
        "surveyed")
    fake_llm(model)

    agent = ft.build_full_agent([], "You are under test.", ft.build_backend(tmp_path / "run"),
                                name="probe")
    state, seconds = ft.invoke_full(agent, "probe the filesystem", max_steps=20)

    log = ft.RunLog()
    row = log.record("probe", "unit-1", state, seconds, 20)

    assert row["tool_calls"]["ls"] == 2, "built-in tool calls were not counted"
    assert row["tool_calls"]["write_todos"] == 1
    assert row["tool_calls_total"] == 3
    assert row["assistant_turns"] >= 3


def test_run_log_aggregates_per_agent():
    """by_agent() is what the report table is built from."""
    log = ft.RunLog()
    log.rows = [
        {"agent": "code", "unit": "1", "assistant_turns": 4, "max_steps": 80, "seconds": 1.0,
         "tool_calls_total": 3, "tool_calls": {"write_todos": 1, "submit_implementation": 2}},
        {"agent": "code", "unit": "2", "assistant_turns": 2, "max_steps": 80, "seconds": 2.0,
         "tool_calls_total": 2, "tool_calls": {"execute": 1, "submit_implementation": 1}},
        {"agent": "test", "unit": "1", "assistant_turns": 1, "max_steps": 80, "seconds": 0.5,
         "tool_calls_total": 1, "tool_calls": {"ls": 1}},
    ]

    per_agent = log.by_agent()
    assert per_agent["code"]["invocations"] == 2
    assert per_agent["code"]["assistant_turns"] == 6
    assert per_agent["code"]["seconds"] == 3.0
    assert per_agent["code"]["tools"]["submit_implementation"] == 3
    assert per_agent["code"]["tools"]["execute"] == 1
    assert per_agent["test"]["invocations"] == 1


def test_report_carries_the_isolation_warning():
    """Artifacts from this arm look identical to the clean-room arm's; only the report tells
    them apart, so the warning must be IN the report rather than only in a docstring."""
    payload = ft.RunLog().as_dict()

    assert payload["isolation"], "the run report shipped without the isolation caveat"
    joined = " ".join(payload["isolation"]).lower()
    assert "no clean-room isolation" in joined
    assert "unsandboxed" in joined


# --- the clean-room arm must be untouched -------------------------------------------
def test_experiments_never_imported_by_the_clean_room_drivers():
    """The two arms must not converge. If a deep driver ever imports this package, the AST
    guards in test_isolation.py would be protecting a module that no longer decides the
    backend — the guarantee would be gone while the tests still passed."""
    import pathlib

    offenders = [
        p.name for p in sorted(pathlib.Path("src/cleanroom/agents/deep").glob("*.py"))
        if "experiments" in p.read_text()
    ]
    assert not offenders, f"clean-room driver(s) {offenders} reference the exploratory arm"


def test_full_toolset_arm_is_outside_the_guarded_package():
    """This arm uses root_dir and a disk backend by design. It stays outside agents/deep/ so
    test_isolation.py's guard keeps failing the build if those appear in the real drivers."""
    import pathlib

    guarded = {p.name for p in pathlib.Path("src/cleanroom/agents/deep").glob("*.py")}
    ours = pathlib.Path(ft.__file__)

    assert ours.name not in guarded
    assert "experiments" in ours.parts, "the exploratory arm moved inside the guarded package"
