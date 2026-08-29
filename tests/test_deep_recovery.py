"""Deepagents recovery driver: contract identity, change detection, and spec-sheet content.

Recovery step (b) is the pipeline's sanctioned clean-room break, so test visibility here is
expected. What must NOT drift is who owns each field of the regenerated file: the planner's
contract owns the identity, the agent owns only the body.
"""

from __future__ import annotations

import pytest

from src.cleanroom.agents.deep import recovery as deep
from tests.conftest import script

PRIOR = "def add(a, b):\n    return a - b\n"


def make_ir() -> dict:
    return {
        "features": [{"id": "1", "functional_requirements": [
            {"id": "1.1", "text": "The system shall add two numbers."}]}],
        "planning": {"contracts": [{
            "fr_id": "1.1", "feature_id": "1", "mvc_layer": "model",
            "file_path": "models/add.py", "signature": "def add(a, b):",
            "docstring": "Return the sum of a and b.",
            "example_inputs_json": '{"a": 1, "b": 2}', "expected_return_json": "3",
            "error_mode": "raise",
        }]},
        "generated_code": {"files": [{
            "fr_id": "1.1", "feature_id": "1", "path": "models/add.py",
            "mvc_layer": "model", "content": PRIOR}]},
    }


FAILURES = [{"fr_id": "1.1", "description": "adds two positives",
             "inputs": {"a": 1, "b": 2}, "expected": 3, "reason": "got -1"}]


def test_identity_comes_from_the_contract_not_the_agent(fake_llm):
    """The agent rewrites the body; fr_id / feature_id / path / mvc_layer come from planning."""
    ir = make_ir()
    fake_llm(script(
        ("read_file", {"file_path": "/code/00_1_1.py"}),
        ("submit_repair", {"fr_id": "1.1", "content": "def add(a, b):\n    return a + b\n"}),
        "fixed",
    ))
    files, metrics = deep.deep_regenerate_with_test_feedback(ir, {"1"}, FAILURES)

    assert len(files) == 1
    f = files[0]
    assert (f.fr_id, f.feature_id, f.path, f.mvc_layer) == ("1.1", "1", "models/add.py", "model")
    assert f.content == "def add(a, b):\n    return a + b\n"
    assert metrics["frs_changed"] == 1


def test_untouched_files_are_not_returned(fake_llm):
    """An FR the agent left alone must not be swapped back in as a no-op rewrite."""
    ir = make_ir()
    fake_llm(script(("read_file", {"file_path": "/code/00_1_1.py"}), "nothing to change"))
    files, metrics = deep.deep_regenerate_with_test_feedback(ir, {"1"}, FAILURES)
    assert files == []
    assert metrics["frs_changed"] == 0
    assert metrics["frs_targeted"] == 1


def test_spec_sheet_carries_requirement_contract_and_failing_cases(fake_llm):
    """The read-only briefing must contain what the deterministic prompt passed per FR."""
    contract = make_ir()["planning"]["contracts"][0]
    sheet = deep._spec_sheet(contract, "The system shall add two numbers.", FAILURES)

    assert "The system shall add two numbers." in sheet
    assert "def add(a, b):" in sheet
    assert "Return the sum of a and b." in sheet
    assert "adds two positives" in sheet
    assert "got -1" in sheet
    assert '{"a": 1, "b": 2}' in sheet


def test_spec_files_are_separate_from_code_files(fake_llm):
    """Spec sheets live under /spec and are never mistaken for regenerated source."""
    ir = make_ir()
    fake_llm(script(
        ("edit_file", {"file_path": "/spec/00_1_1.md",
                       "old_string": "# FR 1.1", "new_string": "# EDITED"}),
        ("submit_repair", {"fr_id": "1.1", "content": "def add(a, b):\n    return a + b\n"}),
        "done",
    ))
    files, _ = deep.deep_regenerate_with_test_feedback(ir, {"1"}, FAILURES)
    assert len(files) == 1
    assert "I edited the spec" not in files[0].content


def test_no_contracts_raises(fake_llm):
    with pytest.raises(ValueError, match="planning.contracts"):
        deep.deep_regenerate_with_test_feedback({"planning": {}}, {"1"}, FAILURES)


def test_unknown_feature_short_circuits(fake_llm):
    ir = make_ir()
    files, metrics = deep.deep_regenerate_with_test_feedback(ir, {"99"}, FAILURES)
    assert files == []
    assert "skipped" in metrics


def test_submit_repair_rejects_an_unknown_fr(fake_llm):
    """The agent names an FR, never a path — so a bad name must be refused, not written."""
    ir = make_ir()
    fake_llm(script(
        ("submit_repair", {"fr_id": "9.9", "content": "malicious"}),
        "tried",
    ))
    files, metrics = deep.deep_regenerate_with_test_feedback(ir, {"1"}, FAILURES)
    assert files == []
    assert metrics["frs_submitted"] == 0


def test_submit_repair_rejects_empty_content(fake_llm):
    """An empty submission must not blank out a working file."""
    ir = make_ir()
    fake_llm(script(("submit_repair", {"fr_id": "1.1", "content": "   "}), "oops"))
    files, metrics = deep.deep_regenerate_with_test_feedback(ir, {"1"}, FAILURES)
    assert files == []
    assert metrics["frs_submitted"] == 0


def test_edit_file_is_picked_up_when_the_agent_does_not_submit(fake_llm):
    """Surgical `edit_file` repairs are collected from the virtual filesystem."""
    ir = make_ir()
    fake_llm(script(
        ("read_file", {"file_path": "/code/00_1_1.py"}),
        ("edit_file", {"file_path": "/code/00_1_1.py",
                       "old_string": "a - b", "new_string": "a + b"}),
        "fixed surgically",
    ))
    files, metrics = deep.deep_regenerate_with_test_feedback(ir, {"1"}, FAILURES)
    assert len(files) == 1
    assert files[0].content == "def add(a, b):\n    return a + b\n"
    assert metrics["frs_submitted"] == 0
