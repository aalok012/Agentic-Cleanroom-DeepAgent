"""Stage 1 + Stage 2 drivers: wiring, validation feedback, and identity handling.

Offline — the model is scripted, so these assert on the drivers' contracts with the rest of
the pipeline (what they return, what they reject, where identity comes from), not on model
quality. The isolation guarantees themselves live in ``test_isolation.py``.
"""

from __future__ import annotations

import pytest

from src.cleanroom.agents.deep import runtime as deep
from tests.conftest import script

pytestmark = pytest.mark.skipif(
    not deep.deepagents_available(), reason="deepagents not installed")

CONTRACT = {
    "fr_id": "1.1",
    "feature_id": "1",
    "signature": "def search(query: str) -> dict",
    "docstring": "Search for a term.",
    "mvc_layer": "controller",
    "file_path": "app/controllers/search.py",
    "error_mode": "raise",
    "example_inputs_json": '{"query": "pizza"}',
    "expected_return_json": '{"status": "ok"}',
}

IR = {"features": [{"functional_requirements": [
    {"id": "1.1", "description": "The system shall search."}]}]}


def probe(*turns):
    """``script`` padded so an extra agent turn cannot exhaust the fake model."""
    return script(*turns, *["done"] * 8)


# --- Stage 2: code generator --------------------------------------------------------
def test_code_generator_returns_a_file_with_identity_from_the_contract(fake_llm):
    """The agent authors ``content`` only — fr_id/path/layer come from the planner."""
    from src.cleanroom.agents.deep.generation import deep_generate_code

    fake_llm(probe(
        ("submit_implementation", {"fr_id": "1.1", "content": "def search(query):\n    return {}\n"}),
        "implemented"))
    files, metrics = deep_generate_code(IR, [CONTRACT])

    assert len(files) == 1
    f = files[0]
    assert f.fr_id == "1.1" and f.feature_id == "1"
    assert f.path == "app/controllers/search.py"       # from the contract, not the agent
    assert f.mvc_layer == "controller"
    assert "def search" in f.content
    assert metrics["driver"] == "deepagent" and metrics["frs_generated"] == 1


def test_code_generator_rejects_an_unknown_fr(fake_llm):
    from src.cleanroom.agents.deep.generation import deep_generate_code

    fake_llm(probe(
        ("submit_implementation", {"fr_id": "9.9", "content": "x = 1"}),
        ("submit_implementation", {"fr_id": "1.1", "content": "def search(q):\n    return {}\n"}),
        "recovered"))
    files, metrics = deep_generate_code(IR, [CONTRACT])

    assert [f.fr_id for f in files] == ["1.1"], "an invented fr_id was accepted"
    assert metrics["frs_submitted"] == 1


def test_code_generator_falls_back_to_the_virtual_file(fake_llm):
    """An agent that edits the file but never calls submit still gets its work committed."""
    from src.cleanroom.agents.deep.generation import deep_generate_code

    path = deep.virtual_path(deep.CODE_ROOT, 0, "1_1.py")
    fake_llm(probe(
        ("write_file", {"file_path": path, "content": "def search(q):\n    return {}\n"}),
        "wrote it directly"))
    files, _ = deep_generate_code(IR, [CONTRACT])

    assert len(files) == 1 and "def search" in files[0].content


def test_code_generator_skips_empty_content(fake_llm):
    from src.cleanroom.agents.deep.generation import deep_generate_code

    fake_llm(probe(("submit_implementation", {"fr_id": "1.1", "content": "   "}), "gave up"))
    files, metrics = deep_generate_code(IR, [CONTRACT])

    assert files == [] and metrics["frs_generated"] == 0


# --- Stage 2: test generator --------------------------------------------------------
def test_test_generator_collects_cases_and_source(fake_llm):
    from src.cleanroom.agents.deep.generation import deep_generate_tests

    fake_llm(probe(
        ("submit_test_case", {
            "requirement_id": "1.1", "description": "finds a match", "inputs": "pizza",
            "expected": "ok", "inputs_json": '{"query": "pizza"}',
            "expected_json": '{"status": "ok"}', "oracle": "eq"}),
        ("submit_test_source", {"test_source": "def test_search():\n    assert True\n"}),
        "done"))
    result, metrics = deep_generate_tests(IR, "1", [CONTRACT])

    assert result is not None
    assert result.feature_id == "1" and len(result.cases) == 1
    assert result.cases[0].requirement_id == "1.1"
    assert "def test_search" in result.test_source
    assert metrics["cases"] == 1 and metrics["has_source"]


def test_test_generator_refuses_source_before_cases(fake_llm):
    """The suite must be backed by recorded cases — the source alone is not auditable."""
    from src.cleanroom.agents.deep.generation import deep_generate_tests

    fake_llm(probe(
        ("submit_test_source", {"test_source": "def test_x(): pass"}),
        ("submit_test_case", {
            "requirement_id": "1.1", "description": "d", "inputs": "i", "expected": "e",
            "inputs_json": "{}"}),
        ("submit_test_source", {"test_source": "def test_y(): pass"}),
        "done"))
    result, _ = deep_generate_tests(IR, "1", [CONTRACT])

    assert "test_y" in result.test_source, "the ordered submission should have won"
    assert len(result.cases) == 1


def test_test_generator_rejects_malformed_inputs_json(fake_llm):
    from src.cleanroom.agents.deep.generation import deep_generate_tests

    fake_llm(probe(
        ("submit_test_case", {
            "requirement_id": "1.1", "description": "d", "inputs": "i", "expected": "e",
            "inputs_json": "{not json"}),
        "gave up"))
    result, metrics = deep_generate_tests(IR, "1", [CONTRACT])

    assert metrics["cases"] == 0, "a case with unparseable inputs_json was accepted"


# --- Stage 2: proof generator -------------------------------------------------------
def test_proof_generator_records_a_module(fake_llm):
    from src.cleanroom.agents.deep.generation import deep_generate_dafny

    fake_llm(probe(
        ("submit_dafny", {"module": "Feature_1", "source": "module Feature_1 { }"}),
        "proved"))
    out, metrics = deep_generate_dafny(IR, "1", [CONTRACT])

    assert out["module"] == "Feature_1" and "module Feature_1" in out["source"]
    assert metrics["has_source"] and metrics["verify_calls"] == 0


def test_proof_generator_uses_the_verifier_when_given_one(fake_llm):
    """With a verifier the agent may iterate — that is a proof checker over its own text,
    not the test oracle, so it is not an isolation break."""
    from src.cleanroom.agents.deep.generation import deep_generate_dafny

    calls: list[str] = []

    def verifier(source: str):
        calls.append(source)
        return (len(calls) > 1, "" if len(calls) > 1 else "assertion might not hold")

    fake_llm(probe(
        ("dafny_verify", {"source": "module F { }"}),
        ("dafny_verify", {"source": "module F { /* fixed */ }"}),
        ("submit_dafny", {"module": "F", "source": "module F { /* fixed */ }"}),
        "verified"))
    out, metrics = deep_generate_dafny(IR, "1", [CONTRACT], verifier=verifier)

    assert len(calls) == 2
    assert metrics["verify_calls"] == 2 and metrics["verified"] is True
    assert "fixed" in out["source"]


def test_proof_generator_survives_a_broken_verifier(fake_llm):
    """A crashing verifier must degrade to no-verification, not kill the run."""
    from src.cleanroom.agents.deep.generation import deep_generate_dafny

    def verifier(source: str):
        raise OSError("dafny not on PATH")

    fake_llm(probe(
        ("dafny_verify", {"source": "module F { }"}),
        ("submit_dafny", {"module": "F", "source": "module F { }"}),
        "submitted anyway"))
    out, metrics = deep_generate_dafny(IR, "1", [CONTRACT], verifier=verifier)

    assert out["source"], "a broken verifier lost the agent's work"
    assert metrics["verified"] is False


# --- Stage 1: planning --------------------------------------------------------------
def test_planning_accepts_a_well_formed_fr(fake_llm):
    from src.cleanroom.agents.deep.planning import deep_design_feature

    fake_llm(probe(
        ("submit_fr_plan", {
            "id": "1.1", "signature": "def search(query: str) -> dict",
            "mvc_layer": "controller", "example_inputs_json": '{"query": "pizza"}',
            "expected_return_json": '{"status": "ok"}',
            "args_json": '[{"name": "query", "description": "the search term"}]'}),
        "planned"))
    plans = deep_design_feature("Search", ["1.1"], {"1.1": "The system shall search."}, {})

    assert set(plans) == {"1.1"}
    assert plans["1.1"].mvc_layer == "controller"
    assert [a.name for a in plans["1.1"].args] == ["query"]


def test_planning_rejects_params_that_disagree_with_the_signature(fake_llm):
    """Three agents bind to this signature independently, so a name mismatch would become
    three inconsistent artifacts. The agent is told exactly what to fix."""
    from src.cleanroom.agents.deep.planning import deep_design_feature

    fake_llm(probe(
        ("submit_fr_plan", {
            "id": "1.1", "signature": "def search(query: str) -> dict",
            "mvc_layer": "controller", "example_inputs_json": '{"q": "pizza"}',
            "expected_return_json": "{}"}),
        ("submit_fr_plan", {
            "id": "1.1", "signature": "def search(query: str) -> dict",
            "mvc_layer": "controller", "example_inputs_json": '{"query": "pizza"}',
            "expected_return_json": "{}"}),
        "fixed it"))
    plans = deep_design_feature("Search", ["1.1"], {"1.1": "search"}, {})

    assert "1.1" in plans
    assert plans["1.1"].example_inputs_json == '{"query": "pizza"}'


def test_planning_rejects_an_invalid_layer(fake_llm):
    from src.cleanroom.agents.deep.planning import deep_design_feature

    fake_llm(probe(
        ("submit_fr_plan", {
            "id": "1.1", "signature": "def f() -> None", "mvc_layer": "database",
            "example_inputs_json": "{}", "expected_return_json": "null"}),
        "gave up"))
    plans = deep_design_feature("Search", ["1.1"], {"1.1": "x"}, {})

    assert plans == {}, "an invalid mvc_layer was accepted"


def test_planning_rejects_an_invented_fr_id(fake_llm):
    from src.cleanroom.agents.deep.planning import deep_design_feature

    fake_llm(probe(
        ("submit_fr_plan", {
            "id": "9.9", "signature": "def f() -> None", "mvc_layer": "model",
            "example_inputs_json": "{}", "expected_return_json": "null"}),
        "gave up"))
    plans = deep_design_feature("Search", ["1.1"], {"1.1": "x"}, {})

    assert plans == {}, "an invented FR id was accepted"


# --- wiring: the agents delegate when gen_driver="deepagent" -------------------------
def test_code_agent_delegates_to_the_deep_driver(monkeypatch):
    """``--gen-driver deepagent`` must route CodeAgent.generate through the deep driver,
    grouped per feature and preserving the planner's dependency order."""
    from src.cleanroom.agents.code import agent as code_mod

    seen: list[list[str]] = []

    def fake_deep(ir, contracts, **kw):
        seen.append([c["fr_id"] for c in contracts])
        return [], {}

    monkeypatch.setattr("src.cleanroom.agents.deep.generation.deep_generate_code", fake_deep)
    ir = {
        "features": [{"functional_requirements": [{"id": "1.1", "description": "d"}]}],
        "planning": {"contracts": [
            dict(CONTRACT), dict(CONTRACT, fr_id="2.1", feature_id="2")]},
    }
    code_mod.CodeAgent(llm=object(), gen_driver="deepagent").generate(ir)

    assert seen == [["1.1"], ["2.1"]], "expected one deep invocation per feature, in order"


def test_code_agent_deep_driver_skips_proof_backed_features(monkeypatch):
    """``skip_feature_ids`` (features whose logic ships from the Dafny tier) must still be
    honoured by the deep path, or they would be implemented twice."""
    from src.cleanroom.agents.code import agent as code_mod

    seen: list[list[str]] = []
    monkeypatch.setattr(
        "src.cleanroom.agents.deep.generation.deep_generate_code",
        lambda ir, contracts, **kw: (seen.append([c["fr_id"] for c in contracts]), ([], {}))[1])
    ir = {
        "features": [{"functional_requirements": [{"id": "1.1", "description": "d"}]}],
        "planning": {"contracts": [
            dict(CONTRACT), dict(CONTRACT, fr_id="2.1", feature_id="2")]},
    }
    code_mod.CodeAgent(llm=object(), gen_driver="deepagent").generate(
        ir, skip_feature_ids={"2"})

    assert seen == [["1.1"]], "a skipped feature was still sent to the code agent"


def test_deterministic_remains_the_default():
    """Every recorded result was produced deterministically; the deep path must be opt-in."""
    from src.cleanroom.agents.code.agent import CodeAgent
    from src.cleanroom.config import RunConfig

    assert RunConfig().gen_driver == "deterministic"
    assert RunConfig().uses_deep_generation() is False
    assert CodeAgent(llm=object()).gen_driver == "deterministic"
