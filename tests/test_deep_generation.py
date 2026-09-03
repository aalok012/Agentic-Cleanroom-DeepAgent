"""Stage 1 + Stage 2 drivers: wiring, validation feedback, and identity handling.

Offline — the model is scripted, so these assert on the drivers' contracts with the rest of
the pipeline (what they return, what they reject, where identity comes from), not on model
quality. The isolation guarantees themselves live in ``test_isolation.py``.
"""

from __future__ import annotations

import pytest

from src.cleanroom.agents.deep import runtime as deep
from tests.conftest import script

# NOT skipped when deepagents is missing. It used to be, correctly, while the deep drivers were
# an opt-in extra. They are now the ONLY generation path and `deepagents` is a hard dependency
# (pyproject), so skipping on a failed import would turn a pipeline that cannot generate anything
# at all into a green suite. Fail instead — loudly, once, with the cause named.
def test_deepagents_is_installed():
    """Guard: every other test in this file, and the whole pipeline, depends on this import."""
    assert deep.deepagents_available(), (
        "deepagents is not importable, so no stage of the pipeline can generate anything. "
        "It is a required dependency: `uv sync`.")

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


def test_code_generator_does_not_harvest_an_unsubmitted_file(fake_llm):
    """The virtual /code tree is a scratchpad. A file the agent wrote but never submitted is
    NOT collected — submit_implementation is the only channel, so the fr_id check, the
    emptiness check and the metrics cannot be routed around."""
    from src.cleanroom.agents.deep.generation import (
        DeepGenerationIncomplete,
        deep_generate_code,
    )

    path = deep.virtual_path(deep.CODE_ROOT, 0, "1_1.py")
    fake_llm(probe(
        ("write_file", {"file_path": path, "content": "def search(q):\n    return {}\n"}),
        "wrote it directly"))

    with pytest.raises(DeepGenerationIncomplete, match="1.1"):
        deep_generate_code(IR, [CONTRACT])


def test_code_generator_raises_when_an_fr_is_never_submitted(fake_llm):
    """Empty content is rejected by the tool, so the FR stays unsubmitted. That must fail
    loudly rather than return a short list a later stage discovers as a missing file."""
    from src.cleanroom.agents.deep.generation import (
        DeepGenerationIncomplete,
        deep_generate_code,
    )

    fake_llm(probe(("submit_implementation", {"fr_id": "1.1", "content": "   "}), "gave up"))

    with pytest.raises(DeepGenerationIncomplete, match="1.1"):
        deep_generate_code(IR, [CONTRACT])


def test_code_generator_lets_an_fr_be_resubmitted(fake_llm):
    """Fixing an implementation is just another call; the last submission wins."""
    from src.cleanroom.agents.deep.generation import deep_generate_code

    fake_llm(probe(
        ("submit_implementation", {"fr_id": "1.1", "content": "def search(q):\n    return 1\n"}),
        ("submit_implementation", {"fr_id": "1.1", "content": "def search(q):\n    return 2\n"}),
        "revised"))
    files, metrics = deep_generate_code(IR, [CONTRACT])

    assert len(files) == 1 and "return 2" in files[0].content
    assert metrics["frs_submitted"] == 1


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
    """The case is refused, so no case covers FR 1.1 and no source is submitted — an
    incomplete suite must fail loudly rather than be returned empty."""
    from src.cleanroom.agents.deep.generation import (
        DeepGenerationIncomplete,
        deep_generate_tests,
    )

    fake_llm(probe(
        ("submit_test_case", {
            "requirement_id": "1.1", "description": "d", "inputs": "i", "expected": "e",
            "inputs_json": "{not json"}),
        "gave up"))
    with pytest.raises(DeepGenerationIncomplete):
        deep_generate_tests(IR, "1", [CONTRACT])


# --- Stage 2: proof generator -------------------------------------------------------
def test_proof_generator_records_a_module(fake_llm):
    from src.cleanroom.agents.deep.generation import deep_generate_dafny

    src = "module Feature_1Domain refines Domain { }"
    fake_llm(probe(
        ("submit_dafny", {"module_name": "Feature_1", "source": src}),
        "proved"))
    out, metrics = deep_generate_dafny(IR, "1", [CONTRACT])

    assert out["module"] == "Feature_1" and "Feature_1Domain" in out["source"]
    assert metrics["has_source"] and metrics["verify_calls"] == 0


def test_proof_generator_rejects_a_renamed_module(fake_llm):
    """The file name, the proved-module record and the adapter's `<mod>Domain` import all key
    off this name, so a module the agent renamed must be refused, not silently overwritten."""
    from src.cleanroom.agents.deep.generation import deep_generate_dafny

    good = "module F9_9Domain refines Domain { }"
    fake_llm(probe(
        ("submit_dafny", {"module_name": "WhateverILike",
                          "source": "module WhateverILikeDomain refines Domain { }"}),
        ("submit_dafny", {"module_name": "F9_9", "source": good}),
        "renamed it"))
    out, _ = deep_generate_dafny(IR, "9.9", [CONTRACT], module="F9_9")

    assert out["module"] == "F9_9", "a renamed module was accepted"
    assert out["source"] == good


def test_proof_generator_requires_the_kernel_refinement(fake_llm):
    """A proof that verifies but does not refine the kernel is useless downstream: the
    packager compiles out/<mod>-py/ and the adapter imports <mod>Domain."""
    from src.cleanroom.agents.deep.generation import deep_generate_dafny

    fake_llm(probe(
        ("submit_dafny", {"module_name": "F1", "source": "lemma Trivial() ensures true {}"}),
        "gave up"))
    out, _ = deep_generate_dafny(IR, "1", [CONTRACT], module="F1")

    assert out["source"] == "", "a module with no <mod>Domain refinement was accepted"


def test_proof_prompt_states_the_kernel_contract():
    """This briefing used to reach the model through DafnyAgent's own prompt. That prompt was
    deleted with the deterministic driver, taking the only statement of the kernel contract
    with it — nothing else names the required members."""
    from src.cleanroom.agents.deep.generation import PROOF_PROMPT

    for required in ('include "Replay.dfy"', "refines Domain", "refines Kernel", "datatype Action",
                     "ghost predicate Inv", "function Normalize", "InitSatisfiesInv",
                     "StepPreservesInv"):
        assert required in PROOF_PROMPT, f"the proof prompt no longer states {required!r}"


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
        ("submit_dafny", {"module_name": "F",
                          "source": "module FDomain refines Domain { /* fixed */ }"}),
        "verified"))
    out, metrics = deep_generate_dafny(IR, "1", [CONTRACT], module="F", verifier=verifier)

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
        ("submit_dafny", {"module_name": "F", "source": "module FDomain refines Domain { }"}),
        "submitted anyway"))
    out, metrics = deep_generate_dafny(IR, "1", [CONTRACT], module="F", verifier=verifier)

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


# --- wiring: the agents delegate to the deep drivers ---------------------------------
def test_code_agent_delegates_to_the_deep_driver(monkeypatch):
    """CodeAgent.generate routes through the deep driver — there is no other path — grouped
    per feature and preserving the planner's dependency order."""
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
    code_mod.CodeAgent(llm=object()).generate(ir)

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
    code_mod.CodeAgent(llm=object()).generate(ir, skip_feature_ids={"2"})

    assert seen == [["1.1"]], "a skipped feature was still sent to the code agent"


def test_generation_is_deep_only():
    """The deterministic generation arm is gone: there is no switch left to select it, and the
    run report must say so, so an old `--gen-driver` invocation cannot silently mean something
    other than what it used to."""
    import argparse

    from src.cleanroom.config import RunConfig

    cfg = RunConfig()
    assert not hasattr(cfg, "gen_driver"), "the generation driver switch is still selectable"
    assert cfg.as_dict()["gen_driver"] == "deepagent"
    # A stale --gen-driver in a script must not be silently honoured as a real choice.
    assert not hasattr(RunConfig.from_args(argparse.Namespace(gen_driver="deterministic")),
                       "gen_driver")


def test_test_agent_delegates_to_the_deep_driver(monkeypatch):
    """TestAgent.generate routes through the deep driver, one invocation per feature, briefed
    with that feature's planner contracts."""
    from src.cleanroom.agents.test import agent as test_mod

    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        "src.cleanroom.agents.deep.generation.deep_generate_tests",
        lambda ir, fid, contracts, **kw: (
            seen.append((fid, [c["fr_id"] for c in contracts])), (None, {}))[1])
    ir = {
        "features": [{"id": "1", "name": "Search", "description": "d",
                      "functional_requirements": [{"id": "1.1", "description": "d"}]}],
        "planning": {"contracts": [dict(CONTRACT)]},
    }
    test_mod.TestAgent(llm=object()).generate(ir)

    assert seen == [("1", ["1.1"])]


def test_planning_agent_designs_through_the_deep_driver(monkeypatch):
    """_design_feature has no structured-output path left; it must call the planning agent."""
    from src.cleanroom.agents.planning.agent import PlanningAgent

    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        "src.cleanroom.agents.deep.planning.deep_design_feature",
        lambda name, fr_order, text_by_id, contracts_by_fr, **kw: (
            seen.append((name, list(fr_order))), {})[1])

    agent = PlanningAgent.__new__(PlanningAgent)      # skip __init__ (no LLM client wanted)
    agent.stack = "python"
    agent._design_feature("Search", ["1.1"], {"1.1": "search"}, {})

    assert seen == [("Search", ["1.1"])]


def test_empty_proof_returns_a_well_formed_feature(monkeypatch, tmp_path):
    """A proof agent that produces nothing must still return a valid FeatureDafny.

    `residual_errors` is list[dict] ({line, col, message}) — passing a bare string raised a
    pydantic ValidationError that killed the whole run instead of recording an empty proof.
    """
    from src.cleanroom.agents.dafny import agent as dafny_mod

    monkeypatch.setattr(
        "src.cleanroom.agents.deep.generation.deep_generate_dafny",
        lambda *a, **kw: ({"feature_id": "1", "module": "F1", "source": "  "},
                          {"verify_calls": 0}))

    agent = dafny_mod.DafnyAgent.__new__(dafny_mod.DafnyAgent)   # skip __init__ scaffolding
    agent.model = "test"
    agent.dafny_dir = tmp_path        # _abstract_domain() reads Replay.dfy from here
    out = agent._generate_feature_deep({}, "1", "F1", tmp_path / "F1.dfy")

    assert out.verified is False and out.dafny_source == ""
    assert out.residual_errors == [
        {"line": 0, "col": 0, "message": "the proof agent produced no Dafny source"}]


# --- prompt hygiene: gaps found reviewing the agents against deepagents usage ---------
def test_every_prompt_renders_for_every_stack():
    """Every prompt must render with every placeholder, for each stack it branches on.

    A missing placeholder is a KeyError raised mid-run, after the spend that got there — the
    prompts are formatted at agent-build time, not at import, so nothing else catches it.
    """
    from src.cleanroom.agents.deep.generation import CODE_PROMPT, PROOF_PROMPT, TEST_PROMPT
    from src.cleanroom.agents.deep.planning import PROMPT as PLANNING_PROMPT

    from src.cleanroom.agents.deep.generation import code_stack_block

    for stack in ("python", "fastapi"):
        CODE_PROMPT.format(language="Python", spec_root="/spec", code_root="/code",
                           skills_block="", max_steps=90,
                           stack_block=code_stack_block(stack))
    from src.cleanroom.agents.deep.generation import test_stack_block

    for stack in ("python", "fastapi"):
        TEST_PROMPT.format(language="Python", spec_root="/spec", test_root="/tests",
                           skills_block="", max_steps=90,
                           stack_block=test_stack_block(stack))
    PROOF_PROMPT.format(spec_root="/spec", proof_root="/proof", verify_note="",
                        skills_block="", max_steps=90, module="F1_1", domain="abstract module D")
    from src.cleanroom.agents.deep.planning import stack_block

    for stack in ("python", "fastapi", "java", "spring"):
        PLANNING_PROMPT.format(spec_root="/spec", max_steps=90, stack_block=stack_block(stack))


def test_planning_prompt_does_not_forbid_resubmission():
    """`submit_fr_plan` replaces on resubmit and every rejection says "resubmit", so a prompt
    telling the agent to call it "exactly once" would make a rejection look terminal — and a
    dropped FR falls back to a default contract."""
    from src.cleanroom.agents.deep.planning import PROMPT

    assert "exactly once" not in PROMPT
    assert "REPLACES" in PROMPT or "replaces" in PROMPT


def test_contract_sheet_carries_prerequisite_signatures():
    """The single-call generator passed prereq_ifaces into its prompt; the deep path dropped
    it, leaving an FR to guess the interface of something another feature creates."""
    from src.cleanroom.agents.deep.generation import contract_sheet

    sheet = contract_sheet(
        dict(CONTRACT, fr_id="2.1", feature_id="2"),
        "The system shall submit an order.",
        [{"fr_id": "1.1", "layer": "model",
          "signature": "def create_order(customer: str) -> dict",
          "example_inputs_json": '{"customer": "c1"}'}])

    assert "def create_order(customer: str) -> dict" in sheet
    assert "FR 1.1" in sheet


def test_all_three_generators_get_the_same_prerequisite_briefing():
    """contract_sheet is deliberately identical across Code/Test/Proof. Giving prerequisites
    to one alone would make an agreement between their artifacts partly an artifact of the
    briefing rather than evidence of independent derivation."""
    import inspect

    from src.cleanroom.agents.deep import generation

    src = inspect.getsource(generation)
    assert src.count("_seed_specs(contracts, req_text, _all_contracts(ir))") == 3


def test_entity_identifier_must_be_a_key_of_example_inputs():
    """The prompt states this rule; until now nothing enforced it, and the value flows on into
    the code agent's prompt as though the planner had vouched for it."""
    from src.cleanroom.agents.deep.planning import _param_mismatch
    from src.cleanroom.agents.planning.schema.plan import FRPlan

    def plan(entity):
        return FRPlan(id="1.1", signature="def add(name: str) -> dict", args=[], returns="",
                      mvc_layer="model", example_inputs_json='{"name": "x"}',
                      expected_return_json="{}", error_mode="raise", failure_inputs_json="",
                      entity_identifier=entity)

    sig = "def add(name: str) -> dict"
    assert "entity_identifier" in _param_mismatch(sig, plan("not_a_field"))
    assert _param_mismatch(sig, plan("name")) == ""
    assert _param_mismatch(sig, plan("")) == "", "an empty entity_identifier is legitimate"


def test_test_prompt_is_stack_aware():
    """The old template branched on stack: a FastAPI run is an HTTP app, so a failure is a 4xx
    and the suite must drive it with TestClient. The deep path passed only `language`, so the
    agent wrote direct-call ValueError tests against a web app — every one of which fails a
    correct implementation."""
    from src.cleanroom.agents.deep.generation import TEST_PROMPT, test_stack_block

    fastapi = TEST_PROMPT.format(language="Python", spec_root="/spec", test_root="/tests",
                                 skills_block="", max_steps=90,
                                 stack_block=test_stack_block("fastapi"))
    plain = TEST_PROMPT.format(language="Python", spec_root="/spec", test_root="/tests",
                               skills_block="", max_steps=90,
                               stack_block=test_stack_block("python"))

    assert '{"raises": "HTTPException"}' in fastapi
    assert "TestClient" in fastapi and "status_code >= 400" in fastapi
    assert "database starts EMPTY" in fastapi, "setup guidance for a stateful app is missing"
    assert "TestClient" not in plain and "ValueError" in plain


def test_test_agent_passes_its_stack_to_the_driver(monkeypatch):
    """TestAgent.stack is documented as shaping the emitted module and the failure oracle; it
    has to actually reach the driver for that to be true."""
    from src.cleanroom.agents.test import agent as test_mod

    seen: list[str] = []
    monkeypatch.setattr(
        "src.cleanroom.agents.deep.generation.deep_generate_tests",
        lambda ir, fid, contracts, **kw: (seen.append(kw.get("stack")), (None, {}))[1])
    ir = {
        "features": [{"id": "1", "name": "Search", "description": "d",
                      "functional_requirements": [{"id": "1.1", "description": "d"}]}],
        "planning": {"contracts": [dict(CONTRACT)]},
    }
    test_mod.TestAgent(llm=object(), stack="fastapi").generate(ir)

    assert seen == ["fastapi"], "the run's stack never reached the test agent"


def test_planning_prompt_keeps_the_design_rubric():
    """plan_feature.j2 carried ~60 lines the deep prompt never had. mvc_layer decides the file
    path and so the whole MVC layout, and it had ONE line of guidance where the template had a
    full rubric with tie-breakers and examples."""
    from src.cleanroom.agents.deep.planning import PROMPT, stack_block

    rendered = PROMPT.format(spec_root="/spec", max_steps=90, stack_block=stack_block("fastapi"))

    # The signature constraint that keeps fn(**json.loads(inputs)) working at certification.
    assert "JSON-SERIALIZABLE" in rendered and "Pydantic" in rendered
    # The layer rubric, not just the enum.
    for cue in ("owns DATA", "owns PRESENTATION", "owns BEHAVIOUR", "Tie-breakers"):
        assert cue in rendered, f"the mvc_layer rubric lost {cue!r}"
    # Practical rules that prevent downstream breakage.
    assert "DISTINCT function names" in rendered
    assert "ACTUALLY PROVIDES" in rendered, "entity_identifier lookup guidance is missing"


def test_planning_prompt_is_stack_aware():
    """A layer means something different per target; the planner has to know which."""
    from src.cleanroom.agents.deep.planning import PROMPT, stack_block

    def render(stack):
        return PROMPT.format(spec_root="/spec", max_steps=90, stack_block=stack_block(stack))

    assert "SQLAlchemy" in render("fastapi") and "APIRouter" in render("fastapi")
    assert "SQLAlchemy" not in render("python")
    assert "Java" in render("java")


def test_code_prompt_carries_the_stack_conventions():
    """generate_code.j2 had a large FastAPI structural block: APIRouter, SessionLocal,
    Body(embed=True), HTTPException instead of ValueError, and the entity-key lookup. Without
    it a FastAPI run emits plain functions the packager cannot mount."""
    from src.cleanroom.agents.deep.generation import CODE_PROMPT, code_stack_block

    def render(stack):
        return CODE_PROMPT.format(language="Python", spec_root="/spec", code_root="/code",
                                  skills_block="", max_steps=90,
                                  stack_block=code_stack_block(stack))

    fastapi = render("fastapi")
    for cue in ("APIRouter", "SessionLocal", "embed=True", "HTTPException",
                '@router.post("")', "NEVER create your own engine"):
        assert cue in fastapi, f"the FastAPI conventions lost {cue!r}"
    assert "NEVER raise a bare ValueError" in fastapi

    plain = render("python")
    assert "APIRouter" not in plain and "ValueError" in plain


def test_code_agent_passes_its_stack_to_the_driver(monkeypatch):
    from src.cleanroom.agents.code import agent as code_mod

    seen: list[str] = []
    monkeypatch.setattr(
        "src.cleanroom.agents.deep.generation.deep_generate_code",
        lambda ir, contracts, **kw: (seen.append(kw.get("stack")), ([], {}))[1])
    ir = {"features": [{"functional_requirements": [{"id": "1.1", "description": "d"}]}],
          "planning": {"contracts": [dict(CONTRACT)]}}
    code_mod.CodeAgent(llm=object(), stack="fastapi").generate(ir)

    assert seen == ["fastapi"], "the run's stack never reached the code agent"
