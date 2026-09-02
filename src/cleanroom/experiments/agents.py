"""The six pipeline stages as full-toolset deep agents.

Spec and Dependency are converted here for the first time — in the clean-room arm they are
still single ``with_structured_output`` calls. Planning, Code, Test and Proof already run as
agents there; what changes for them in this arm is the backend (real disk, shared) and the
toolset (shell ``execute`` added), not the prompt.

Prompts are imported verbatim from the clean-room drivers wherever one exists, so an A/B
between the arms varies the toolset and backend only. Read the isolation warning at the top of
``full_toolset.py`` before drawing any conclusion from what this produces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import ValidationError

from src.cleanroom.agents.code.schema.code import GeneratedFile
from src.cleanroom.agents.deep.generation import (
    CODE_PROMPT,
    PROOF_PROMPT,
    TEST_PROMPT,
    contract_sheet,
)
from src.cleanroom.agents.deep.planning import PROMPT as PLANNING_PROMPT
from src.cleanroom.agents.deep.planning import _spec_sheet
from src.cleanroom.agents.deep.runtime import load_skills
from src.cleanroom.agents.planning.schema.plan import ArgDoc, FRPlan
from src.cleanroom.agents.spec_agent.schema.ir import FRContract
from src.cleanroom.agents.test.schema.tests import FeatureTests, TestCase
from src.cleanroom.experiments.full_toolset import (
    CODE_DIR,
    PROOF_DIR,
    SKILL_DIR,
    SPEC_DIR,
    TEST_DIR,
    RunLog,
    build_full_agent,
    full_toolset_max_steps,
    invoke_full,
    read_disk,
    seed_disk,
)
from src.cleanroom.utils.contracts import requirement_text

# The clean-room prompts describe a virtual filesystem ("/spec/", "/code/"). Here the paths are
# real and relative to the run root, so the roots are substituted at format time. Everything
# else in the prompt text is untouched.
_SHELL_NOTE = """
You also have `execute`, a real shell in this directory. Use it when it genuinely helps —
checking that a file you wrote imports, listing what exists, running a quick sanity command.
Do not use it to inspect another agent's work.
"""


def _skills_block(skills: dict[str, str]) -> str:
    if not skills:
        return ""
    return (f"* {SKILL_DIR}/ — authored guidance, read what is relevant:\n"
            + "\n".join(f"  * `{p}`" for p in sorted(skills)))


def _seed_skills(root: Path, names: list[str]) -> dict[str, str]:
    """Copy the authored skill docs onto the real disk under ``skills/``."""
    loaded = load_skills(names)                       # {/skills/<name>.md: content}
    on_disk = {f"{SKILL_DIR}/{Path(p).name}": c for p, c in loaded.items()}
    seed_disk(root, on_disk)
    return on_disk


def _requirement_index(ir: dict) -> dict[str, str]:
    index: dict[str, str] = {}
    for feature in ir.get("features", []) or []:
        for req in feature.get("functional_requirements", []) or []:
            index[req["id"]] = requirement_text(req)
    return index


def _brief(exc: ValidationError) -> str:
    return "\n".join(
        f"- {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8])


# ======================================================================================
# Stage 1 — Spec / contract extraction   (NEW: single-shot in the clean-room arm)
# ======================================================================================

SPEC_AGENT_PROMPT = """\
You are the specification agent. For EACH functional requirement of one feature you write a
BEHAVIORAL CONTRACT: a design-by-contract statement of what the operation must do.

Your filesystem:
* {spec_dir}/ — one read-only sheet per FR: its requirement text. READ THESE FIRST.
{shell_note}
A contract is a specification, NOT code: describe WHAT must hold, not how to implement it.
Keep it stack-agnostic (no function names, no frameworks). Ground every field strictly in the
requirement's own text; never invent behaviour it does not state.

Submit each one with `submit_contract(...)`:
* id             — the FR id, copied VERBATIM from its sheet. Never invent one.
* stimulus       — the input event/data/action that triggers the operation.
* precondition   — the boolean condition on the INPUTS that must hold first ("none" if any
                   input is acceptable). Constrains the stimulus only, never stored state.
* response       — what the system does in reaction.
* postcondition  — what is guaranteed true after it completes.

`submit_contract` is the ONLY way to deliver a contract; a file you write is not collected.
Resubmitting an id replaces its earlier contract. You are done when every FR is accepted.
Start with `write_todos` listing the FR ids, then read every sheet.
"""


def spec_contracts(root: Path, feature_name: str, feature_id: str, frs: list[dict],
                   log: RunLog, *, model: str | None = None) -> dict[str, FRContract]:
    """Behavioral contracts for one feature's FRs. Returns ``{fr_id: FRContract}``."""
    if not frs:
        return {}

    sheets = {
        f"{SPEC_DIR}/{str(r['id']).replace('.', '_')}.md":
            f"# FR {r['id']}\n\n## Requirement\n{requirement_text(r)}\n"
        for r in frs
    }
    seed_disk(root, sheets)

    known = {str(r["id"]).strip() for r in frs}
    accepted: dict[str, FRContract] = {}

    @tool
    def submit_contract(
        id: Annotated[str, "The FR id, copied verbatim from its sheet."],
        stimulus: Annotated[str, "The triggering input event, data or action."],
        precondition: Annotated[str, 'Boolean condition on the inputs, or "none".'],
        response: Annotated[str, "What the system does in reaction."],
        postcondition: Annotated[str, "What is guaranteed true afterwards."],
    ) -> str:
        """Record the behavioral contract for ONE functional requirement."""
        key = (id or "").strip().strip("[]").strip()
        if key not in known:
            return f"Unknown FR id {id!r}. Contract only: {', '.join(sorted(known))}."
        try:
            contract = FRContract(id=key, stimulus=stimulus, precondition=precondition,
                                  response=response, postcondition=postcondition)
        except ValidationError as exc:
            return f"Rejected — fix these fields and resubmit FR {id}:\n{_brief(exc)}"
        accepted[key] = contract
        remaining = sorted(known - set(accepted))
        return (f"Accepted FR {key}. "
                + (f"Still to contract: {', '.join(remaining)}." if remaining
                   else "All FRs contracted — you are done."))

    agent = build_full_agent(
        [submit_contract],
        SPEC_AGENT_PROMPT.format(spec_dir=SPEC_DIR, shell_note=_SHELL_NOTE),
        _backend_for(root), name="spec", model=model)
    listing = "\n".join(f"- FR {r['id']}" for r in frs)
    steps = full_toolset_max_steps() + 6 * len(frs)
    state, secs = invoke_full(
        agent,
        f"Feature: {feature_name} (id {feature_id})\n\nWrite the behavioral contract for these "
        f"{len(frs)} functional requirement(s):\n{listing}\n\nTheir sheets are in {SPEC_DIR}/.",
        max_steps=steps)
    log.record("spec", str(feature_id), state, secs, steps)
    return accepted


# ======================================================================================
# Stage 2 — Dependency inference   (NEW: single-shot in the clean-room arm)
# ======================================================================================

DEPENDENCY_PROMPT = """\
You are the dependency agent. Within ONE feature you decide which functional requirements
must run BEFORE which others — a real data/state prerequisite, not a narrative ordering.

Your filesystem:
* {spec_dir}/ — one read-only sheet per FR. READ THESE FIRST.
{shell_note}
An edge FR-A -> FR-B means B must have happened for A to be meaningful (A consumes an entity
B creates, A updates state B established). Do NOT add an edge merely because two FRs mention
the same noun, or because one is written after the other.

Call `submit_dependency(id=..., prerequisite_ids=...)` once per FR that HAS prerequisites,
with `prerequisite_ids` as a JSON list of FR ids from this feature. FRs with none need no
call. Resubmitting an id replaces its earlier edge set. Call `no_more_dependencies()` when
finished — including when there are none at all, which is a perfectly good answer.
Start with `write_todos`, then read every sheet.
"""


def dependency_edges(root: Path, feature_name: str, feature_id: str, frs: list[dict],
                     log: RunLog, *, model: str | None = None) -> list[tuple[str, str]]:
    """Inferred within-feature FR edges as ``[(source, prerequisite), ...]``."""
    if len(frs) < 2:
        return []

    sheets = {
        f"{SPEC_DIR}/{str(r['id']).replace('.', '_')}.md":
            f"# FR {r['id']}\n\n## Requirement\n{requirement_text(r)}\n"
        for r in frs
    }
    seed_disk(root, sheets)

    within = {str(r["id"]).strip() for r in frs}
    edges: dict[str, list[str]] = {}
    done: dict[str, bool] = {}

    @tool
    def submit_dependency(
        id: Annotated[str, "The FR that has prerequisites."],
        prerequisite_ids: Annotated[str, 'JSON list of FR ids, e.g. ["1.1","1.2"].'],
    ) -> str:
        """Record which FRs must run before ONE functional requirement."""
        source = (id or "").strip().strip("[]").strip()
        if source not in within:
            return f"Unknown FR id {id!r}. Use only: {', '.join(sorted(within))}."
        try:
            targets = json.loads(prerequisite_ids or "[]")
        except ValueError as exc:
            return f"prerequisite_ids is not valid JSON ({exc}). Resubmit as a JSON list."
        if not isinstance(targets, list):
            return "prerequisite_ids must be a JSON LIST of ids."
        kept = [t for t in ((str(x).strip().strip("[]").strip()) for x in targets)
                if t in within and t != source]
        dropped = len(targets) - len(kept)
        edges[source] = kept
        return (f"Recorded {len(kept)} prerequisite(s) for FR {source}."
                + (f" Dropped {dropped} that were not FRs of this feature." if dropped else ""))

    @tool
    def no_more_dependencies() -> str:
        """Declare the dependency analysis for this feature complete."""
        done["yes"] = True
        return f"Finished: {sum(len(v) for v in edges.values())} edge(s) recorded."

    agent = build_full_agent(
        [submit_dependency, no_more_dependencies],
        DEPENDENCY_PROMPT.format(spec_dir=SPEC_DIR, shell_note=_SHELL_NOTE),
        _backend_for(root), name="dependency", model=model)
    listing = "\n".join(f"- FR {r['id']}" for r in frs)
    steps = full_toolset_max_steps() + 4 * len(frs)
    state, secs = invoke_full(
        agent,
        f"Feature: {feature_name} (id {feature_id})\n\nInfer the prerequisite edges among "
        f"these {len(frs)} functional requirement(s):\n{listing}\n\n"
        f"Their sheets are in {SPEC_DIR}/.",
        max_steps=steps)
    log.record("dependency", str(feature_id), state, secs, steps)

    return [(src, tgt) for src, targets in edges.items() for tgt in targets]


# ======================================================================================
# Stages 3-6 — Planning / Code / Test / Proof, same prompts, full toolset
# ======================================================================================

def plan_feature(root: Path, feature_name: str, fr_order: list[str],
                 text_by_id: dict[str, str], contracts_by_fr: dict, log: RunLog,
                 *, stack: str = "python", model: str | None = None) -> dict[str, FRPlan]:
    """Interface design for one feature. Mirrors ``deep_design_feature`` with a real disk."""
    from src.cleanroom.agents.planning.agent import _norm_id

    if not fr_order:
        return {}

    seed_disk(root, {
        f"{SPEC_DIR}/{str(rid).replace('.', '_')}.md":
            _spec_sheet(rid, text_by_id.get(rid, ""), contracts_by_fr.get(rid), stack)
        for rid in fr_order})

    known = {_norm_id(r) for r in fr_order}
    accepted: dict[str, FRPlan] = {}

    @tool
    def submit_fr_plan(
        id: Annotated[str, "The FR id, copied verbatim from its spec sheet."],
        signature: Annotated[str, "One-line function signature in Python syntax."],
        mvc_layer: Annotated[str, "Exactly one of: model, view, controller."],
        example_inputs_json: Annotated[str, 'JSON object of happy-path kwargs.'],
        expected_return_json: Annotated[str, "JSON value returned on the happy path."],
        args_json: Annotated[str, 'JSON list of {"name","description"}.'] = "[]",
        returns: Annotated[str, "What the function returns; empty for None."] = "",
        error_mode: Annotated[str, '"raise" or "return".'] = "raise",
        failure_inputs_json: Annotated[str, "JSON kwargs that must fail; empty if none."] = "",
        entity_identifier: Annotated[str, "Unique key field for a persisted entity."] = "",
    ) -> str:
        """Record the finished interface design for ONE functional requirement."""
        norm = _norm_id(id)
        if norm not in known:
            return f"Unknown FR id {id!r}. Plan only: {', '.join(sorted(known))}."
        try:
            args = [ArgDoc(**a) for a in json.loads(args_json or "[]")]
        except (ValueError, TypeError) as exc:
            return f"args_json is not a JSON list of name/description objects: {exc}."
        try:
            plan = FRPlan(
                id=id, signature=signature, args=args, returns=returns, mvc_layer=mvc_layer,
                example_inputs_json=example_inputs_json,
                expected_return_json=expected_return_json, error_mode=error_mode,
                failure_inputs_json=failure_inputs_json, entity_identifier=entity_identifier)
        except ValidationError as exc:
            return f"Rejected — fix these fields and resubmit FR {id}:\n{_brief(exc)}"
        accepted[norm] = plan
        remaining = sorted(known - set(accepted))
        return (f"Accepted FR {id}. "
                + (f"Still to plan: {', '.join(remaining)}." if remaining
                   else "All FRs planned — you are done."))

    agent = build_full_agent(
        [submit_fr_plan],
        PLANNING_PROMPT.format(spec_root=SPEC_DIR) + _SHELL_NOTE,
        _backend_for(root), name="planning", model=model)
    steps = full_toolset_max_steps() + 8 * len(fr_order)
    state, secs = invoke_full(
        agent,
        f"Feature: {feature_name}\nTarget stack: {stack}\n\nDesign the interface for these "
        f"{len(fr_order)} functional requirement(s):\n"
        + "\n".join(f"- FR {rid}" for rid in fr_order)
        + f"\n\nTheir spec sheets are in {SPEC_DIR}/.",
        max_steps=steps)
    log.record("planning", feature_name, state, secs, steps)
    return accepted


def generate_code(root: Path, ir: dict, contracts: list[dict], log: RunLog,
                  *, language: str = "Python", model: str | None = None) -> list[GeneratedFile]:
    """Implementation for one feature's FRs. Same prompt as the clean-room code agent."""
    if not contracts:
        return []

    req_text = _requirement_index(ir)
    seed_disk(root, {
        f"{SPEC_DIR}/{str(c['fr_id']).replace('.', '_')}.md":
            contract_sheet(c, req_text.get(c["fr_id"], ""))
        for c in contracts})
    skills = _seed_skills(root, ["clean-room-implementation"])

    known = {c["fr_id"] for c in contracts}
    submitted: dict[str, str] = {}

    @tool
    def submit_implementation(
        fr_id: Annotated[str, "The functional requirement id, e.g. '1.1'."],
        content: Annotated[str, "The COMPLETE source of that FR's file."],
    ) -> str:
        """Record the finished implementation for ONE functional requirement."""
        if fr_id not in known:
            return f"Unknown fr_id {fr_id!r}. Implement only: {', '.join(sorted(known))}."
        if not (content or "").strip():
            return "Rejected: content was empty. Submit the complete source of the file."
        replaced = fr_id in submitted
        submitted[fr_id] = content
        remaining = sorted(known - set(submitted))
        return (f"{'Replaced' if replaced else 'Recorded'} FR {fr_id} "
                f"({len(content.splitlines())} lines). "
                + (f"Still to implement: {', '.join(remaining)}." if remaining
                   else "All FRs implemented — you are done."))

    agent = build_full_agent(
        [submit_implementation],
        CODE_PROMPT.format(language=language, spec_root=SPEC_DIR, code_root=CODE_DIR,
                           skills_block=_skills_block(skills)) + _SHELL_NOTE,
        _backend_for(root), name="code", model=model)
    steps = full_toolset_max_steps() + 10 * len(contracts)
    state, secs = invoke_full(
        agent,
        f"Implement these {len(contracts)} functional requirement(s):\n"
        + "\n".join(f"- FR {c['fr_id']} -> {CODE_DIR}/"
                    f"{str(c['fr_id']).replace('.', '_')}.py" for c in contracts)
        + f"\n\nTheir contract sheets are in {SPEC_DIR}/.",
        max_steps=steps)
    log.record("code", contracts[0]["feature_id"], state, secs, steps)

    # Unlike the clean-room arm this DOES fall back to the file on disk: the point of the pass
    # is to observe what the agent does with write_file, so work it left there is kept rather
    # than discarded. Which channel each file came from is recorded for the report.
    files: list[GeneratedFile] = []
    for c in contracts:
        fr_id = c["fr_id"]
        content = submitted.get(fr_id) or read_disk(
            root, f"{CODE_DIR}/{str(fr_id).replace('.', '_')}.py")
        if not content.strip():
            continue
        files.append(GeneratedFile(
            fr_id=fr_id, feature_id=c["feature_id"], path=c["file_path"],
            mvc_layer=c["mvc_layer"], content=content))
    log.rows[-1]["submitted_via_tool"] = len(submitted)
    log.rows[-1]["recovered_from_disk"] = len(files) - len(submitted)
    return files


def generate_tests(root: Path, ir: dict, feature_id: str, contracts: list[dict], log: RunLog,
                   *, language: str = "Python", model: str | None = None) -> FeatureTests | None:
    """Black-box suite for one feature. NOTE: in this arm the code is on the same disk."""
    if not contracts:
        return None

    req_text = _requirement_index(ir)
    seed_disk(root, {
        f"{SPEC_DIR}/{str(c['fr_id']).replace('.', '_')}.md":
            contract_sheet(c, req_text.get(c["fr_id"], ""))
        for c in contracts})
    skills = _seed_skills(root, ["blackbox-testing"])

    known = {c["fr_id"] for c in contracts}
    cases: list[TestCase] = []
    source: dict[str, str] = {}

    @tool
    def submit_test_case(
        requirement_id: Annotated[str, "The FR this case verifies."],
        description: Annotated[str, "What behaviour is being checked."],
        inputs: Annotated[str, "Human-readable summary of the inputs."],
        expected: Annotated[str, "Human-readable summary of the expected result."],
        inputs_json: Annotated[str, "JSON object of kwargs passed to the function."],
        expected_json: Annotated[str, "JSON value expected back."] = "",
        oracle: Annotated[str, '"eq" or "raises".'] = "eq",
        setup_json: Annotated[str, "JSON list of prior calls; empty if none."] = "",
    ) -> str:
        """Record ONE black-box test case derived from a contract."""
        if requirement_id not in known:
            return f"Unknown requirement_id {requirement_id!r}. Test only: {', '.join(sorted(known))}."
        try:
            case = TestCase(requirement_id=requirement_id, description=description,
                            inputs=inputs, expected=expected, inputs_json=inputs_json,
                            expected_json=expected_json, oracle=oracle, setup_json=setup_json)
        except ValidationError as exc:
            return f"Rejected — fix these fields and resubmit:\n{_brief(exc)}"
        try:
            json.loads(inputs_json or "{}")
        except ValueError as exc:
            return f"inputs_json is not valid JSON ({exc}). Resubmit this case."
        cases.append(case)
        return f"Recorded case {len(cases)} for FR {requirement_id}."

    @tool
    def submit_test_source(
        test_source: Annotated[str, "The COMPLETE runnable test file."],
    ) -> str:
        """Record the finished test file, once every case is recorded."""
        if not (test_source or "").strip():
            return "Rejected: test_source was empty."
        if not cases:
            return "Rejected: record the individual cases with submit_test_case first."
        source["src"] = test_source
        return f"Recorded the test source ({len(test_source.splitlines())} lines)."

    agent = build_full_agent(
        [submit_test_case, submit_test_source],
        TEST_PROMPT.format(language=language, spec_root=SPEC_DIR, test_root=TEST_DIR,
                           skills_block=_skills_block(skills)) + _SHELL_NOTE,
        _backend_for(root), name="test", model=model)
    steps = full_toolset_max_steps() + 10 * len(contracts)
    state, secs = invoke_full(
        agent,
        f"Write black-box tests for feature {feature_id}, covering these {len(contracts)} "
        f"functional requirement(s):\n"
        + "\n".join(f"- FR {c['fr_id']}: {c.get('signature', '')}" for c in contracts)
        + f"\n\nTheir contract sheets are in {SPEC_DIR}/. Write the file to "
          f"{TEST_DIR}/test_{str(feature_id).replace('.', '_')}.py.",
        max_steps=steps)
    log.record("test", str(feature_id), state, secs, steps)

    test_source = source.get("src") or read_disk(
        root, f"{TEST_DIR}/test_{str(feature_id).replace('.', '_')}.py")
    return FeatureTests(feature_id=str(feature_id), cases=cases, test_source=test_source)


def generate_dafny(root: Path, ir: dict, feature_id: str, contracts: list[dict], log: RunLog,
                   *, verifier=None, model: str | None = None) -> dict:
    """Dafny module for one feature, with the verifier tool when one is available."""
    if not contracts:
        return {}

    req_text = _requirement_index(ir)
    seed_disk(root, {
        f"{SPEC_DIR}/{str(c['fr_id']).replace('.', '_')}.md":
            contract_sheet(c, req_text.get(c["fr_id"], ""))
        for c in contracts})
    skills = _seed_skills(root, ["dafny-proofs", "dafny-patterns"])

    module = f"Feature_{str(feature_id).replace('.', '_')}"
    submitted: dict[str, str] = {}
    verifications: list[bool] = []

    @tool
    def submit_dafny(
        module: Annotated[str, "The Dafny module name."],
        source: Annotated[str, "The COMPLETE Dafny module source."],
    ) -> str:
        """Record the finished Dafny module for this feature."""
        if not (source or "").strip():
            return "Rejected: source was empty."
        submitted["module"] = module
        submitted["source"] = source
        return f"Recorded module {module} ({len(source.splitlines())} lines)."

    tools = [submit_dafny]
    verify_note = ""
    if verifier is not None:
        verify_note = ("* Call `dafny_verify(source=...)` to check a draft. Iterate until it "
                       "verifies or you run out of budget; report honestly if it does not.\n")

        @tool
        def dafny_verify(source: Annotated[str, "The Dafny source to verify."]) -> str:
            """Run the Dafny verifier over a draft and return its output."""
            if not (source or "").strip():
                return "Nothing to verify: source was empty."
            try:
                ok, output = verifier(source)
            except Exception as exc:
                return f"The verifier could not be run ({exc}). Continue without it."
            verifications.append(bool(ok))
            return ("Verification SUCCEEDED.\n" if ok else "Verification FAILED.\n") + (output or "")

        tools.append(dafny_verify)

    agent = build_full_agent(
        tools,
        PROOF_PROMPT.format(spec_root=SPEC_DIR, proof_root=PROOF_DIR,
                            verify_note=verify_note,
                            skills_block=_skills_block(skills)) + _SHELL_NOTE,
        _backend_for(root), name="proof", model=model)
    steps = full_toolset_max_steps() + 12 * len(contracts)
    state, secs = invoke_full(
        agent,
        f"Specify and prove feature {feature_id} in Dafny module `{module}`, covering these "
        f"{len(contracts)} functional requirement(s):\n"
        + "\n".join(f"- FR {c['fr_id']}: {c.get('signature', '')}" for c in contracts)
        + f"\n\nTheir contract sheets are in {SPEC_DIR}/.",
        max_steps=steps)
    row = log.record("proof", str(feature_id), state, secs, steps)
    row["verify_calls"] = len(verifications)
    row["verified"] = bool(verifications and verifications[-1])

    source = submitted.get("source") or read_disk(root, f"{PROOF_DIR}/{module}.dfy")
    return {"feature_id": str(feature_id),
            "module": submitted.get("module") or module,
            "source": source}


# --------------------------------------------------------------------------------------
# One backend per run root, reused across every agent — that sharing IS this arm's premise.
# --------------------------------------------------------------------------------------
_BACKENDS: dict[Path, object] = {}


def _backend_for(root: Path):
    from src.cleanroom.experiments.full_toolset import build_backend

    key = Path(root).resolve()
    if key not in _BACKENDS:
        _BACKENDS[key] = build_backend(key)
    return _BACKENDS[key]
