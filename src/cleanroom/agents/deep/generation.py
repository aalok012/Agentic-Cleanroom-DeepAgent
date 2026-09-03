"""Deepagents drivers for the clean-room generators.

Code, Test, Proof and the Frontend each derive their artifact from the planner's contract ALONE. This
module is where the paper's central claim is implemented, so read the isolation note before
changing anything here.

==========================  ISOLATION  ==========================
Each generator is a SEPARATE top-level ``create_deep_agent`` invocation with its own
``StateBackend``, seeded with exactly one pool:

    deep_generate_code     seeds /spec (contracts) + /code   — never /tests, never /proof
    deep_generate_tests    seeds /spec (contracts) + /tests  — never /code,  never /proof
    deep_generate_dafny    seeds /spec (contracts) + /proof  — never /code,  never /tests
    deep_generate_frontend seeds /spec (contracts) + /ui     — never /code,  never /tests

The frontend is an UNSCORED deliverable — no oracle tests a UI, so it is not part of pass@k
or the verification ratio. It is still held to the same isolation rule: it calls the backend
over HTTP, and the routes it needs are derived from the contract's ``file_path`` by the same
``route_for`` the packager and the cert oracle use, so it never needs to read the generated
code to know where to POST.

``/spec`` is the ONE shared input: the planner's contract is deliberately common read-only
ground, which is what makes three independent derivations target the same interface. Nothing
derived by one generator is ever seeded into another.

This is enforced structurally, not by prompt. Under ``StateBackend`` there is no disk beneath
the agent and the foreign files simply do not exist in its filesystem, so ``ls``/``glob``/
``grep``/``read_file`` cannot surface one however hard the model tries. The prompts state the
restriction as well, but only so the model does not waste turns looking.

Three things that would silently break it, all avoided here:
  * a ``subagents=`` relationship — a subagent inherits the parent's backend, merging pools;
  * a ``FilesystemBackend`` — exposes the real repository, where all three pools coexist;
  * a ``StoreBackend`` — persists across threads, so a later run could read an earlier one's.

``tests/test_isolation.py`` asserts all of this, including static guards against the three
regressions above. See ``deep/README.md``.
=================================================================

=====================  TOOL CALLING IS THE CONTRACT  =====================
An artifact counts ONLY when it arrives through this module's validated ``submit_*`` tool.
The agent's virtual filesystem is a scratchpad: a file it happens to leave at some path is
NOT harvested as output, and a free-text reply is never scraped for a code fence.

That is deliberate. The submit tools are where the fr_id is checked against the requested
set, emptiness is rejected and pydantic validation errors are handed back for the agent to
fix. Accepting an unsubmitted file would route around all of it, and would let a run whose
tool calling silently failed still report artifacts — making ``frs_submitted`` a number that
cannot be trusted. A requested artifact that never gets submitted raises
``DeepGenerationIncomplete`` instead, naming what is missing.
==========================================================================
"""

from __future__ import annotations

import json
import re
from typing import Annotated

from langchain_core.tools import tool
from pydantic import ValidationError

from src.cleanroom.agents.code.schema.code import GeneratedFile
from src.cleanroom.agents.deep.runtime import (
    CODE_ROOT,
    PROOF_ROOT,
    SPEC_ROOT,
    TEST_ROOT,
    UI_ROOT,
    agent_step_count,
    build_agent,
    deep_max_steps,
    invoke_agent,
    nudge_agent,
    load_skills,
    seed_files,
    skill_index,
    virtual_path,
)
from src.cleanroom.agents.test.schema.tests import FeatureTests, TestCase
from src.cleanroom.utils.contracts import prereq_ifaces, requirement_text, route_for


class DeepGenerationIncomplete(RuntimeError):
    """An agent finished its step budget without submitting every artifact it was asked for.

    Raised rather than returning a partial result: a silently short generation shows up much
    later as an unexplained missing file, whereas this names the stage and the missing ids.
    """

# --------------------------------------------------------------------------------------
# Shared: the contract sheet every generator reads, and nothing else.
# --------------------------------------------------------------------------------------

def contract_sheet(contract: dict, requirement: str,
                   prerequisites: list[dict] | None = None) -> str:
    """The read-only briefing for one FR — identical for all three generators.

    Deliberately identical: if Code, Test and Proof were briefed differently, a disagreement
    between their artifacts would be an artifact of the briefing rather than evidence about
    independent derivation. So ``prerequisites`` is added to ALL THREE or none — never to the
    code agent alone, however tempting, because that would make an agreement between code and
    tests partly an artifact of code having been told more.

    ``prerequisites`` carries the SIGNATURES of the FRs this one depends on — never their
    bodies, and taken from the planner's contracts rather than any generated artifact, so the
    input stays purely spec-derived. Without it an FR that calls something another feature
    creates has only the planner's docstring note ("Prerequisite: create_order (req 1.1)") and
    must guess that function's parameters and return shape.
    """
    lines = [
        f"# FR {contract.get('fr_id')} (feature {contract.get('feature_id')})",
        "",
        "## Requirement",
        requirement or "(not recorded)",
        "",
        "## Interface — implement/verify EXACTLY this",
        "```python",
        contract.get("signature", ""),
        '"""' + (contract.get("docstring", "") or "") + '"""',
        "```",
        "",
        f"- layer: {contract.get('mvc_layer', '')}",
        f"- error mode: {contract.get('error_mode', 'raise')}",
    ]
    if prerequisites:
        lines += ["", "## Prerequisites — these run BEFORE this FR",
                  "Their signatures only. Call them as written; do not reimplement them.", ""]
        for p in prerequisites:
            lines.append(f"- FR {p['fr_id']} ({p.get('layer', '')}): `{p.get('signature', '')}`")
            if p.get("example_inputs_json"):
                lines.append(f"    example inputs: `{p['example_inputs_json']}`")
    for label, key in (("Example inputs", "example_inputs_json"),
                       ("Expected return", "expected_return_json"),
                       ("Failure inputs (must fail)", "failure_inputs_json")):
        if contract.get(key):
            lines += ["", f"## {label}", "```json", contract[key], "```"]
    return "\n".join(lines)


def _requirement_index(ir: dict) -> dict[str, str]:
    index: dict[str, str] = {}
    for feature in ir.get("features", []) or []:
        for req in feature.get("functional_requirements", []) or []:
            index[req["id"]] = requirement_text(req)
    return index


def _seed_specs(contracts: list[dict], req_text: dict[str, str],
                by_fr: dict[str, dict] | None = None) -> dict[str, str]:
    """``/spec`` sheets for a set of contracts — the shared read-only input.

    ``by_fr`` indexes EVERY contract in the run, not just the ones being seeded: a
    prerequisite often lives in another feature, and that is exactly the case the sheet needs
    to cover. Only the prerequisite's signature crosses over, never its implementation.
    """
    by_fr = by_fr or {}
    return {
        virtual_path(SPEC_ROOT, i, f"{str(c.get('fr_id')).replace('.', '_')}.md"):
            contract_sheet(c, req_text.get(c.get("fr_id"), ""),
                           prereq_ifaces(c, by_fr) if by_fr else None)
        for i, c in enumerate(contracts)
    }


def _all_contracts(ir: dict) -> dict[str, dict]:
    """``{fr_id: contract}`` over the WHOLE run, for prerequisite lookup."""
    return {c["fr_id"]: c for c in ((ir.get("planning") or {}).get("contracts") or [])
            if c.get("fr_id")}


def _skills_block(skills: dict[str, str]) -> str:
    """The `/skills` section of a system prompt — a catalogue, not the documents themselves.

    Progressive disclosure: the agent sees one line per document and reads the body only when
    it decides it needs it. On a 32k context that difference is the whole point.
    """
    if not skills:
        return ""
    return ("* /skills/ — authored guidance, read the ones relevant to your task:\n"
            + "\n".join("  " + ln for ln in skill_index(skills).splitlines()))


def _brief(exc: ValidationError) -> str:
    return "\n".join(
        f"- {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8])


# --------------------------------------------------------------------------------------
# Code Agent — sees contracts. Never tests.
# --------------------------------------------------------------------------------------

CODE_PROMPT = """\
You are the implementation agent in a CLEAN-ROOM pipeline. You write {language} source that
satisfies each functional requirement's contract.

Your filesystem:
* {spec_root}/ — one read-only contract sheet per FR. READ THESE FIRST.
* {code_root}/ — where your implementation goes, one file per FR (paths are listed for you).
{skills_block}

You will NEVER be given the test suite. It is written independently from the same contracts
by another agent that never sees your code, and it does not exist in your filesystem — do not
look for it. Implement the requirement as specified, in general, not against guessed cases.

Rules:
* The signature is a contract. Do not rename the function, reorder or rename parameters, or
  change the return shape — the tests and the proofs bind to it exactly as written.
* Each file must be self-contained and importable: every import it needs, no prose outside code.
* Honour the error mode: "raise" means raise ValueError on a precondition violation; "return"
  means return an error dict.
* FRs in one feature share data shapes. Use `read_file` across {code_root}/ before you invent
  a shape, so two files do not disagree.
* Prerequisites listed on a sheet are build-order dependencies. Do NOT call an update/setup
  prerequisite from a read/query function. If you must call one for its value, import it from
  its file_path and invoke it with the canonical inputs shown.

{stack_block}

Submit each file with `submit_implementation(fr_id=..., content=...)` giving the FULL source.
This tool call is the ONLY way to deliver an implementation. Drafting into {code_root}/ with
`write_file` is encouraged — it is how you cross-read your own files for consistent data shapes
— but a file left there is NOT collected: nothing you have not submitted counts, and you are not
finished until `submit_implementation` has accepted every FR. (`write_file` cannot overwrite an
existing file — use `edit_file` to revise a draft.) Resubmitting an FR replaces its earlier
submission, so fixing one is just another call.
Start with `write_todos` listing the FRs, then read every contract sheet before writing.

You have about {max_steps} tool calls for this whole feature. Reading every sheet first is
worth it; re-reading one you have already read is not. If the budget runs short, submit what
you have — an unsubmitted FR is a failed one, and a rough implementation beats nothing.
"""


_SUBMIT_RETRIES = 2


def _missing_nudge(missing: list[str], tool_name: str) -> str:
    """The follow-up turn for an agent that finished without delivering everything."""
    return (f"You have not delivered {len(missing)} item(s): {', '.join(missing)}. "
            f"Describing the work is not delivering it — only a `{tool_name}` tool call counts, "
            f"and nothing you wrote to the filesystem is collected. "
            f"Call `{tool_name}` now for each one listed, with its complete content.")


def deep_generate_code(
    ir: dict,
    contracts: list[dict],
    *,
    language: str = "Python",
    stack: str = "python",
    temperature: float = 0.0,
    model: str | None = None,
    max_steps: int | None = None,
) -> tuple[list[GeneratedFile], dict]:
    """Agent-driven counterpart to ``CodeAgent.generate``.

    Returns ``(files, metrics)``. Identity fields come from the contract, never the agent —
    only ``content`` is authored, exactly as the deterministic generator does.
    """
    if not contracts:
        return [], {"skipped": "no contracts"}

    req_text = _requirement_index(ir)
    seeds = _seed_specs(contracts, req_text, _all_contracts(ir))          # /spec — shared read-only input
    skills = load_skills(["clean-room-implementation"])
    seeds.update(skills)                              # /skills — authored guidance
    path_by_fr: dict[str, str] = {}
    for i, c in enumerate(contracts):
        fr_id = c["fr_id"]
        path = virtual_path(CODE_ROOT, i, f"{str(fr_id).replace('.', '_')}.py")
        path_by_fr[fr_id] = path
    # The /code paths are deliberately NOT pre-seeded: `write_file` refuses to overwrite an
    # existing file, so an empty placeholder would block the agent from using it. The paths
    # are given in the opening message instead.
    #
    # No TEST_ROOT and no PROOF_ROOT entry is ever added here. That absence IS the isolation
    # guarantee; tests/test_isolation.py asserts it.

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

    # The budget is stated IN the prompt, so it has to exist before the prompt is built.
    steps = max_steps or (deep_max_steps() + 10 * len(contracts))
    agent = build_agent(
        [submit_implementation],
        CODE_PROMPT.format(language=language, spec_root=SPEC_ROOT, code_root=CODE_ROOT,
                           skills_block=_skills_block(skills), max_steps=steps,
                           stack_block=code_stack_block(stack)),
        temperature=temperature, model=model, name="code")
    listing = "\n".join(f"- FR {c['fr_id']} -> {path_by_fr[c['fr_id']]}" for c in contracts)
    opening = (f"Implement these {len(contracts)} functional requirement(s):\n{listing}\n\n"
               f"Their contract sheets are in {SPEC_ROOT}/.")

    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)

    # Submitted content ONLY — the /code files the agent drafted into are a scratchpad and are
    # deliberately not harvested. See "TOOL CALLING IS THE CONTRACT" in the module docstring.
    #
    # An agent that stops one message short of the contract — writing "the implementation has
    # been submitted" instead of calling submit_implementation — is not a failed run, it is an
    # unfinished one. Ask it to finish before giving up: cheaper than losing every other stage,
    # and it does not weaken the contract, because the artifact still has to arrive through the
    # validated tool.
    for _ in range(_SUBMIT_RETRIES):
        missing = sorted(known - set(submitted))
        if not missing:
            break
        state = nudge_agent(agent, state, _missing_nudge(missing, "submit_implementation"),
                            max_steps=steps)

    missing = sorted(known - set(submitted))
    if missing:
        raise DeepGenerationIncomplete(
            f"the code agent finished without submitting {len(missing)} of {len(contracts)} "
            f"FR(s): {', '.join(missing)}. It took {agent_step_count(state)} of {steps} steps. "
            f"Every implementation must arrive via submit_implementation.")

    files = [
        GeneratedFile(fr_id=c["fr_id"], feature_id=c["feature_id"], path=c["file_path"],
                      mvc_layer=c["mvc_layer"], content=submitted[c["fr_id"]])
        for c in contracts
    ]

    return files, {
        "driver": "deepagent", "frs_requested": len(contracts), "frs_generated": len(files),
        "frs_submitted": len(submitted), "agent_steps": agent_step_count(state),
        "max_steps": steps, "temperature": temperature,
    }


_CODE_STACK_FASTAPI = """\
TARGET STACK — FastAPI + SQLAlchemy. This is a REAL web service backed by a shared database.
NEVER fake state with a module-level list/dict and NEVER hardcode a returned value; every value
comes from the database via the session. These are STRUCTURAL conventions — derive all
behaviour from the contract sheets alone.

* Imports (exactly what you use): `from fastapi import APIRouter, Body, HTTPException`,
  `from sqlalchemy import Column, Integer, String, Float, Boolean`,
  `from sqlalchemy.orm import Session`, `from app.extensions import Base, SessionLocal`.
  NEVER create your own engine or call FastAPI().
* model layer: a `Base` subclass with typed `Column`s, exactly one `primary_key=True`;
  persist and read through a `Session` from `SessionLocal()` in a try/finally. Import any
  model you depend on from `app.models.<module>`.
* controller/view layer: ONE module-level `router = APIRouter()`, and decorate the contract
  function with `@router.post("")` — EMPTY path, the app registers it under a unique prefix.
  DEFINE the router only; never register it on an app.
* HTTP shape: keep the exact function name and return type. Give EVERY parameter a
  `= Body(..., embed=True)` default (e.g. `action: str = Body(..., embed=True)`) — `embed=True`
  is required on every one, so the request body is always the JSON object
  `{"<param>": <value>, ...}` even with a single parameter. Open `db = SessionLocal()` inside
  the function. Return the JSON-serializable response on success (it becomes the 200 body).
* On a precondition or validation failure `raise HTTPException(status_code=400, detail="...")`
  — 404 for a missing record, 403 for a forbidden actor. NEVER raise a bare ValueError.
* ENTITY KEY: when the sheet names an entity identifier, persist and look the entity up by
  THAT field (`db.query(Model).filter(Model.<key> == <key>).first()`) and store it on create.
  Do not key by any other field, and do not require an identifier the caller never sends.
"""

_CODE_STACK_PLAIN = """\
TARGET — plain, framework-free Python. Implement each FR as an ordinary importable function.

* Keep the EXACT function name, parameters and return type from the signature.
* Use ONLY JSON-serializable types (str, int, float, bool, list, dict).
* The function MUST succeed when called as `fn(**json.loads(<example inputs>))` and return a
  value equal to the sheet's expected return on the happy path — same length, keys and values.
* For a parameter like `request: dict`, read keys on the parameter directly
  (`request["auth_token"]`), NEVER nested as `request["request"][...]`.
* Raise ValueError on a precondition violation when the error mode is "raise".
* Do NOT call the network, sockets or the filesystem — simulate in memory when preconditions
  hold. Standard-library imports only.
"""

CODE_STACK_BLOCKS = {"fastapi": _CODE_STACK_FASTAPI}


def code_stack_block(stack: str) -> str:
    """Structural conventions for the run's target stack.

    These say how a file must be SHAPED to assemble into the packager's app — routers, session
    handling, the HTTP error type — never what any requirement does. That is the same
    structural knowledge the planner and test agent already have, so isolation is unaffected;
    without it a FastAPI run produces plain functions the packager cannot mount.
    """
    return CODE_STACK_BLOCKS.get(stack, _CODE_STACK_PLAIN)


# --------------------------------------------------------------------------------------
# Test Agent — sees contracts. Never code.
# --------------------------------------------------------------------------------------

TEST_PROMPT = """\
You are the test agent in a CLEAN-ROOM pipeline. You write BLACK-BOX {language} tests from
each functional requirement's contract.

Your filesystem:
* {spec_root}/ — one read-only contract sheet per FR. READ THESE FIRST.
* {test_root}/ — where your test source goes.
{skills_block}

You will NEVER be given the implementation. It is written independently from the same
contracts by another agent that never sees your tests, and it does not exist in your
filesystem — do not look for it. This is the point: your tests must follow from the
SPECIFICATION, so that agreement between them and the implementation is evidence, not
coincidence.

Rules:
* Test only what the contract states: the signature, the documented behaviour, the error mode.
* Never assume an internal detail — no private helpers, no attribute names, no data structure
  the contract does not specify. If you cannot express a check from the contract alone, it
  does not belong in the suite.
* Cover the happy path, the documented failure case, and any boundary the contract implies.
* `oracle="eq"` asserts a returned value; `oracle="raises"` asserts a failure.
{stack_block}
For each case call `submit_test_case(...)`. When every case is recorded, call
`submit_test_source(...)` with the complete runnable test file. These two tool calls are the
ONLY way to deliver the suite: a file left in {test_root}/ is NOT collected, and you are not
finished until both have been accepted. Cover every FR you were given — a case is required for
each one.
Start with `write_todos`, then read every contract sheet.

Each `submit_test_case` call adds a NEW case — there is no way to withdraw one, so do not
re-send a case that was already accepted; a corrected duplicate leaves BOTH in the suite and
the stale one will fail against correct code. Get each case right before you send it.
`submit_test_source` may be called again, and the last file wins.

You have about {max_steps} tool calls for this feature. If the budget runs short, make sure
every FR has at least one case and the source is submitted — a partial suite is rejected.
"""


def deep_generate_tests(
    ir: dict,
    feature_id: str,
    contracts: list[dict],
    *,
    language: str = "Python",
    stack: str = "python",
    temperature: float = 0.0,
    model: str | None = None,
    max_steps: int | None = None,
) -> tuple[FeatureTests | None, dict]:
    """Agent-driven counterpart to ``TestAgent.generate`` for ONE feature."""
    if not contracts:
        return None, {"skipped": "no contracts"}

    req_text = _requirement_index(ir)
    seeds = _seed_specs(contracts, req_text, _all_contracts(ir))          # /spec — same sheets the code agent got
    skills = load_skills(["blackbox-testing"])
    seeds.update(skills)
    source_path = virtual_path(TEST_ROOT, 0, f"test_{str(feature_id).replace('.', '_')}.py")
    # Not pre-seeded — see the note in deep_generate_code.
    # No CODE_ROOT and no PROOF_ROOT entry. See the module docstring.

    known = {c["fr_id"] for c in contracts}
    cases: list[TestCase] = []
    source: dict[str, str] = {}

    @tool
    def submit_test_case(
        requirement_id: Annotated[str, "The FR this case verifies, e.g. '1.1'."],
        description: Annotated[str, "What behaviour is being checked."],
        inputs: Annotated[str, "Human-readable summary of the inputs."],
        expected: Annotated[str, "Human-readable summary of the expected result."],
        inputs_json: Annotated[str, "JSON object of kwargs passed to the function."],
        expected_json: Annotated[str, "JSON value expected back (ignored when oracle='raises')."] = "",
        oracle: Annotated[str, '"eq" to compare a return value, "raises" for a ValueError.'] = "eq",
        setup_json: Annotated[str, "JSON list of prior calls needed to set up state; empty if none."] = "",
    ) -> str:
        """Record ONE black-box test case derived from a contract."""
        if requirement_id not in known:
            return (f"Unknown requirement_id {requirement_id!r}. Test only: "
                    f"{', '.join(sorted(known))}.")
        try:
            case = TestCase(
                requirement_id=requirement_id, description=description, inputs=inputs,
                expected=expected, inputs_json=inputs_json, expected_json=expected_json,
                oracle=oracle, setup_json=setup_json)
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
        """Record the finished test file. Call this once every case is recorded."""
        if not (test_source or "").strip():
            return "Rejected: test_source was empty."
        if not cases:
            return "Rejected: record the individual cases with submit_test_case first."
        source["src"] = test_source
        return f"Recorded the test source ({len(test_source.splitlines())} lines)."

    steps = max_steps or (deep_max_steps() + 10 * len(contracts))
    agent = build_agent(
        [submit_test_case, submit_test_source],
        TEST_PROMPT.format(language=language, spec_root=SPEC_ROOT, test_root=TEST_ROOT,
                           skills_block=_skills_block(skills), max_steps=steps,
                           stack_block=test_stack_block(stack)),
        temperature=temperature, model=model, name="test")
    if stack == "fastapi":
        # The endpoint is derived from file_path, which the contract sheet does not carry, so
        # the agent cannot work it out from its filesystem alone.
        listing = "\n".join(
            f"- FR {c['fr_id']}: {c.get('signature', '')}\n"
            f"    endpoint: POST /{(c.get('file_path') or '').removesuffix('.py').strip('/')}"
            + (f"\n    entity identifier: {c['entity_identifier']}"
               if c.get("entity_identifier") else "")
            for c in contracts)
    else:
        listing = "\n".join(f"- FR {c['fr_id']}: {c.get('signature', '')}" for c in contracts)
    opening = (f"Write black-box tests for feature {feature_id}, covering these "
               f"{len(contracts)} functional requirement(s):\n{listing}\n\n"
               f"Their contract sheets are in {SPEC_ROOT}/. Write the file to {source_path}.")

    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)

    # Same one-message-short recovery as the code agent: ask before discarding the run.
    for _ in range(_SUBMIT_RETRIES):
        gaps = ([] if source.get("src", "").strip() else ["the test source"]) + \
               [f"a case for FR {fr}" for fr in sorted(known - {c.requirement_id for c in cases})]
        if not gaps:
            break
        state = nudge_agent(agent, state, _missing_nudge(gaps, "submit_test_case/submit_test_source"),
                            max_steps=steps)

    # Submitted content ONLY — see "TOOL CALLING IS THE CONTRACT" in the module docstring.
    test_source = source.get("src", "")
    untested = sorted(known - {c.requirement_id for c in cases})
    if not test_source.strip() or untested:
        detail = (f"no test source was submitted" if not test_source.strip()
                  else f"no case covers FR(s) {', '.join(untested)}")
        raise DeepGenerationIncomplete(
            f"the test agent finished for feature {feature_id} but {detail}. It recorded "
            f"{len(cases)} case(s) in {agent_step_count(state)} of {steps} steps. The suite must "
            f"arrive via submit_test_case/submit_test_source.")

    result = FeatureTests(feature_id=str(feature_id), cases=cases, test_source=test_source)
    return result, {
        "driver": "deepagent", "frs_requested": len(contracts), "cases": len(cases),
        "has_source": bool(test_source.strip()), "agent_steps": agent_step_count(state),
        "max_steps": steps, "temperature": temperature,
    }


# --------------------------------------------------------------------------------------
# Proof Agent — sees contracts. Never code, never tests.
# --------------------------------------------------------------------------------------

_TEST_STACK_PLAIN = """\
* This target calls the functions DIRECTLY. `oracle="raises"` means a `ValueError`, and
  `expected_json` is ignored for those cases.
* `test_source` is a plain pytest module that imports each FR's function and calls it:
  `with pytest.raises(ValueError): func(**inputs)` for a failure case.
* `setup_json` is a list of prior CALLS needed to reach the state under test. Keep it minimal.
"""

_TEST_STACK_FASTAPI = """\
* This target is a REAL FastAPI web app, not a set of importable functions. Each FR is an HTTP
  endpoint at `POST /<its file_path without .py>`, taking the canonical inputs as a JSON body.
* For a failure case set `expected_json` to `{"raises": "HTTPException"}` — a precondition
  violation surfaces as a 4xx response, NOT a ValueError.
* SETUP MATTERS: the app's database starts EMPTY. A case that edits, deletes, cancels or reads
  an entity MUST first create it through `setup_json` — a JSON array of prior calls replayed
  against the same database, each `{"inputs": <body-object>}` (add `"route": "<endpoint>"`
  only to hit a different FR's endpoint). The seed call and the main call MUST use the SAME
  value for the requirement's entity identifier, or the lookup finds nothing and a correct
  implementation fails your test. Leave it empty for pure-create cases, and for a "raises" case
  that expects "not found" — that one must NOT create the entity it looks up.
* `test_source` must drive the app over HTTP with TestClient — do NOT import or call the route
  functions directly; they are FastAPI endpoints with Body params, not plain functions.
  Header: `import json`, `from fastapi.testclient import TestClient`, `from app import
  create_app`. Build ONE client at module level: `client = TestClient(create_app())`. For a
  "raises" case assert `resp.status_code >= 400`. Standard library + pytest + fastapi only.
"""

TEST_STACK_BLOCKS = {"fastapi": _TEST_STACK_FASTAPI}


def test_stack_block(stack: str) -> str:
    """Stack-specific test guidance — how a case is DRIVEN, not what the code does.

    Knowing the target is a web app rather than a module is structural, not seeing the
    implementation, so this preserves isolation exactly as the Code and Planning agents'
    stack awareness does. Without it the agent writes direct-call tests and ValueError
    expectations against an HTTP app, and every one of them fails a correct implementation.
    """
    return TEST_STACK_BLOCKS.get(stack, _TEST_STACK_PLAIN)


PROOF_PROMPT = """\
You are the proof agent in a CLEAN-ROOM pipeline. You write DAFNY that specifies and proves
each functional requirement's contract.

Your filesystem:
* {spec_root}/ — one read-only contract sheet per FR. READ THESE FIRST.
* {proof_root}/ — where your Dafny module goes.
{skills_block}

You will NEVER be given the implementation or the test suite. Both are derived independently
from the same contracts, and neither exists in your filesystem — do not look for them. Your
proof must follow from the SPECIFICATION alone.

You are not writing free-standing Dafny. The module REFINES a vendored kernel, and the rest of
the pipeline compiles and imports it by that structure — a proof that verifies but does not
refine the kernel is useless downstream. Write exactly one file that:

* starts with `include "Replay.dfy"`;
* defines `module {module}Domain refines Domain {{ ... }}` containing a concrete `type Model`,
  a `datatype Action` with ONE variant per FR, a `ghost predicate Inv`, `function Init`,
  `function Apply` (one match case per Action — NO `requires`), and `function Normalize`
  (repairs a Model so Inv holds);
* proves lemmas `InitSatisfiesInv` and `StepPreservesInv` (NO `requires` — both are inherited
  from the abstract Domain, and repeating them is an ERROR);
* proves at least one domain-specific lemma whose `ensures` matches a postcondition from the
  contract sheets;
* defines `module {module}Kernel refines Kernel {{ import D = {module}Domain }}`.

Use those module names EXACTLY — `{module}Domain` and `{module}Kernel`. The adapter that ships
this proof imports them by name.

=== THE ABSTRACT DOMAIN YOU ARE REFINING (do not redefine it) ===
{domain}

Rules:
* Model the requirement's behaviour with `requires`/`ensures` that state the contract's
  preconditions and postconditions.
* The error mode matters: a "raise" contract means the precondition belongs in `requires`.
* Prove what you specify. A method whose `ensures` is trivially true is worth nothing.
* `/skills/dafny-patterns.md` has a complete worked example of this exact structure. Read it
  before writing — the shape is easy to get subtly wrong and the verifier is unforgiving.
{verify_note}
Submit with `submit_dafny(module=..., source=...)` giving the COMPLETE module source. This
tool call is the ONLY way to deliver the module: a draft left in {proof_root}/ is NOT collected,
and you are not finished until it has been accepted. Submit even a proof that does not fully
verify — an honest partial module is a result; silence is not. Resubmitting replaces your
earlier module, so tightening a proof is just another call.
Start with `write_todos` listing the FRs, then read every contract sheet.

You have about {max_steps} tool calls for this feature, and `dafny_verify` spends them fast.
Submit a working module BEFORE you run low, then keep improving it — an unsubmitted proof
scores nothing, however good the draft.
"""

_VERIFY_NOTE = ("* Call `dafny_verify(source=...)` to check a draft. Iterate until it verifies "
                "or you run out of budget; report honestly if it does not.\n")


def _verify_failed_nudge(module: str, output: str) -> str:
    """Hand Dafny's own diagnostics back to the agent as the next turn."""
    return (f"Your submitted module `{module}` does NOT verify. Dafny reports:\n\n"
            f"{(output or '').strip()[:4000]}\n\n"
            f"A parse error means the file is not valid Dafny at all — fix the syntax first. "
            f"Note that a multi-field model must be `datatype Model = Model(field: T, ...)`; "
            f"`type Model = (a: T, b: U)` and `type Model = {{ a: T }}` are both invalid, and a "
            f"refining function or lemma must not repeat inherited requires/ensures clauses. "
            f"Read /skills/dafny-patterns.md if you have not. Then call `submit_dafny` again "
            f"with the corrected COMPLETE module source.")


def deep_generate_dafny(
    ir: dict,
    feature_id: str,
    contracts: list[dict],
    *,
    module: str | None = None,
    domain: str = "",
    verifier=None,
    temperature: float = 0.0,
    model: str | None = None,
    max_steps: int | None = None,
    max_rounds: int = 6,
) -> tuple[dict, dict]:
    """Agent-driven Dafny generation for ONE feature.

    ``verifier`` is an optional ``callable(source: str) -> (ok: bool, output: str)``. When
    given, the agent can check its own drafts and iterate. That is NOT an isolation break:
    the Dafny verifier is a proof checker over the agent's own text, not the test oracle the
    pipeline scores against — unlike the test suite, which stays outside every agent.

    Returns ``({"feature_id", "module", "source"}, metrics)``.
    """
    if not contracts:
        return {}, {"skipped": "no contracts"}

    req_text = _requirement_index(ir)
    seeds = _seed_specs(contracts, req_text, _all_contracts(ir))          # /spec — same sheets the others got
    # The deterministic DafnyAgent inlines these two documents into every system prompt. The
    # deep path seeds them instead, so the agent pays for them only when it reads them.
    skills = load_skills(["dafny-proofs", "dafny-patterns"])
    seeds.update(skills)
    # The CALLER owns this name. DafnyAgent writes the file as <mod>.dfy, records it as the
    # feature's proved module, and the packager compiles out/<mod>-py/ while the adapter imports
    # <mod>Domain — so a name invented here would disagree with every one of them.
    module = module or f"Feature_{str(feature_id).replace('.', '_')}"
    draft_path = virtual_path(PROOF_ROOT, 0, f"{module}.dfy")   # a drafting path, not an output
    # Not pre-seeded — see the note in deep_generate_code.
    # No CODE_ROOT and no TEST_ROOT entry. See the module docstring.

    submitted: dict[str, str] = {}
    verifications: list[bool] = []

    @tool
    def submit_dafny(
        module_name: Annotated[str, "The Dafny module base name — must match the one you were given."],
        source: Annotated[str, "The COMPLETE Dafny module source."],
    ) -> str:
        """Record the finished Dafny module for this feature."""
        if not (source or "").strip():
            return "Rejected: source was empty."
        # Reject a renamed module rather than accept it and let DafnyAgent silently overwrite
        # the name: the file, the proved-module record and the adapter's import all key off it.
        if module_name.strip() != module:
            return (f"Rejected: this feature's module must be named {module!r}, not "
                    f"{module_name!r}. The adapter imports {module}Domain by name. Rename it "
                    f"in the source and resubmit.")
        if f"module {module}Domain" not in source:
            return (f"Rejected: the source does not define `module {module}Domain refines "
                    f"Domain`. See /skills/dafny-patterns.md for the required structure.")
        submitted["module"] = module_name
        submitted["source"] = source
        return f"Recorded module {module_name} ({len(source.splitlines())} lines)."

    tools = [submit_dafny]
    verdicts: dict[str, tuple[bool, str]] = {}   # source -> (ok, output), from the agent's own calls
    verify_note = ""
    if verifier is not None:
        verify_note = _VERIFY_NOTE

        @tool
        def dafny_verify(
            source: Annotated[str, "The Dafny source to verify."],
        ) -> str:
            """Run the Dafny verifier over a draft and return its output."""
            if not (source or "").strip():
                return "Nothing to verify: source was empty."
            try:
                ok, output = verifier(source)
            except Exception as exc:                  # a broken verifier must not kill the run
                return f"The verifier could not be run ({exc}). Continue without it."
            verifications.append(bool(ok))
            verdicts[(source or "").strip()] = (bool(ok), output or "")
            return ("Verification SUCCEEDED.\n" if ok else "Verification FAILED.\n") + (output or "")

        tools.append(dafny_verify)

    steps = max_steps or (deep_max_steps() + 12 * len(contracts))
    agent = build_agent(
        tools,
        PROOF_PROMPT.format(spec_root=SPEC_ROOT, proof_root=PROOF_ROOT, module=module,
                            domain=domain or "(the kernel source was not available; follow "
                                             "/skills/dafny-patterns.md exactly)",
                            verify_note=verify_note, skills_block=_skills_block(skills),
                            max_steps=steps),
        temperature=temperature, model=model, name="proof")
    listing = "\n".join(f"- FR {c['fr_id']}: {c.get('signature', '')}" for c in contracts)
    opening = (f"Specify and prove feature {feature_id} in Dafny module `{module}`, covering "
               f"these {len(contracts)} functional requirement(s):\n{listing}\n\n"
               f"Their contract sheets are in {SPEC_ROOT}/. Draft at {draft_path} if it helps, "
               f"but deliver the module with submit_dafny.")

    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)

    # ---- Driver-owned generate -> verify -> revise loop --------------------------------
    # The agent is HANDED `dafny_verify` and told to iterate until its module verifies, and
    # this design deliberately dropped the old fixed round loop in favour of that. Measured
    # over 37 foodsaver features on Qwen3-32B-AWQ, it called the verifier ZERO times on every
    # single one: write once, submit, stop. 28 of 32 submitted modules did not even parse.
    #
    # So the loop cannot depend on the agent choosing to iterate. Verify what it submitted
    # ourselves and hand the verifier's own output back — Dafny names the file, line, column
    # and offending token, which is exactly what a revision needs. The agent still delivers
    # only through submit_dafny, so "TOOL CALLING IS THE CONTRACT" is untouched.
    for _ in range(max(1, max_rounds)):
        src = submitted.get("source", "").strip()
        if not src:
            state = nudge_agent(agent, state, _missing_nudge([f"module {module}"], "submit_dafny"),
                                max_steps=steps)
            continue
        if verifier is None:
            break
        if src in verdicts:
            ok, output = verdicts[src]          # the agent already checked this exact text
        else:
            try:
                ok, output = verifier(src)
            except Exception as exc:            # a broken verifier must not lose the agent's work
                submitted["source"] = src
                break
            verifications.append(bool(ok))
            verdicts[src] = (bool(ok), output or "")
        if ok:
            break
        submitted["source"] = ""      # force a genuine resubmission, not silent reuse
        state = nudge_agent(agent, state, _verify_failed_nudge(module, output), max_steps=steps)
    source_after_loop = submitted.get("source", "") or src
    submitted["source"] = source_after_loop

    # Submitted content ONLY — see "TOOL CALLING IS THE CONTRACT" in the module docstring.
    # Unlike code and tests this does NOT raise: an unproved feature is a legitimate outcome the
    # pipeline already records (DafnyAgent turns an empty source into an unverified FeatureDafny),
    # so a proof agent that gives up must not abort the whole run.
    source = submitted.get("source", "")
    out = {"feature_id": str(feature_id),
           "module": submitted.get("module") or module,
           "source": source}
    return out, {
        "driver": "deepagent", "frs_requested": len(contracts),
        "has_source": bool(source.strip()), "verify_calls": len(verifications),
        "verified": bool(verifications and verifications[-1]),
        "agent_steps": agent_step_count(state), "max_steps": steps,
        "temperature": temperature,
    }


# --------------------------------------------------------------------------------------
# Frontend Agent — sees contracts. Never code, never tests.
# --------------------------------------------------------------------------------------

FRONTEND_PROMPT = """\
You are the frontend agent in a CLEAN-ROOM pipeline. You write the browser UI for ONE feature
of a web app whose backend is being implemented independently from the same contracts.

Your filesystem:
* {spec_root}/ — one read-only contract sheet per FR. READ THESE FIRST.
* {ui_root}/ — where your page goes.
{skills_block}

You will NEVER be given the backend implementation. It does not exist in your filesystem — do
not look for it. You do not need it: every endpoint you must call is listed for you below,
derived from the same contract you are reading.

What to build — ONE self-contained HTML page for this feature:
* Plain HTML, CSS and vanilla JavaScript in a SINGLE file. No build step, no framework, no CDN
  or external URL of any kind — the app must run on a machine with no network.
* One clearly labelled section per functional requirement, in the order given.
* For each FR, a form whose inputs are that FR's parameters, prefilled with its example inputs
  so the page is usable immediately, and a button that POSTs to its endpoint.
* Send `fetch(endpoint, {{method:'POST', headers:{{'Content-Type':'application/json'}},
  body: JSON.stringify(payload)}})`. The payload is a JSON object keyed by parameter name —
  exactly the shape of the example inputs.
* Render the JSON reply into the page. On a non-2xx status show the error `detail` from the
  body; do not fail silently and never use `alert()`.
* Use RELATIVE endpoint paths (`/controllers/foo`) so the page works from the app's own origin.

Make it genuinely usable, not a debug harness: sensible labels drawn from the requirement text,
grouped controls, visible success and error states, and enough CSS that it reads as a real
screen. Do not invent behaviour the contracts do not describe, and do not add navigation to
features that are not yours.

Submit with `submit_page(html=...)` giving the COMPLETE file. That tool call is the ONLY way to
deliver it — a file left in {ui_root}/ is NOT collected. Resubmitting replaces the earlier page.
Start with `write_todos` listing the FRs, then read every contract sheet.

You have about {max_steps} tool calls for this feature.
"""


def deep_generate_frontend(
    ir: dict,
    feature_id: str,
    feature_name: str,
    contracts: list[dict],
    *,
    temperature: float = 0.0,
    model: str | None = None,
    max_steps: int | None = None,
) -> tuple[str, dict]:
    """One self-contained HTML page for a feature's FRs. Returns ``(html, metrics)``.

    UNSCORED: nothing certifies a UI, so an empty result degrades the deliverable and never
    fails the run — unlike code and tests, which raise. Isolation still holds: the agent is
    seeded with contract sheets and its own /ui pool, and the endpoints it needs are computed
    here from each contract's ``file_path`` rather than read out of the generated app.
    """
    if not contracts:
        return "", {"skipped": "no contracts"}

    req_text = _requirement_index(ir)
    seeds = _seed_specs(contracts, req_text, _all_contracts(ir))
    skills: dict[str, str] = {}
    page_path = virtual_path(UI_ROOT, 0, f"{str(feature_id).replace('.', '_')}.html")
    # No CODE_ROOT and no TEST_ROOT entry. See the module docstring.

    submitted: dict[str, str] = {}

    @tool
    def submit_page(
        html: Annotated[str, "The COMPLETE self-contained HTML page for this feature."],
    ) -> str:
        """Record the finished UI page for this feature."""
        if not (html or "").strip():
            return "Rejected: html was empty."
        lowered = html.lower()
        if "<form" not in lowered and "fetch(" not in lowered:
            return ("Rejected: the page neither renders a form nor calls fetch(), so it cannot "
                    "exercise the feature. Add the per-FR forms and resubmit.")
        # A CDN or any absolute external URL breaks an app running without network access.
        external = re.findall(r"""(?:src|href)\s*=\s*["'](https?://[^"']+)""", html, re.I)
        if external:
            return (f"Rejected: the page loads {len(external)} external URL(s) "
                    f"(e.g. {external[0]}). The app must run with no network — inline "
                    f"everything and resubmit.")
        replaced = bool(submitted)
        submitted["html"] = html
        return (f"{'Replaced' if replaced else 'Recorded'} the page "
                f"({len(html.splitlines())} lines). You are done.")

    steps = max_steps or (deep_max_steps() + 10 * len(contracts))
    agent = build_agent(
        [submit_page],
        FRONTEND_PROMPT.format(spec_root=SPEC_ROOT, ui_root=UI_ROOT,
                               skills_block=_skills_block(skills), max_steps=steps),
        temperature=temperature, model=model, name="frontend")

    # The endpoint is derived from file_path by the SAME route_for the packager and the cert
    # oracle use, so the page cannot drift from where the router is actually mounted — and the
    # agent never has to read the generated app to find out.
    listing = "\n".join(
        f"- FR {c['fr_id']}: {c.get('signature', '')}\n"
        f"    endpoint: POST {route_for(c.get('file_path', ''))}\n"
        f"    example body: {c.get('example_inputs_json', '{}')}"
        for c in contracts)
    opening = (f"Build the UI for feature {feature_id} — {feature_name} — covering these "
               f"{len(contracts)} functional requirement(s):\n{listing}\n\n"
               f"Their contract sheets are in {SPEC_ROOT}/. Write the page to {page_path}.")

    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)

    html = submitted.get("html", "")
    return html, {
        "driver": "deepagent", "frs_requested": len(contracts),
        "has_page": bool(html.strip()), "lines": len(html.splitlines()),
        "agent_steps": agent_step_count(state), "max_steps": steps,
        "temperature": temperature,
    }
