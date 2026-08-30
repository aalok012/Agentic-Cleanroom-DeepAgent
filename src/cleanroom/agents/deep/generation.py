"""Deepagents drivers for the three clean-room generators (Stage 2 of the migration).

Code, Test and Proof each derive their artifact from the planner's contract ALONE. This
module is where the paper's central claim is implemented, so read the isolation note before
changing anything here.

==========================  ISOLATION  ==========================
Each generator is a SEPARATE top-level ``create_deep_agent`` invocation with its own
``StateBackend``, seeded with exactly one pool:

    deep_generate_code   seeds /spec (contracts) + /code   — never /tests, never /proof
    deep_generate_tests  seeds /spec (contracts) + /tests  — never /code,  never /proof
    deep_generate_dafny  seeds /spec (contracts) + /proof  — never /code,  never /tests

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
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import ValidationError

from src.cleanroom.agents.code.schema.code import GeneratedFile
from src.cleanroom.agents.deep.runtime import (
    CODE_ROOT,
    PROOF_ROOT,
    SPEC_ROOT,
    TEST_ROOT,
    agent_step_count,
    build_agent,
    deep_max_steps,
    final_files,
    invoke_agent,
    load_skills,
    seed_files,
    skill_index,
    virtual_path,
)
from src.cleanroom.agents.test.schema.tests import FeatureTests, TestCase
from src.cleanroom.utils.contracts import requirement_text

# --------------------------------------------------------------------------------------
# Shared: the contract sheet every generator reads, and nothing else.
# --------------------------------------------------------------------------------------

def contract_sheet(contract: dict, requirement: str) -> str:
    """The read-only briefing for one FR — identical for all three generators.

    Deliberately identical: if Code, Test and Proof were briefed differently, a disagreement
    between their artifacts would be an artifact of the briefing rather than evidence about
    independent derivation.
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


def _seed_specs(contracts: list[dict], req_text: dict[str, str]) -> dict[str, str]:
    """``/spec`` sheets for a set of contracts — the shared read-only input."""
    return {
        virtual_path(SPEC_ROOT, i, f"{str(c.get('fr_id')).replace('.', '_')}.md"):
            contract_sheet(c, req_text.get(c.get("fr_id"), ""))
        for i, c in enumerate(contracts)
    }


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

Submit each file with `submit_implementation(fr_id=..., content=...)` giving the FULL source.
(You may also `write_file` to the listed path; `submit_implementation` is preferred because it
validates the fr_id. Note `write_file` cannot overwrite a file that already exists — use
`edit_file` to revise one.)
Start with `write_todos` listing the FRs, then read every contract sheet before writing.
"""


def deep_generate_code(
    ir: dict,
    contracts: list[dict],
    *,
    language: str = "Python",
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
    seeds = _seed_specs(contracts, req_text)          # /spec — shared read-only input
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
        submitted[fr_id] = content
        remaining = sorted(known - set(submitted))
        return (f"Recorded FR {fr_id} ({len(content.splitlines())} lines). "
                + (f"Still to implement: {', '.join(remaining)}." if remaining
                   else "All FRs implemented — you are done."))

    agent = build_agent(
        [submit_implementation],
        CODE_PROMPT.format(language=language, spec_root=SPEC_ROOT, code_root=CODE_ROOT,
                           skills_block=_skills_block(skills)),
        temperature=temperature, model=model, name="code")
    listing = "\n".join(f"- FR {c['fr_id']} -> {path_by_fr[c['fr_id']]}" for c in contracts)
    opening = (f"Implement these {len(contracts)} functional requirement(s):\n{listing}\n\n"
               f"Their contract sheets are in {SPEC_ROOT}/.")

    steps = max_steps or (deep_max_steps() + 10 * len(contracts))
    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)
    produced = final_files(state)

    files: list[GeneratedFile] = []
    for c in contracts:
        fr_id = c["fr_id"]
        content = submitted.get(fr_id) or produced.get(path_by_fr[fr_id]) or ""
        if not content.strip():
            continue
        files.append(GeneratedFile(
            fr_id=fr_id, feature_id=c["feature_id"], path=c["file_path"],
            mvc_layer=c["mvc_layer"], content=content))

    return files, {
        "driver": "deepagent", "frs_requested": len(contracts), "frs_generated": len(files),
        "frs_submitted": len(submitted), "agent_steps": agent_step_count(state),
        "max_steps": steps, "temperature": temperature,
    }


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
* `oracle="eq"` asserts a returned value; `oracle="raises"` asserts a ValueError.

For each case call `submit_test_case(...)`. When every case is recorded, call
`submit_test_source(...)` with the complete runnable test file.
Start with `write_todos`, then read every contract sheet.
"""


def deep_generate_tests(
    ir: dict,
    feature_id: str,
    contracts: list[dict],
    *,
    language: str = "Python",
    temperature: float = 0.0,
    model: str | None = None,
    max_steps: int | None = None,
) -> tuple[FeatureTests | None, dict]:
    """Agent-driven counterpart to ``TestAgent.generate`` for ONE feature."""
    if not contracts:
        return None, {"skipped": "no contracts"}

    req_text = _requirement_index(ir)
    seeds = _seed_specs(contracts, req_text)          # /spec — same sheets the code agent got
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

    agent = build_agent(
        [submit_test_case, submit_test_source],
        TEST_PROMPT.format(language=language, spec_root=SPEC_ROOT, test_root=TEST_ROOT,
                           skills_block=_skills_block(skills)),
        temperature=temperature, model=model, name="test")
    listing = "\n".join(f"- FR {c['fr_id']}: {c.get('signature', '')}" for c in contracts)
    opening = (f"Write black-box tests for feature {feature_id}, covering these "
               f"{len(contracts)} functional requirement(s):\n{listing}\n\n"
               f"Their contract sheets are in {SPEC_ROOT}/. Write the file to {source_path}.")

    steps = max_steps or (deep_max_steps() + 10 * len(contracts))
    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)

    test_source = source.get("src") or final_files(state).get(source_path) or ""
    result = FeatureTests(feature_id=str(feature_id), cases=cases, test_source=test_source)
    return result, {
        "driver": "deepagent", "frs_requested": len(contracts), "cases": len(cases),
        "has_source": bool(test_source.strip()), "agent_steps": agent_step_count(state),
        "max_steps": steps, "temperature": temperature,
    }


# --------------------------------------------------------------------------------------
# Proof Agent — sees contracts. Never code, never tests.
# --------------------------------------------------------------------------------------

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

Rules:
* Model the requirement's behaviour as Dafny functions/methods with `requires` and `ensures`
  that state the contract's preconditions and postconditions.
* The error mode matters: a "raise" contract means the precondition belongs in `requires`.
* Prove what you specify. A method whose `ensures` is trivially true is worth nothing.
{verify_note}
Submit with `submit_dafny(module=..., source=...)` giving the COMPLETE module source.
Start with `write_todos` listing the FRs, then read every contract sheet.
"""

_VERIFY_NOTE = ("* Call `dafny_verify(source=...)` to check a draft. Iterate until it verifies "
                "or you run out of budget; report honestly if it does not.\n")


def deep_generate_dafny(
    ir: dict,
    feature_id: str,
    contracts: list[dict],
    *,
    verifier=None,
    temperature: float = 0.0,
    model: str | None = None,
    max_steps: int | None = None,
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
    seeds = _seed_specs(contracts, req_text)          # /spec — same sheets the others got
    # The deterministic DafnyAgent inlines these two documents into every system prompt. The
    # deep path seeds them instead, so the agent pays for them only when it reads them.
    skills = load_skills(["dafny-proofs", "dafny-patterns"])
    seeds.update(skills)
    module = f"Feature_{str(feature_id).replace('.', '_')}"
    draft_path = virtual_path(PROOF_ROOT, 0, f"{module}.dfy")
    # Not pre-seeded — see the note in deep_generate_code.
    # No CODE_ROOT and no TEST_ROOT entry. See the module docstring.

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
            return ("Verification SUCCEEDED.\n" if ok else "Verification FAILED.\n") + (output or "")

        tools.append(dafny_verify)

    agent = build_agent(
        tools,
        PROOF_PROMPT.format(spec_root=SPEC_ROOT, proof_root=PROOF_ROOT,
                            verify_note=verify_note, skills_block=_skills_block(skills)),
        temperature=temperature, model=model, name="proof")
    listing = "\n".join(f"- FR {c['fr_id']}: {c.get('signature', '')}" for c in contracts)
    opening = (f"Specify and prove feature {feature_id} in Dafny module `{module}`, covering "
               f"these {len(contracts)} functional requirement(s):\n{listing}\n\n"
               f"Their contract sheets are in {SPEC_ROOT}/.")

    steps = max_steps or (deep_max_steps() + 12 * len(contracts))
    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)

    source = submitted.get("source") or final_files(state).get(draft_path) or ""
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
