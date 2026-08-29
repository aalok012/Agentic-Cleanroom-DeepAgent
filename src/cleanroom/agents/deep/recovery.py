"""Deepagents driver for the recovery loop's test-informed regeneration (step (b)).

``RecoveryLoop`` step (b) is the pipeline's ONE sanctioned clean-room break: after a feature
has already failed both proof and the clean-room first pass, its code is regenerated with the
failing test cases visible. The deterministic implementation does that as one independent LLM
call per failing FR — each call sees only its own contract, prior content, and failing cases.

This driver replaces those independent calls with a single agent that holds ALL the failing
FRs of a feature at once. That is the substantive difference: a failure caused by two FRs
disagreeing about a shared data shape is invisible to per-FR regeneration but visible here,
and the agent can re-read the other FRs' current source while fixing one.

What is deliberately NOT changed:
  * Identity fields (fr_id, feature_id, path, mvc_layer) come from the planner's contract, as
    in ``CodeAgent.regenerate_with_test_feedback``. The agent authors ``content`` only.
  * The agent gets no test-execution tool. The frozen suite is re-run by the OUTER recovery
    loop's re-certification step, so the agent cannot iterate against the oracle it is being
    scored on. Keeping the verifier outside the agent is what stops step (b) from degenerating
    into fitting the test suite directly.
  * Tests are never regenerated — the oracle must not move.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool

from src.cleanroom.agents.code.schema.code import GeneratedFile
from src.cleanroom.agents.deep.runtime import (
    CODE_ROOT,
    agent_step_count,
    build_agent,
    deep_max_steps,
    final_files,
    invoke_agent,
    seed_files,
    virtual_path,
)
from src.cleanroom.utils.contracts import requirement_text

SPEC_ROOT = "/spec"

PROMPT = """\
You are repairing generated {language} source that FAILS its specification's test cases.

Layout of your filesystem:
* {code_root}/ — the current source, one file per functional requirement (FR). EDIT THESE.
* {spec_root}/ — for each FR: its requirement text, the signature and docstring it must
  implement, and the test cases it currently fails. READ-ONLY; never edit these.

Rules:
* The signature in the spec is a contract. Do not rename the function, change its parameter
  names or order, or change its return shape — callers and tests bind to it exactly.
* Fix the LOGIC so the requirement holds in general. Do not special-case the literal inputs
  from the failing cases; the cases are a sample of the requirement, not the requirement.
* These FRs belong to one feature and share data shapes. If two files disagree about a shape,
  fix the disagreement rather than patching one side — use `read_file` across {code_root}/ to
  check before you edit.
* Keep every file self-contained and importable: all imports it needs, no prose outside code.

How to submit a fix:
* For a small, surgical change use `edit_file`.
* For a rewrite of a whole file, call `submit_repair(fr_id=..., content=...)` with the FULL new
  source. (`write_file` cannot overwrite an existing file, so do not try to use it for this.)

Start by using `write_todos` to list the FRs you must fix, then read each FR's spec file and
its source before editing.
"""


def deep_regenerate_with_test_feedback(
    ir: dict,
    feature_ids: set[str],
    failures: list[dict],
    *,
    temperature: float = 0.4,
    language: str = "Python",
    max_steps: int | None = None,
) -> tuple[list[GeneratedFile], dict]:
    """Agent-driven counterpart to ``CodeAgent.regenerate_with_test_feedback``.

    Returns ``(regenerated_files, agent_metrics)``. Only files whose content the agent
    actually changed are returned, so ``RecoveryLoop._swap_files`` still swaps by fr_id.
    """
    contracts = (ir.get("planning") or {}).get("contracts")
    if not contracts:
        raise ValueError("deep_regenerate_with_test_feedback requires 'planning.contracts'.")

    feature_ids = {str(f) for f in feature_ids}
    targets = [c for c in contracts if str(c.get("feature_id")) in feature_ids]
    if not targets:
        return [], {"skipped": "no contracts for the failing features"}

    req_text = _requirement_index(ir)
    prior_by_fr = {f["fr_id"]: f.get("content") or ""
                   for f in (ir.get("generated_code") or {}).get("files", []) if f.get("fr_id")}
    fails_by_fr: dict[str, list[dict]] = {}
    for d in failures or []:
        fails_by_fr.setdefault(d.get("fr_id", ""), []).append({
            "description": d.get("description", ""),
            "inputs": d.get("inputs", ""),
            "expected": d.get("expected", ""),
            "reason": d.get("reason", ""),
        })

    seeds: dict[str, str] = {}
    path_by_fr: dict[str, str] = {}
    original: dict[str, str] = {}
    for i, contract in enumerate(targets):
        fr_id = contract["fr_id"]
        name = f"{str(fr_id).replace('.', '_')}.py"
        code_path = virtual_path(CODE_ROOT, i, name)
        path_by_fr[fr_id] = code_path
        original[fr_id] = prior_by_fr.get(fr_id, "")
        seeds[code_path] = original[fr_id]
        seeds[virtual_path(SPEC_ROOT, i, f"{str(fr_id).replace('.', '_')}.md")] = _spec_sheet(
            contract, req_text.get(fr_id, ""), fails_by_fr.get(fr_id, []))

    known_frs = {c["fr_id"] for c in targets}
    submitted: dict[str, str] = {}

    @tool
    def submit_repair(
        fr_id: Annotated[str, "The functional requirement id, e.g. '1.1'."],
        content: Annotated[str, "The COMPLETE new source of that FR's file."],
    ) -> str:
        """Replace one FR's file with a full new source. Use this for whole-file rewrites."""
        if fr_id not in known_frs:
            return (f"Unknown fr_id {fr_id!r}. You may only repair: "
                    f"{', '.join(sorted(known_frs))}.")
        if not (content or "").strip():
            return "Rejected: content was empty. Submit the complete source of the file."
        submitted[fr_id] = content
        return f"Recorded the repair for FR {fr_id} ({len(content.splitlines())} lines)."

    agent = build_agent([submit_repair],
                        PROMPT.format(language=language, code_root=CODE_ROOT,
                                      spec_root=SPEC_ROOT),
                        temperature=temperature, name="recovery-regen")
    listing = "\n".join(f"- {path_by_fr[c['fr_id']]}  (FR {c['fr_id']}, spec in {SPEC_ROOT}/)"
                        for c in targets)
    opening = (
        f"{len(targets)} functional requirement(s) fail their test cases:\n{listing}\n\n"
        "Read each spec sheet and its source, then repair the source files."
    )

    steps = max_steps or (deep_max_steps() + 10 * len(targets))
    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)
    produced = final_files(state)

    out: list[GeneratedFile] = []
    for contract in targets:
        fr_id = contract["fr_id"]
        # A `submit_repair` call is the agent's explicit answer, so it wins over the virtual
        # file; `edit_file` users are picked up from the filesystem instead.
        content = submitted.get(fr_id) or produced.get(path_by_fr[fr_id])
        if not content or content.strip() == (original[fr_id] or "").strip():
            continue                       # agent left this FR alone — keep the existing file
        out.append(GeneratedFile(
            fr_id=fr_id,
            feature_id=contract["feature_id"],
            path=contract["file_path"],
            mvc_layer=contract["mvc_layer"],
            content=content,               # identity from the contract, body from the agent
        ))

    metrics = {
        "driver": "deepagent",
        "frs_targeted": len(targets),
        "frs_changed": len(out),
        "frs_submitted": len(submitted),
        "agent_steps": agent_step_count(state),
        "max_steps": steps,
        "temperature": temperature,
    }
    return out, metrics


def _spec_sheet(contract: dict, requirement: str, failing_cases: list[dict]) -> str:
    """The read-only briefing for one FR: what it must do, and what it currently gets wrong."""
    lines = [
        f"# FR {contract['fr_id']} (feature {contract['feature_id']})",
        "",
        "## Requirement",
        requirement or "(not recorded)",
        "",
        "## Contract — implement exactly this signature",
        "```python",
        contract.get("signature", ""),
        '"""' + (contract.get("docstring", "") or "") + '"""',
        "```",
        "",
        f"- layer: {contract.get('mvc_layer', '')}",
        f"- error mode: {contract.get('error_mode', 'raise')}",
    ]
    if contract.get("example_inputs_json"):
        lines += ["", "## Example inputs", "```json", contract["example_inputs_json"], "```"]
    if contract.get("expected_return_json"):
        lines += ["", "## Expected return", "```json", contract["expected_return_json"], "```"]
    lines += ["", f"## Failing test cases ({len(failing_cases)})"]
    if not failing_cases:
        lines.append("(none recorded for this FR — it may fail only via a shared data shape.)")
    for i, case in enumerate(failing_cases, 1):
        lines += [
            "",
            f"### Case {i}: {case.get('description', '')}",
            f"- inputs: `{_fmt(case.get('inputs'))}`",
            f"- expected: `{_fmt(case.get('expected'))}`",
            f"- observed: {case.get('reason', '')}",
        ]
    return "\n".join(lines)


def _fmt(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return repr(value)


def _requirement_index(ir: dict) -> dict[str, str]:
    """fr_id -> requirement text. Same construction as ``CodeAgent._requirement_index`` so
    the agent reads the requirement in exactly the wording the first pass saw."""
    index: dict[str, str] = {}
    for feature in ir.get("features", []) or []:
        for req in feature.get("functional_requirements", []) or []:
            index[req["id"]] = requirement_text(req)
    return index
