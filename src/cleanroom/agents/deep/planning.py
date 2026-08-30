"""Deepagents driver for the Planning stage (Stage 1 of the migration).

Drop-in for ``PlanningAgent._design_feature``: same inputs, same ``{fr_id: FRPlan}`` return,
so ``PlanningAgent.plan`` keeps its normalization, dedupe and docstring composition unchanged.
The difference is who drives the design:

  deterministic : ONE ``with_structured_output(FeaturePlan)`` call per feature — the model
                  sees every FR at once and must emit the whole plan in a single shot.
  deepagent     : the agent reads each FR's spec sheet from its filesystem, plans with
                  ``write_todos``, and submits FRs ONE AT A TIME through ``submit_fr_plan``.

Submitting per FR is the substantive gain: a rejected field (bad ``mvc_layer``, malformed
JSON, an invented id) comes back as a tool error naming the problem, and the agent fixes that
one FR instead of the whole feature's structured output failing validation at once. That
matters most on the smaller open-weight models we now serve ourselves.

==========================  ISOLATION  ==========================
The planning stage has NO isolation requirement: its output — the per-FR contract — is
deliberately the shared read-only input to the Code, Test and Proof agents. That is what
makes their derivations independent *of each other* while still targeting one interface.

It still runs on ``StateBackend`` like every other driver, seeded with spec sheets only, so
the planner cannot read the repository it is planning for. Nothing here may be reused to give
a downstream agent a wider filesystem — see ``deep/README.md``.
=================================================================
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import tool
from pydantic import ValidationError

from src.cleanroom.agents.deep.runtime import (
    agent_step_count,
    build_agent,
    deep_max_steps,
    invoke_agent,
    seed_files,
    virtual_path,
)
from src.cleanroom.agents.planning.schema.plan import ArgDoc, FRPlan

SPEC_ROOT = "/spec"

PROMPT = """\
You are the planning agent for a clean-room software pipeline. You design the INTERFACE for
each functional requirement (FR) of one feature. You do not write implementations or tests.

Your filesystem:
* {spec_root}/ — one read-only spec sheet per FR: its requirement text and its behavioral
  contract (preconditions, postconditions, error cases). READ THESE FIRST.

For every FR you must call `submit_fr_plan` exactly once with:
* id                     — the FR id, copied VERBATIM from the spec sheet. Never invent one.
* signature              — a one-line function signature in Python syntax (name, typed
                           params, return type). This is the canonical, language-neutral
                           interface; code generation realizes it in the target language.
* args_json              — JSON list of {{"name": ..., "description": ...}}, one per
                           parameter, names matching the signature EXACTLY. `[]` if none.
* returns                — what the function returns; empty string for a None return.
* mvc_layer              — exactly one of: model, view, controller.
* example_inputs_json    — JSON object mapping parameter names to happy-path values.
* expected_return_json   — the JSON value returned on that happy path.
* error_mode             — "raise" (ValueError) or "return" (an error dict).
* failure_inputs_json    — JSON kwargs that must fail; "" if the FR has no failure case.
* entity_identifier      — for a requirement that persists an entity, the ONE field that
                           uniquely keys it (must appear in example_inputs_json); "" otherwise.

Design rules:
* The signature is a contract three downstream agents bind to independently. Make it precise
  and self-explanatory; they cannot ask you what you meant.
* Parameter names in args_json, example_inputs_json and the signature must agree exactly.
* Prefer the vocabulary of the requirement text over invented terms.

Start with `write_todos` listing every FR id you must plan, then read each spec sheet, then
submit them one at a time. Finish only when every FR has been accepted.
"""


def deep_design_feature(
    feature_name: str,
    fr_order: list[str],
    text_by_id: dict[str, str],
    contracts_by_fr: dict,
    *,
    stack: str = "python",
    temperature: float = 0.0,
    model: str | None = None,
    max_steps: int | None = None,
    return_metrics: bool = False,
):
    """Agent-driven counterpart to ``PlanningAgent._design_feature``.

    Returns ``{normalized_fr_id: FRPlan}`` — or ``(plans, metrics)`` when ``return_metrics``.
    FRs the agent never submitted are simply absent, exactly as a short structured response
    would leave them, so the caller's existing gap handling still applies.
    """
    from src.cleanroom.agents.planning.agent import _norm_id

    if not fr_order:
        return ({}, {"skipped": "no FRs"}) if return_metrics else {}

    seeds: dict[str, str] = {}
    for i, rid in enumerate(fr_order):
        seeds[virtual_path(SPEC_ROOT, i, f"{str(rid).replace('.', '_')}.md")] = _spec_sheet(
            rid, text_by_id.get(rid, ""), contracts_by_fr.get(rid), stack)

    known = {_norm_id(r) for r in fr_order}
    accepted: dict[str, FRPlan] = {}

    @tool
    def submit_fr_plan(
        id: Annotated[str, "The FR id, copied verbatim from its spec sheet."],
        signature: Annotated[str, "One-line function signature in Python syntax."],
        mvc_layer: Annotated[str, "Exactly one of: model, view, controller."],
        example_inputs_json: Annotated[str, 'JSON object of happy-path kwargs, e.g. {"q": "x"}.'],
        expected_return_json: Annotated[str, "JSON value returned on the happy path."],
        args_json: Annotated[str, 'JSON list of {"name","description"}, one per parameter.'] = "[]",
        returns: Annotated[str, "What the function returns; empty for None."] = "",
        error_mode: Annotated[str, '"raise" or "return".'] = "raise",
        failure_inputs_json: Annotated[str, "JSON kwargs that must fail; empty if none."] = "",
        entity_identifier: Annotated[str, "Unique key field for a persisted entity; empty otherwise."] = "",
    ) -> str:
        """Record the finished interface design for ONE functional requirement."""
        norm = _norm_id(id)
        if norm not in known:
            return (f"Unknown FR id {id!r}. Plan only these, copied verbatim: "
                    f"{', '.join(sorted(known))}.")
        try:
            args = [ArgDoc(**a) for a in json.loads(args_json or "[]")]
        except (ValueError, TypeError) as exc:
            return (f"args_json is not a JSON list of objects with 'name' and 'description': "
                    f"{exc}. Resubmit this FR with valid JSON.")
        try:
            plan = FRPlan(
                id=id, signature=signature, args=args, returns=returns, mvc_layer=mvc_layer,
                example_inputs_json=example_inputs_json,
                expected_return_json=expected_return_json, error_mode=error_mode,
                failure_inputs_json=failure_inputs_json, entity_identifier=entity_identifier,
            )
        except ValidationError as exc:
            # Hand the agent the precise field error so it can fix THIS FR and resubmit,
            # rather than the whole feature's output failing validation at once.
            return f"Rejected — fix these fields and resubmit FR {id}:\n{_brief(exc)}"

        missing = _param_mismatch(signature, plan)
        if missing:
            return f"Rejected — {missing} Resubmit FR {id} with them consistent."

        accepted[norm] = plan
        remaining = sorted(known - set(accepted))
        return (f"Accepted FR {id}. "
                + (f"Still to plan: {', '.join(remaining)}." if remaining
                   else "All FRs planned — you are done."))

    agent = build_agent([submit_fr_plan], PROMPT.format(spec_root=SPEC_ROOT),
                        temperature=temperature, model=model, name="planning")
    listing = "\n".join(f"- FR {rid}" for rid in fr_order)
    opening = (
        f"Feature: {feature_name}\nTarget stack: {stack}\n\n"
        f"Design the interface for these {len(fr_order)} functional requirement(s):\n"
        f"{listing}\n\nTheir spec sheets are in {SPEC_ROOT}/."
    )

    steps = max_steps or (deep_max_steps() + 8 * len(fr_order))
    state = invoke_agent(agent, opening, seed_files(seeds), max_steps=steps)

    if not return_metrics:
        return accepted
    return accepted, {
        "driver": "deepagent",
        "frs_requested": len(fr_order),
        "frs_planned": len(accepted),
        "agent_steps": agent_step_count(state),
        "max_steps": steps,
    }


def _brief(exc: ValidationError) -> str:
    """Field errors as short lines the model can act on, without pydantic's URLs."""
    return "\n".join(
        f"- {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8])


def _param_mismatch(signature: str, plan: FRPlan) -> str:
    """Complain when args/example inputs disagree with the signature's parameters.

    The three downstream agents bind to this signature independently, so a name that appears
    in one place and not another becomes three inconsistent implementations.
    """
    from src.cleanroom.agents.planning.agent import _parse_signature_params

    params = [p for p in _parse_signature_params(signature) if p not in ("self", "cls")]
    if not params:
        return ""
    named = [a.name for a in plan.args]
    if named and set(named) != set(params):
        return (f"args_json names {named} but the signature takes {params}; they must match "
                f"exactly.")
    try:
        example = json.loads(plan.example_inputs_json or "{}")
    except ValueError:
        return "example_inputs_json is not valid JSON."
    if isinstance(example, dict) and example:
        unknown = sorted(set(example) - set(params))
        if unknown:
            return (f"example_inputs_json has key(s) {unknown} that are not parameters of the "
                    f"signature {params}.")
    return ""


def _spec_sheet(fr_id: str, requirement: str, contract, stack: str) -> str:
    """The read-only briefing for one FR: what it must do, per the parsed SRS."""
    lines = [
        f"# FR {fr_id}",
        "",
        "## Requirement",
        requirement or "(not recorded)",
        "",
        f"- target stack: {stack}",
    ]
    if contract:
        lines += ["", "## Behavioral contract", "```json",
                  json.dumps(contract, indent=2, default=str), "```"]
    else:
        lines += ["", "## Behavioral contract", "(none extracted — design from the text above.)"]
    return "\n".join(lines)
