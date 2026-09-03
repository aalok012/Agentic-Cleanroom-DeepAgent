"""Deepagents driver for the Planning stage.

The implementation of ``PlanningAgent._design_feature``: the agent reads each FR's spec sheet
from its filesystem, plans with ``write_todos``, and submits FRs ONE AT A TIME through
``submit_fr_plan``. ``PlanningAgent.plan`` keeps its normalization, dedupe and docstring
composition on top of the ``{fr_id: FRPlan}`` this returns.

Submitting per FR is the substantive gain over the single ``with_structured_output(FeaturePlan)``
call this replaced: a rejected field (bad ``mvc_layer``, malformed JSON, an invented id) comes
back as a tool error naming the problem, and the agent fixes that one FR instead of the whole
feature's structured output failing validation at once. That matters most on the smaller
open-weight models we now serve ourselves.

``submit_fr_plan`` is the ONLY channel: a plan the agent merely wrote down is not read back.
Unlike the code and test generators — which raise ``DeepGenerationIncomplete`` on an unsubmitted
artifact — an FR the agent never submits is simply absent here, because ``PlanningAgent.plan``
already has a documented degraded path for it (a default contract plus a recorded note). Aborting
the run would be strictly worse than that.

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

# How the three layers land in each target stack. The deleted plan_feature.j2 branched on this;
# without it the planner classifies layers with no idea what a "model" IS in the run's target,
# and mvc_layer decides the file path every downstream agent binds to.
_STACK_BLOCKS = {
    "fastapi": """\
TARGET STACK — FastAPI + SQLAlchemy, mapped per layer:
* model      -> a SQLAlchemy model plus the function that persists/queries it (DB access
                lives here, and only here).
* controller -> an APIRouter handler that validates input, calls the model, returns JSON.
* view       -> an APIRouter read endpoint / serializer returning the formatted response.
Every signature must stay JSON-serializable — these become HTTP endpoints taking a JSON body.""",
    "java": """\
TARGET STACK — plain Java (JDK standard library). The Python signature you write is the
reference interface; it is realized as a Java method.
* model      -> classes that hold, define or query data.
* controller -> classes that orchestrate a state change or workflow.
* view       -> classes that format and return a payload.
Keep signatures JSON-serializable — they map to Map/List/String/int/double/boolean.""",
    "spring": """\
TARGET STACK — Spring Boot. The Python signature you write is the reference interface.
* model      -> the persisted entity and its repository access.
* controller -> a @RestController handler orchestrating the state change.
* view       -> the response shape returned to the caller.
Keep signatures JSON-serializable — they map to Map/List/String/int/double/boolean.""",
}

_STACK_PLAIN = """\
TARGET STACK — plain Python. Signatures are called directly as `fn(**json.loads(inputs))`.
* model      -> data/state functions.
* controller -> behaviour and orchestration.
* view       -> functions that format and return a payload."""


def stack_block(stack: str) -> str:
    """Per-stack layer mapping for the planning prompt.

    Knowing the target stack is structural — it shapes what a signature and a layer MEAN — and
    is the same information the Code and Test agents already get. It reveals nothing about any
    implementation, so isolation is unaffected.
    """
    return _STACK_BLOCKS.get(stack, _STACK_PLAIN)

PROMPT = """\
You are the planning agent for a clean-room software pipeline. You design the INTERFACE for
each functional requirement (FR) of one feature. You do not write implementations or tests.

Your filesystem:
* {spec_root}/ — one spec sheet per FR: its requirement text and its behavioral contract
  (stimulus, precondition, response, postcondition). READ THESE FIRST — `ls {spec_root}/` to
  see them, then `read_file` each one. Treat them as read-only; you have nothing to add there.

For every FR you must call `submit_fr_plan` with:
* id                     — the FR id, copied VERBATIM from the spec sheet. Never invent one.
* signature              — a one-line function signature in Python syntax (name, typed
                           params, return type), no body. This is the canonical,
                           language-neutral interface; code generation realizes it in the
                           target language, so use Python syntax whatever the target.
                           Use JSON-SERIALIZABLE TYPES ONLY — str, int, float, bool, list,
                           dict. NO custom classes, Pydantic models or ORM types: the case
                           runner invokes the function as `fn(**json.loads(inputs))`, and a
                           non-serializable parameter cannot be built from JSON.
                           e.g. `def search_videos(query: str) -> dict:`
* args_json              — JSON list of {{"name": ..., "description": ...}}, one per
                           parameter, names matching the signature EXACTLY, each describing
                           what it is and its constraints. `[]` if the signature takes none.
* returns                — what the function returns; empty string for a None return.
* mvc_layer              — exactly one of: model, view, controller. This decides the file
                           path, so it shapes the whole app. Judge it from the CONTRACT
                           (stimulus + response), never from keywords in the name:
                             model      — owns DATA. Defines, persists, queries or updates a
                                          stored entity. No request handling, no formatting.
                             view       — owns PRESENTATION. Shapes data for output and never
                                          mutates state: a read whose response IS the payload.
                             controller — owns BEHAVIOUR. Orchestrates a state change: takes
                                          input, invokes the model, applies rules, returns.
                           Flow: controller takes input -> calls model -> view serializes.
                           Tie-breakers, by PRIMARY effect:
                             changes persisted state -> controller (even if it also reads);
                             pure read returning a formatted payload -> view;
                             only defines/stores an entity, no orchestration -> model.
                           e.g. "cancel an order before serving" -> controller (mutates);
                                "display available dishes" -> view (read-only payload);
                                "add/edit/delete staff records" -> model (entity CRUD).
* example_inputs_json    — JSON OBJECT of happy-path kwargs. The function is always invoked
                           as `fn(**inputs)`, so EVERY top-level key must match a signature
                           parameter exactly — no extra keys, none missing. `{{}}` for a
                           no-arg function.
* expected_return_json   — the JSON value returned on that happy path.
* error_mode             — "raise" (ValueError) or "return" (an error dict).
* failure_inputs_json    — JSON kwargs for ONE precondition violation that must fail. Use
                           the same parameter names as the signature and genuinely violate
                           the stated precondition (empty required string, false where true
                           is required, invalid enum). Use "" when the precondition is
                           "none"/empty, the function takes no parameters, or no clear
                           testable violation exists.
* entity_identifier      — for a STATEFUL CRUD requirement (create/edit/delete/look up a
                           PERSISTED entity), the ONE field that uniquely keys it for lookup.
                           It MUST be a key of example_inputs_json (possibly nested inside an
                           entity object). Pick a key the stimulus ACTUALLY PROVIDES on
                           edit/delete — if records are referenced by name, use "name", not
                           an "id" the caller never supplies. "" for stateless requirements
                           and pure computations with no persisted entity.

Design rules:
* The signature is a contract three downstream agents bind to independently. Make it precise
  and self-explanatory; they cannot ask you what you meant.
* Parameter names in args_json, example_inputs_json and the signature must agree exactly.
* Prefer the vocabulary of the requirement text over invented terms.
* Use DISTINCT function names when two FRs do different things — do not reuse one name for
  both (prefer `sort_torrent_results` / `sort_video_results` over two `sort_search_results`).
  Two FRs sharing a name collide on one file path and get suffixed apart.
* Prefer explicit scalar parameters (`update: bool`, `criteria: str`) over a single opaque
  `request: dict` when the stimulus names one clear input.

{stack_block}

Start with `write_todos` listing every FR id you must plan, then read each spec sheet, then
submit them one at a time. Finish only when every FR has been ACCEPTED — the tool tells you
which ids are still outstanding after each call.

A rejection is not final. It names the field that was wrong; fix that one FR and call
`submit_fr_plan` again. Resubmitting an id REPLACES its earlier plan, so you can also revise
one you already got accepted. An FR you give up on is dropped from the design entirely, which
is far worse than a rough plan.

You have about {max_steps} tool calls for this feature. Reading every sheet first is worth it;
re-reading one is not.
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

    # The budget is stated IN the prompt, so it has to exist before the prompt is built.
    steps = max_steps or (deep_max_steps() + 8 * len(fr_order))
    agent = build_agent([submit_fr_plan],
                        PROMPT.format(spec_root=SPEC_ROOT, max_steps=steps,
                                      stack_block=stack_block(stack)),
                        temperature=temperature, model=model, name="planning")
    # Name each sheet's exact path. virtual_path() prefixes an index (/spec/00_1_1.md), so an
    # agent told only the directory must spend a turn on `ls` to learn what the driver already
    # knows. The code agent lists its paths the same way.
    sheet_of = {rid: virtual_path(SPEC_ROOT, i, f"{str(rid).replace('.', '_')}.md")
                for i, rid in enumerate(fr_order)}
    listing = "\n".join(f"- FR {rid} -> {sheet_of[rid]}" for rid in fr_order)
    opening = (
        f"Feature: {feature_name}\nTarget stack: {stack}\n\n"
        f"Design the interface for these {len(fr_order)} functional requirement(s):\n"
        f"{listing}"
    )

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
    # The prompt states entity_identifier "must appear in example_inputs_json". Every rule
    # beside it is enforced here, which made this one look enforced too — it was not, and the
    # value flows on into the code agent's prompt as if the planner had vouched for it.
    entity = (plan.entity_identifier or "").strip()
    if entity and isinstance(example, dict) and example and entity not in example:
        return (f"entity_identifier {entity!r} is not a key of example_inputs_json "
                f"{sorted(example)}; it must name the field that uniquely keys the entity.")
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
