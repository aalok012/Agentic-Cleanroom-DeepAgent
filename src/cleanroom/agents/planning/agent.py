"""Planning Agent.

Walks the functional requirements in dependency order (outer feature order, then inner
FR order from the nested dependency graph) and attaches IMPLEMENTATION metadata to each
FR's spec-derived behavioral contract, producing a CONTRACT per FR:

  - signature : a concrete function signature (name, typed params, return type)   [LLM]
  - docstring : Google-style (Napoleon) — summary + Args + Returns + Raises + Notes,
                composed DETERMINISTICALLY from the behavioral contract + fr_edges
  - mvc_layer : model | view | controller                                          [LLM, deterministic fallback]
  - file_path : derived DETERMINISTICALLY from the layer
  - prerequisite_fr_ids : the FR's prerequisites, taken from the inner fr_edges     [deterministic]
  - contract  : the spec-derived behavioral contract this implementation realizes

DETERMINISTIC-FIRST: ordering, ids, prerequisites, the layer->path mapping, and the whole
docstring assembly are plain code. The LLM is used only for interpretation (signature
naming, per-arg/return docs, layer choice). The LLM never assigns or invents ids. Token
discipline: ONE structured call per feature (never the whole IR); via with_structured_output().

The planner runs ONCE globally and consumes the SPEC + behavioral contracts only — no code,
no tests — preserving the isolation the later code/test agents depend on.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from src.cleanroom.agents.planning.schema.plan import Contract, PlanningOutput
from src.cleanroom.agents.spec_agent.schema.ir import BehavioralContract
from src.cleanroom.utils.ir import feature_id_of, normalize_ir_features, requirement_text
from src.cleanroom.utils.llm_client import get_llm
from src.cleanroom.utils.prompt_renderer import PromptRenderer, cot_template

# Directory each MVC layer's files live under. The planning agent's `submit_fr_plan` tool is the
# classifier (Literal-typed `mvc_layer`, rejected back to the agent when invalid); there is no
# keyword fallback — anything that still gets through defaults to controller.
_LAYER_DIR = {"model": "models", "view": "views", "controller": "controllers"}


def feature_of(req_id: str) -> str:
    """Feature id is the first two dot-segments of a requirement id: '2.2.6.6' -> '2.2'."""
    return ".".join(req_id.split(".")[:2])


def _numeric_key(id_str: str) -> tuple[tuple[int, int, str], ...]:
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in id_str.split("."))


def _norm_id(raw: str) -> str:
    """Normalize an id the LLM echoed back (e.g. '[2.2.1]') to the bare parser id."""
    return (raw or "").strip().strip("[]").strip()


def _slug(fr_id: str) -> str:
    return re.sub(r"\W", "_", fr_id)


def _contract_field(contract, field: str) -> str:
    """Read a behavioral-contract field (object or dict), treating empty/'none' as absent."""
    if contract is None:
        return ""
    value = getattr(contract, field, None) if not isinstance(contract, dict) else contract.get(field)
    text = (value or "").strip()
    return "" if text.lower() in ("", "none", "n/a", "na") else text


def _func_name(signature: str, fr_id: str) -> str:
    """Deterministic file stem: the function name from the signature, else a slug of the id."""
    match = re.search(r"def\s+([A-Za-z_]\w*)", signature or "")
    if match:
        return match.group(1)
    return "fr_" + _slug(fr_id)


def _normalize_layer(raw: str) -> str:
    """Coerce the LLM's layer to model|view|controller; default to 'controller' if invalid.
    `submit_fr_plan` is the classifier and rejects a bad layer back to the agent — this only
    guards output that somehow survives that."""
    layer = (raw or "").strip().lower()
    return layer if layer in _LAYER_DIR else "controller"


def _file_path(layer: str, signature: str, fr_id: str) -> str:
    """Return a path relative to the project's code output directory.

    Written as {layer_dir}/{func}.py so the Code Agent can write it under any
    base directory without path conflicts.
    """
    return f"{_LAYER_DIR[layer]}/{_func_name(signature, fr_id)}.py"


def _parse_signature_params(signature: str) -> list[str]:
    """Extract parameter names from a one-line function signature."""
    if not signature:
        return []
    match = re.search(r"def\s+\w+\s*\((.*?)\)\s*->", signature, re.DOTALL)
    if not match:
        return []
    inner = match.group(1).strip()
    if not inner:
        return []
    params: list[str] = []
    for part in inner.split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split(":")[0].split("=")[0].strip()
        if name and name != "self":
            params.append(name)
    return params


def _align_kwargs_json(signature: str, json_str: str) -> str:
    """Normalize kwargs JSON so fn(**inputs) matches the signature.

    Certification and verification call functions with spread kwargs. When the LLM
    returns a single ``request: dict`` parameter but puts payload keys at the top level,
    wrap them under ``request``. When there is exactly one parameter and keys do not
    match its name, wrap the whole object as that parameter's value.
    """
    params = _parse_signature_params(signature)
    raw = (json_str or "").strip()
    if not raw:
        return "{}"
    try:
        inputs = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(inputs, dict):
        return raw
    if not params:
        return "{}"
    keys = set(inputs.keys())
    param_set = set(params)
    if keys == param_set:
        return json.dumps(inputs, separators=(",", ":"))
    if len(params) == 1:
        sole = params[0]
        if sole not in keys:
            return json.dumps({sole: inputs}, separators=(",", ":"))
    return json.dumps(inputs, separators=(",", ":"))


def normalize_planning_contracts(planning: dict) -> dict:
    """Apply deterministic I/O alignment and unique file paths to planning contracts."""
    raw = planning.get("contracts") or []
    if not raw:
        return planning
    contracts = [Contract(**c) if isinstance(c, dict) else c for c in raw]
    for contract in contracts:
        contract.example_inputs_json = _align_kwargs_json(
            contract.signature, contract.example_inputs_json
        )
        if contract.failure_inputs_json:
            contract.failure_inputs_json = _align_kwargs_json(
                contract.signature, contract.failure_inputs_json
            )
    _dedupe_file_paths(contracts)
    return {**planning, "contracts": [c.model_dump() for c in contracts]}


def _dedupe_file_paths(contracts: list[Contract]) -> None:
    """Ensure every contract has a unique file_path (same func name across FRs is common)."""
    assigned: set[str] = set()
    for contract in contracts:
        candidate = contract.file_path
        if candidate not in assigned:
            assigned.add(candidate)
            continue
        layer_dir = _LAYER_DIR[contract.mvc_layer]
        stem = _func_name(contract.signature, contract.fr_id)
        candidate = f"{layer_dir}/{stem}_{_slug(contract.fr_id)}.py"
        while candidate in assigned:
            candidate = f"{layer_dir}/{stem}_{_slug(contract.fr_id)}_{len(assigned)}.py"
        contract.file_path = candidate
        assigned.add(candidate)


def _infer_return_type(signature: str) -> str | None:
    """Extract the return type annotation from a function signature.

    E.g. 'def foo(x: int) -> None:' yields 'None', 'def bar() -> dict[str, Any]:' yields 'dict[str, Any]'.
    Returns None if no return type found.
    """
    if not signature:
        return None
    match = re.search(r'->\s*([^:]+):', signature)
    if match:
        return match.group(1).strip()
    return None


class PlanningAgent:
    def __init__(self, llm=None, stack: str = "python", prompt_strategy: str = "baseline") -> None:
        # _design_feature always runs through the deepagents driver (see agents/deep/planning.py):
        # the agent reads each FR's spec sheet and submits FRs ONE AT A TIME through
        # `submit_fr_plan`, so a rejected field comes back as a tool error naming that one FR
        # instead of the whole feature's structured output failing at once. Planning has no
        # isolation requirement — its output is the shared read-only contract every downstream
        # agent binds to.
        # `llm` is injectable so tests/stubs avoid network calls. NOTE: _design_feature does not
        # use it (the deep driver builds its own agent); the other LLM steps in plan() do.
        self.llm = llm if llm is not None else get_llm()
        self.renderer = PromptRenderer()
        # 'baseline' = original prompt; 'cot' = the parallel reason-first variant.
        self.prompt_strategy = prompt_strategy
        # Target stack for the run (e.g. "fastapi"). Threaded into the design prompt so
        # signatures/layers come out shaped for that stack. The stack lives in the run,
        # not hardcoded in the prompt prose, so other profiles can be added later.
        self.stack = stack

    def plan(self, ir: dict) -> PlanningOutput:
        normalize_ir_features(ir)
        graph = ir.get("dependency_graph")
        if graph is None:
            raise ValueError("PlanningAgent requires an IR enriched with 'dependency_graph'.")

        by_id = {feature_id_of(f): f for f in ir.get("features", []) if feature_id_of(f)}
        behavioral_by_fr = {c.get("fr_id"): c for c in ir.get("contracts", [])}
        contracts: list[Contract] = []
        notes: list[str] = []

        for fid in self._feature_order(ir, graph):
            feature = by_id.get(fid)
            if not feature or not feature.get("functional_requirements"):
                continue  # empty features yield no contracts

            frs = feature["functional_requirements"]
            fr_order = feature.get("fr_order") or [r["id"] for r in frs]
            text_by_id = {r["id"]: requirement_text(r) for r in frs}
            name = feature.get("name", "")
            feature_contracts = {rid: behavioral_by_fr.get(rid) for rid in fr_order}

            prereqs: dict[str, list[str]] = defaultdict(list)
            for edge in feature.get("fr_edges", []):
                prereqs[edge["source"]].append(edge["target"])

            designs = self._design_feature(name, fr_order, text_by_id, feature_contracts)

            for rid in fr_order:
                design = designs.get(rid)
                prereq_ids = prereqs.get(rid, [])
                prereq_notes = self._prereq_notes(prereq_ids, designs)
                bc_dict = behavioral_by_fr.get(rid)
                bc = BehavioralContract(**bc_dict) if bc_dict else None

                if design is None:
                    notes.append(f"FR {rid}: no LLM design returned; used a default contract.")
                    signature = f"def fr_{_slug(rid)}() -> None:"
                    args, returns = [], ""
                    layer = _normalize_layer("")
                    example_inputs = "{}"
                    expected_return = "null"
                    error_mode = "raise"
                    failure_inputs = ""
                    entity_identifier = ""
                else:
                    signature = design.signature
                    args, returns = design.args, design.returns
                    layer = _normalize_layer(design.mvc_layer)
                    example_inputs = design.example_inputs_json
                    expected_return = design.expected_return_json
                    error_mode = design.error_mode
                    failure_inputs = design.failure_inputs_json or ""
                    entity_identifier = getattr(design, "entity_identifier", "") or ""

                example_inputs = _align_kwargs_json(signature, example_inputs)
                if failure_inputs:
                    failure_inputs = _align_kwargs_json(signature, failure_inputs)

                contracts.append(
                    Contract(
                        fr_id=rid,
                        feature_id=fid,
                        signature=signature,
                        docstring=self._compose_docstring(
                            bc, args, returns, prereq_notes, signature, fallback_id=rid,
                            stack=self.stack,
                        ),
                        mvc_layer=layer,
                        file_path=_file_path(layer, signature, rid),
                        prerequisite_fr_ids=prereq_ids,
                        contract=bc,
                        example_inputs_json=example_inputs,
                        expected_return_json=expected_return,
                        error_mode=error_mode,
                        failure_inputs_json=failure_inputs,
                        entity_identifier=entity_identifier,
                    )
                )

        _dedupe_file_paths(contracts)
        return PlanningOutput(contracts=contracts, notes=notes)

    @staticmethod
    def normalize_ir_planning(ir: dict) -> dict:
        """Normalize planning contracts in an IR dict in place and return the IR."""
        planning = ir.get("planning")
        if not planning:
            return ir
        ir["planning"] = normalize_planning_contracts(planning)
        return ir

    def enrich(self, ir: dict, output_dir=None) -> dict:
        """Run plan() and fold the result into the IR under 'planning'. Optionally save."""
        planning = self.plan(ir).model_dump()
        enriched = {**ir, "planning": planning}
        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{ir.get('project_name', 'project')}_planning.json").write_text(
                json.dumps(enriched, indent=2)
            )
        return enriched

    # --- helpers ---------------------------------------------------------------
    def _design_feature(self, feature_name, fr_order, text_by_id, contracts_by_fr) -> dict:
        """Design a single feature's FRs with the planning agent. Returns {fr_id: FRPlan}.

        Each requirement is seeded as its own spec sheet — its text plus its behavioral
        contract — so the agent can shape a faithful signature, layer, and per-arg/return docs,
        and submit the FRs one at a time through `submit_fr_plan`.
        """
        from src.cleanroom.agents.deep.planning import deep_design_feature  # noqa: PLC0415

        return deep_design_feature(feature_name, list(fr_order), text_by_id, contracts_by_fr,
                                   stack=self.stack)

    @staticmethod
    def _prereq_notes(prereq_ids: list[str], designs: dict) -> list[str]:
        """Build the docstring's prerequisite Notes deterministically from fr_edges.

        References each prerequisite by its designed function name (when available) plus
        its requirement id, e.g. "Prerequisite: create_lab_test_request (req 2.2.6.6) ...".
        """
        notes: list[str] = []
        for pid in prereq_ids:
            dep = designs.get(pid)
            fname = _func_name(dep.signature, pid) if dep is not None else None
            if fname:
                notes.append(f"Prerequisite: {fname} (req {pid}) must have run first.")
            else:
                notes.append(f"Prerequisite: requirement {pid} must be satisfied first.")
        return notes

    @staticmethod
    def _compose_docstring(
        contract, args, returns: str, prereq_notes: list[str],
        signature: str = "", fallback_id: str = "", stack: str = "python"
    ) -> str:
        """Assemble an explicit Google-style (Napoleon) docstring deterministically.

        Reference: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

        The summary and behavioral guarantees come from the spec-derived behavioral
        contract; the LLM supplies only the per-arg/return descriptions that bind to the
        signature it authored. The whole section structure is plain code. Only recognized
        Napoleon section headers are used (Args, Returns, Raises, Notes) so it renders cleanly;
        the contract's pre/post/invariant guarantees live as labeled lines under Notes.
        """
        summary = PlanningAgent._contract_summary(contract, fallback_id)
        lines = [summary]

        if args:
            lines += ["", "Args:"]
            for arg in args:
                lines.append(f"    {arg.name}: {arg.description}")

        return_text = (returns or "").strip()
        if not return_text and signature:
            return_type = _infer_return_type(signature)
            if return_type:
                return_text = return_type + "."
        if return_text:
            lines += ["", "Returns:", f"    {return_text}"]

        precondition = _contract_field(contract, "precondition")
        if precondition:
            exc = "HTTPException" if stack == "fastapi" else "ValueError"
            lines += ["", "Raises:",
                      f"    {exc}: If the precondition is violated ({precondition})."]

        notes: list[str] = []
        if precondition:
            notes.append(f"Precondition: {precondition}")
        postcondition = _contract_field(contract, "postcondition")
        if postcondition:
            notes.append(f"Postcondition: {postcondition}")
        notes += prereq_notes

        if notes:
            lines += ["", "Notes:"]
            for note in notes:
                lines.append(f"    {note}")

        return "\n".join(lines).strip()

    @staticmethod
    def _contract_summary(contract, fallback_id: str) -> str:
        """One-line summary grounded in the contract's response (else stimulus)."""
        response = _contract_field(contract, "response")
        if response:
            return response if response.endswith(".") else response + "."
        stimulus = _contract_field(contract, "stimulus")
        if stimulus:
            summary = f"Handle {stimulus[0].lower() + stimulus[1:]}"
            return summary if summary.endswith(".") else summary + "."
        return f"Implements requirement {fallback_id}." if fallback_id else "Implements the requirement."

    @staticmethod
    def _feature_order(ir: dict, graph: dict) -> list[str]:
        """Outer feature order: dependency build order, then cycle members, then any
        FR-bearing feature not yet covered (numeric) so nothing is dropped."""
        ordered = list(graph.get("build_order", []))
        for cycle in graph.get("cycles", []):
            ordered += sorted(cycle, key=_numeric_key)
        seen = set(ordered)
        for feature in ir.get("features", []):
            fid = feature_id_of(feature)
            if feature.get("functional_requirements") and fid not in seen:
                ordered.append(fid)
                seen.add(fid)
        return ordered


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.agents.planning.agent <enriched_ir.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        enriched_ir = json.load(f)

    out = PlanningAgent().plan(enriched_ir)
    print(f"Contracts ({len(out.contracts)}):\n")
    for c in out.contracts:
        print(f"  [{c.fr_id}] ({c.mvc_layer}) {c.signature}")
        print(f"        -> {c.file_path}   prereqs={c.prerequisite_fr_ids}")
    if out.notes:
        print("\nNotes:")
        for note in out.notes:
            print(f"  - {note}")