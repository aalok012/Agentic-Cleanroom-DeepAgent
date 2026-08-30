"""Code Agent.

Generates MVC source code from the specification, walking the planner's per-FR
CONTRACTS one at a time. One LLM call per contract, each scoped to ONLY that
requirement's contract (signature + docstring + spec text) plus the *signatures* of
its prerequisite contracts — never the whole IR, never any test artifact.

==================  STRUCTURAL ISOLATION (the project's core)  ==================
This agent generates code as a pure function of the SPECIFICATION. By construction
it cannot receive anything from testing:

  * No public method takes a test-related parameter (no test_results, failures,
    feedback, expected outputs, etc.). The only input is `ir` (the spec + the
    spec-derived planning contracts).
  * There is no retry/debugging loop that regenerates code from test outcomes.
  * It never imports the test agent and never reads `generated_tests`.

Do not add a feedback path here, even to "improve" results — that isolation is the
whole point of the pipeline.
================================================================================
"""

import json
import re
import sys
from pathlib import Path

from src.cleanroom.agents.code.schema.code import FileImplementation, GeneratedCode, GeneratedFile
from src.cleanroom.agents.planning.agent import PlanningAgent, _slug
from src.cleanroom.utils.contracts import prereq_ifaces, requirement_text
from src.cleanroom.utils.ir import normalize_ir_features
from src.cleanroom.targets import get_target
from src.cleanroom.utils.llm_client import get_llm
from src.cleanroom.utils.prompt_renderer import PromptRenderer, cot_template


def _route_from_path(file_path: str) -> str:
    """Deterministic REST route stem for a contract's file_path (Spring stack).

    The planner writes file_path as ``{layer}/{func}.py``; the route mirrors the FastAPI
    packager's ``/{layer}/{module_stem}`` convention so a sample's endpoints are predictable.
    """
    p = (file_path or "").strip("/")
    if p.endswith(".py"):
        p = p[:-3]
    return p


def _strip_java_package_imports(content: str, package_names: list[str]) -> str:
    """Remove invalid Java imports of package names emitted by Dafny translation.

    Java can import a class from a package, but not the package itself. Dafny Java packages like
    ``F4__9Domain`` are referenced with fully-qualified class names such as
    ``F4__9Domain.__default`` and ``F4__9Domain.Action``.
    """
    out = content or ""
    for package in package_names:
        if not package:
            continue
        out = re.sub(rf"(?m)^\s*import\s+{re.escape(package)}\s*;\s*\n?", "", out)
    return out


def _extract_code_block(text: str) -> str:
    """Pull source out of a free-text model reply. Prefer the largest fenced ```...``` block (the
    implementation, leaving any reasoning prose outside the fence); fall back to the whole stripped
    text when there is no fence. Used only when structured output fails to parse."""
    blocks = re.findall(r"```[a-zA-Z0-9_+\-]*\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return max((b.strip() for b in blocks), key=len)
    return text.strip()


class CodeAgent:
    def __init__(self, llm=None, stack: str = "python", language: str = "python",
                 prompt_strategy: str = "baseline", gen_driver: str = "deterministic") -> None:
        # 'deepagent' routes generate() through agents/deep/generation.py. The deep driver's
        # filesystem is seeded with contracts only — never tests — so isolation is unchanged.
        self.gen_driver = gen_driver
        # `llm` is injectable purely so tests can avoid network calls; it is NOT a
        # channel for test data. Defaults to the shared cost-control model client.
        self.llm = llm if llm is not None else get_llm()
        self.renderer = PromptRenderer()
        # 'baseline' = original prompts; 'cot' = the parallel reason-first variants. CoT changes
        # only the prompt wording (reason from the SPEC first) — the clean-room isolation is intact.
        self.prompt_strategy = prompt_strategy
        # Target stack for the run (e.g. "fastapi"). Threaded into the code prompt as
        # STRUCTURAL conventions only — never test-derived — so isolation/pass@k hold.
        self.stack = stack
        # The language target picks the codegen template (python vs java); knowing the target
        # language is structural, not seeing tests — isolation holds.
        self.language = language
        self.target = get_target(language, stack)

    def generate(self, ir: dict, skip_feature_ids: set[str] | None = None) -> GeneratedCode:
        """Generate code for every contract, in planning (dependency) order.

        Input is the spec + the spec-derived contracts only. The planner already owns
        the signature, docstring, mvc_layer, file_path and prerequisite ids of each FR;
        this agent only turns each contract into a concrete file body.

        ``skip_feature_ids`` omits whole features whose logic ships from elsewhere (the Dafny
        proof tier): those features get an adapter via :meth:`generate_adapter` instead.
        """
        contracts = (ir.get("planning") or {}).get("contracts")
        if not contracts:
            raise ValueError("CodeAgent requires an IR enriched with 'planning.contracts'.")
        PlanningAgent.normalize_ir_planning(ir)
        normalize_ir_features(ir)
        contracts = ir["planning"]["contracts"]

        skip = set(skip_feature_ids or ())
        if getattr(self, "gen_driver", "deterministic") == "deepagent":
            return self._generate_deep(ir, contracts, skip)

        req_text = self._requirement_index(ir)        # fr_id -> spec requirement text
        by_fr = {c["fr_id"]: c for c in contracts}     # fr_id -> contract (for prereq signatures)
        files: list[GeneratedFile] = []

        for contract in contracts:                     # already in dependency order
            if contract["feature_id"] in skip:
                continue
            fr_id = contract["fr_id"]

            # (a) ONLY this FR's contract + its spec text.
            # (b) ONLY the signatures of prerequisite contracts — never their bodies,
            #     and they are taken from the planner's contracts, not from any generated
            #     code, so the input here is purely spec-derived.
            prerequisites = prereq_ifaces(contract, by_fr)

            prompt = self.renderer.render(
                cot_template(self.target.code_template(), self.prompt_strategy),
                {
                    "fr_id": fr_id,
                    "feature_id": contract["feature_id"],
                    "mvc_layer": contract["mvc_layer"],
                    "signature": contract["signature"],
                    "docstring": contract["docstring"],
                    "requirement": req_text.get(fr_id, ""),
                    "prerequisites": prerequisites,
                    "stack": self.stack,
                    "route": _route_from_path(contract["file_path"]),
                    "example_inputs_json": contract.get("example_inputs_json", "{}"),
                    "expected_return_json": contract.get("expected_return_json", "null"),
                    "error_mode": contract.get("error_mode", "raise"),
                    "failure_inputs_json": contract.get("failure_inputs_json", ""),
                    "entity_identifier": contract.get("entity_identifier", ""),
                },
            )

            # One structured call per contract. Schema via with_structured_output —
            # no JSON schema in the prompt.
            # Resilience: the structured-output coercion returns None when a reply is unparseable —
            # e.g. a verbose CoT/MoT reasoning prompt makes the model answer in prose + a fenced
            # code block instead of a clean tool call. That is deterministic at temperature 0, so a
            # plain retry would just reproduce it; instead fall back to a plain (non-structured)
            # call and extract the source from the reply (a text reply IS the file content).
            structured = self.llm.with_structured_output(FileImplementation)
            impl: FileImplementation | None = structured.invoke(prompt)
            if impl is None:
                raw = self.llm.invoke(prompt)
                raw_text = raw.content if isinstance(raw.content, str) else str(raw.content)
                code = _extract_code_block(raw_text)
                impl = FileImplementation(content=code) if code.strip() else None
            if impl is None:
                raise RuntimeError(
                    f"code generation returned no parseable implementation for FR {fr_id}")

            # The planner owns ids/path/layer; the LLM only supplies the body.
            files.append(
                GeneratedFile(
                    fr_id=fr_id,
                    feature_id=contract["feature_id"],
                    path=contract["file_path"],
                    mvc_layer=contract["mvc_layer"],
                    content=impl.content,
                )
            )

        return GeneratedCode(files=files)

    def _generate_deep(self, ir: dict, contracts: list[dict], skip: set[str]) -> GeneratedCode:
        """deepagents counterpart to :meth:`generate`, one agent per feature.

        Per feature rather than per FR (as the deterministic loop does) or all at once: FRs in
        a feature share data shapes, so the agent can cross-read them and keep them consistent,
        while a whole-IR invocation would blow past the context window on a large SRS.
        Features keep their dependency order, so a later feature is still generated after the
        ones it depends on.
        """
        from src.cleanroom.agents.deep.generation import deep_generate_code  # noqa: PLC0415

        by_feature: dict[str, list[dict]] = {}
        for contract in contracts:                     # already in dependency order
            if contract["feature_id"] in skip:
                continue
            by_feature.setdefault(contract["feature_id"], []).append(contract)

        files: list[GeneratedFile] = []
        for feature_contracts in by_feature.values():
            produced, _metrics = deep_generate_code(
                ir, feature_contracts, language=self.language)
            files.extend(produced)
        return GeneratedCode(files=files)

    def generate_adapter(
        self,
        ir: dict,
        feature_id: str,
        module: str,
        dafny_source: str,
        java_api: dict | None = None,
    ) -> GeneratedFile:
        """Write the thin FastAPI glue over a PROVED Dafny core (no business logic here).

        For a feature the proof tier verified, the logic ships as the compiled Dafny module; this
        emits ONE controller file with a route per FR that imports the compiled core, loads/saves
        the state in the DB, marshals request<->Dafny via ``dafny_marshal``, and calls the proved
        ``Normalize(Apply(state, action))``. Still spec-only + the agent's own proved Dafny — it
        never reads tests, so isolation holds.

        ``module`` is the Dafny module base (e.g. ``F4_1``): the compiled core dir is
        ``<module>-py`` and the importable core module is ``<module>Domain``.
        """
        PlanningAgent.normalize_ir_planning(ir)
        normalize_ir_features(ir)
        contracts = [c for c in ir["planning"]["contracts"] if c["feature_id"] == feature_id]
        if not contracts:
            raise ValueError(f"no contracts for feature {feature_id}")
        req_text = self._requirement_index(ir)
        feat = next((f for f in ir.get("features", []) if str(f.get("id")) == str(feature_id)), {})
        java_api = java_api or {}
        core_java_domain_package = java_api.get("domain_package") or f"{module}Domain".replace("_", "__")
        core_java_kernel_package = java_api.get("kernel_package") or f"{module}Kernel".replace("_", "__")

        prompt = self.renderer.render(
            cot_template(self.target.adapter_template(), self.prompt_strategy),
            {
                "feature_id": feature_id,
                "feature_name": feat.get("name", ""),
                # Dafny's Python backend mangles `_` -> `__` in module names, so the Dafny module
                # `F4_1Domain` compiles to the importable Python module `F4__1Domain`.
                "core_module": f"{module}Domain".replace("_", "__"),
                "core_dir": f"{module}-py",              # compiled core directory (on sys.path, NOT mangled)
                # Java/Spring path: Dafny's Java backend mangles `_` -> `__` in package names,
                # so module F4_9 emits packages like F4__9Domain and F4__9Kernel. The top-level
                # F4_9.java file is in the default package and is not importable from Spring's
                # named controller packages.
                "module": module,
                "core_java_package": core_java_domain_package,
                "core_java_domain_package": core_java_domain_package,
                "core_java_kernel_package": core_java_kernel_package,
                "core_java_api": java_api,
                "core_java_api_summary": java_api.get("summary", ""),
                "dafny_source": dafny_source,
                "contracts": [
                    {
                        "fr_id": c["fr_id"],
                        "signature": c.get("signature", ""),
                        "file_path": c.get("file_path", ""),
                        "route": _route_from_path(c.get("file_path", "")),
                        "requirement": req_text.get(c["fr_id"], ""),
                        "contract": c.get("contract") or {},
                        "entity_identifier": c.get("entity_identifier", ""),
                    }
                    for c in contracts
                ],
            },
        )
        # Same intermittent unparseable-reply fallback as generate(): on None, a plain call returns
        # the source as text, which we extract and wrap.
        structured = self.llm.with_structured_output(FileImplementation)
        impl: FileImplementation | None = structured.invoke(prompt)
        if impl is None:
            raw = self.llm.invoke(prompt)
            raw_text = raw.content if isinstance(raw.content, str) else str(raw.content)
            code = _extract_code_block(raw_text)
            impl = FileImplementation(content=code) if code.strip() else None
        if impl is None:
            raise RuntimeError(
                f"adapter generation returned no parseable implementation for feature {feature_id}")
        path = self.target.adapter_file_path(feature_id)
        content = impl.content
        if self.language == "java" and self.stack == "spring":
            content = _strip_java_package_imports(
                content,
                [
                    module,
                    core_java_domain_package,
                    core_java_kernel_package,
                ],
            )
        return GeneratedFile(
            fr_id=contracts[0]["fr_id"],
            feature_id=feature_id,
            path=path,
            mvc_layer="controller",
            content=content,
        )

    # ===================  COMPILE-INFORMED REPAIR (JAVA ONLY)  ===================
    # This repair path receives compiler diagnostics (javac/Maven) only. It does not receive test
    # cases, expected outputs, or test assertions, so it preserves the clean-room separation while
    # enforcing that generated Java is syntactically and structurally buildable before certification.
    # =============================================================================
    def repair_compile_error(
        self,
        ir: dict,
        generated_file: dict,
        diagnostics: str,
        *,
        proved_modules: dict[str, str] | None = None,
        proved_java_apis: dict[str, dict] | None = None,
    ) -> GeneratedFile:
        """Repair one generated Java file using compiler diagnostics only."""
        PlanningAgent.normalize_ir_planning(ir)
        normalize_ir_features(ir)
        contracts = (ir.get("planning") or {}).get("contracts") or []
        by_fr = {c["fr_id"]: c for c in contracts}
        req_text = self._requirement_index(ir)
        fr_id = generated_file.get("fr_id", "")
        feature_id = generated_file.get("feature_id", "")
        contract = by_fr.get(fr_id, {})
        proved_modules = proved_modules or {}
        proved_java_apis = proved_java_apis or {}
        module = proved_modules.get(feature_id, "")
        java_api = proved_java_apis.get(feature_id, {})
        core_java_domain_package = (
            java_api.get("domain_package")
            or (f"{module}Domain".replace("_", "__") if module else "")
        )
        core_java_kernel_package = (
            java_api.get("kernel_package")
            or (f"{module}Kernel".replace("_", "__") if module else "")
        )
        prompt = self.renderer.render(
            cot_template("repair_compile_errors_java.j2", self.prompt_strategy),
            {
                "fr_id": fr_id,
                "feature_id": feature_id,
                "mvc_layer": generated_file.get("mvc_layer", ""),
                "path": generated_file.get("path", ""),
                "signature": contract.get("signature", ""),
                "docstring": contract.get("docstring", ""),
                "requirement": req_text.get(fr_id, ""),
                "route": _route_from_path(contract.get("file_path", generated_file.get("path", ""))),
                "example_inputs_json": contract.get("example_inputs_json", "{}"),
                "expected_return_json": contract.get("expected_return_json", "null"),
                "error_mode": contract.get("error_mode", "raise"),
                "failure_inputs_json": contract.get("failure_inputs_json", ""),
                "prior_content": generated_file.get("content", ""),
                "diagnostics": diagnostics,
                "stack": self.stack,
                "is_adapter": bool(module),
                "module": module,
                "core_java_domain_package": core_java_domain_package,
                "core_java_kernel_package": core_java_kernel_package,
                "core_java_api_summary": java_api.get("summary", ""),
            },
        )
        impl: FileImplementation = self.llm.with_structured_output(FileImplementation).invoke(prompt)
        content = impl.content
        if module and self.language == "java" and self.stack == "spring":
            content = _strip_java_package_imports(
                content,
                [
                    module,
                    core_java_domain_package,
                    core_java_kernel_package,
                ],
            )
        return GeneratedFile(
            fr_id=fr_id,
            feature_id=feature_id,
            path=generated_file.get("path", ""),
            mvc_layer=generated_file.get("mvc_layer", ""),
            content=content,
        )

    # ===================  TEST-INFORMED REPAIR (RECOVERY ONLY)  ===================
    # DELIBERATE, CONTAINED clean-room break, enabled by the user for the recovery loop:
    # this method — and ONLY this method — receives failing TEST CASES and regenerates code
    # to satisfy them. It is never called on the first pass (that stays clean-room via
    # `generate`); it runs only after a feature has already failed both proof and pass@1.
    # Features repaired here are labelled "TESTED (repaired-with-tests)" so the report never
    # claims a clean-room pass we did not earn. Do NOT call this from `generate`/`run`.
    # =============================================================================
    def regenerate_with_test_feedback(
        self, ir: dict, feature_ids: set[str], failures: list[dict]
    ) -> list[GeneratedFile]:
        """Regenerate the code of the given (failing) features WITH their failing test cases.

        ``failures`` is the certification result's per-case diagnostics
        (fr_id/inputs/expected/reason). For every contract whose feature is in ``feature_ids``
        we re-emit its file, feeding in that FR's failing cases. The Code Agent's temperature is
        set by the caller (the recovery loop escalates it per iteration). Returns the regenerated
        files only — the caller swaps them into ``generated_code`` by fr_id.
        """
        contracts = (ir.get("planning") or {}).get("contracts")
        if not contracts:
            raise ValueError("CodeAgent.regenerate_with_test_feedback requires 'planning.contracts'.")
        PlanningAgent.normalize_ir_planning(ir)
        normalize_ir_features(ir)
        contracts = ir["planning"]["contracts"]
        by_fr = {c["fr_id"]: c for c in contracts}
        req_text = self._requirement_index(ir)
        prior_by_fr = {f["fr_id"]: f["content"]
                       for f in (ir.get("generated_code") or {}).get("files", []) if f.get("fr_id")}

        fails_by_fr: dict[str, list[dict]] = {}
        for d in failures:
            fails_by_fr.setdefault(d.get("fr_id", ""), []).append({
                "description": d.get("description", ""),
                "inputs": d.get("inputs", ""),
                "expected": d.get("expected", ""),
                "reason": d.get("reason", ""),
            })

        feature_ids = set(feature_ids)
        out: list[GeneratedFile] = []
        for contract in contracts:
            if contract["feature_id"] not in feature_ids:
                continue
            fr_id = contract["fr_id"]
            prompt = self.renderer.render(
                cot_template(self.target.feedback_template(), self.prompt_strategy),
                {
                    "fr_id": fr_id,
                    "feature_id": contract["feature_id"],
                    "mvc_layer": contract["mvc_layer"],
                    "signature": contract["signature"],
                    "docstring": contract["docstring"],
                    "requirement": req_text.get(fr_id, ""),
                    "prerequisites": prereq_ifaces(contract, by_fr),
                    "prior_content": prior_by_fr.get(fr_id, ""),
                    "failing_cases": fails_by_fr.get(fr_id, []),
                    "stack": self.stack,
                    "route": _route_from_path(contract["file_path"]),
                    "example_inputs_json": contract.get("example_inputs_json", "{}"),
                    "expected_return_json": contract.get("expected_return_json", "null"),
                    "error_mode": contract.get("error_mode", "raise"),
                    "failure_inputs_json": contract.get("failure_inputs_json", ""),
                    "entity_identifier": contract.get("entity_identifier", ""),
                },
            )
            impl: FileImplementation = self.llm.with_structured_output(FileImplementation).invoke(prompt)
            out.append(GeneratedFile(
                fr_id=fr_id,
                feature_id=contract["feature_id"],
                path=contract["file_path"],
                mvc_layer=contract["mvc_layer"],
                content=impl.content,
            ))
        return out

    def run(self, ir: dict, output_dir: Path = Path("outputs/generated")) -> GeneratedCode:
        """Generate code and write it out, organized by MVC layer."""
        code = self.generate(ir)
        self.write_files(code, output_dir / ir.get("project_name", "project"))
        return code

    @staticmethod
    def write_files(code: GeneratedCode, output_dir: Path) -> list[Path]:
        """Write every generated file under output_dir, using the path from the contract.

        The planning agent sets file_path as '{layer_dir}/{func}.py' (e.g. 'models/func.py'),
        so the final destination is output_dir/models/func.py — matching the contract exactly.
        """
        written: list[Path] = []
        used: set[Path] = set()
        for f in code.files:
            dest = output_dir / f.path
            if dest in used:  # avoid clobbering a same-named file from another contract
                dest = dest.with_name(f"{dest.stem}_{_slug(f.fr_id)}{dest.suffix}")
                f.path = str(dest.relative_to(output_dir))
            used.add(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f.content)
            written.append(dest)
        return written

    # --- helpers ---------------------------------------------------------------
    @staticmethod
    def _requirement_index(ir: dict) -> dict[str, str]:
        """Map functional-requirement id -> its spec text, from the spec's features."""
        index: dict[str, str] = {}
        for feature in ir.get("features", []):
            for req in feature.get("functional_requirements", []):
                index[req["id"]] = requirement_text(req)
        return index


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.agents.code.agent <enriched_ir.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as fh:
        enriched_ir = json.load(fh)

    generated_code = CodeAgent().run(enriched_ir)
    for file in generated_code.files:
        print(f"  [{file.mvc_layer}] {file.fr_id} -> {file.path}  ({len(file.content)} chars)")
