"""Test Agent.

Derives black-box test cases from the SPECIFICATION, one feature at a time. One
LLM call per feature, each scoped to ONLY that feature's requirements — never the
whole IR.

==================  STRUCTURAL ISOLATION (the project's core)  ==================
This agent is the mirror image of the Code Agent: tests are a pure function of the
SPECIFICATION, and by construction the agent cannot see the implementation.

  * No public method takes a code-related parameter (no code, implementation,
    source, etc.). The only input is `ir`, and only its spec slice is read.
  * It never imports anything from src/cleanroom/agents/code/.
  * It never reads the 'generated_code' key, even though that key exists in the IR.
    All IR access goes through spec_reader.feature_units, which touches only
    ir['features'].
  * It GENERATES tests; it does not execute them against any implementation. Running
    tests against code is a separate, later certification stage.

Do not let this agent peek at the generated code to "make better tests" — that would
destroy the spec/code separation that is the whole point of the pipeline.
================================================================================
"""

import json
import sys
import time
from pathlib import Path

from src.cleanroom.agents.planning.agent import PlanningAgent
from src.cleanroom.agents.test.schema.tests import FeatureTests, GeneratedTests, TestSourceRepair
from src.cleanroom.agents.test.tools.spec_reader import feature_units
from src.cleanroom.targets import get_target
from src.cleanroom.utils.ir import normalize_ir_features
from src.cleanroom.utils.llm_client import get_llm
from src.cleanroom.utils.prompt_renderer import PromptRenderer, cot_template


_FEATURE_TESTS_OUTPUT_CONTRACT = """
Return exactly one structured object matching this shape:
{
  "feature_id": "string",
  "cases": [
    {
      "requirement_id": "string",
      "description": "string",
      "inputs": "string",
      "expected": "string",
      "inputs_json": "{}",
      "expected_json": "{}",
      "oracle": "eq",
      "setup_json": ""
    }
  ],
  "test_source": "full runnable test module source as one string"
}

Do not return prose, markdown, or explanations after the object. `oracle` must be exactly
"eq" or "raises". `inputs_json`, `expected_json`, and `setup_json` must be JSON strings.
"""


class TestAgent:
    # Keep pytest from collecting this class as a test (its name starts with "Test").
    __test__ = False

    def __init__(self, llm=None, stack: str = "python", language: str = "python",
                 gen_driver: str = "deterministic",
                 prompt_strategy: str = "baseline") -> None:
        # 'deepagent' routes generate() through agents/deep/generation.py. That driver's
        # filesystem holds contracts and its own test file only — never generated code.
        self.gen_driver = gen_driver
        # `llm` is injectable only so tests can avoid network calls; it is NOT a
        # channel for implementation data. Defaults to the shared cost-control model client.
        self.llm = llm if llm is not None else get_llm()
        self.renderer = PromptRenderer()
        # 'baseline' = original prompts; 'cot' = the parallel reason-first variants. CoT reasons
        # about coverage FROM THE SPEC ONLY — the agent still never sees generated code.
        self.prompt_strategy = prompt_strategy
        # Target stack (spec/run-level, NOT implementation): shapes the emitted pytest
        # module — plain function calls (python) vs TestClient HTTP requests (fastapi) —
        # and the failure oracle (ValueError vs HTTPException). Knowing the stack is not
        # seeing code, so isolation is preserved, exactly as in the Code/Planning agents.
        self.stack = stack
        # The language target picks the test template + framework (pytest vs JUnit).
        self.language = language
        self.target = get_target(language, stack)

    def generate(self, ir: dict) -> GeneratedTests:
        """Generate tests for every feature from the spec only. Input is the spec."""
        normalize_ir_features(ir)
        PlanningAgent.normalize_ir_planning(ir)
        features: list[FeatureTests] = []

        deep = getattr(self, "gen_driver", "deterministic") == "deepagent"
        contracts_by_feature: dict[str, list[dict]] = {}
        if deep:
            # The deep test agent is briefed from the planner's CONTRACTS, exactly as the code
            # and proof agents are, so all three derive from byte-identical sheets. Its
            # filesystem never contains generated code — see agents/deep/generation.py.
            from src.cleanroom.agents.deep.generation import deep_generate_tests  # noqa: PLC0415

            for c in (ir.get("planning") or {}).get("contracts", []) or []:
                contracts_by_feature.setdefault(str(c["feature_id"]), []).append(c)

        for unit in feature_units(ir):  # the ONLY IR read: features/requirements
            if deep:
                feature_id = str(unit["feature_id"])
                result, _metrics = deep_generate_tests(
                    ir, feature_id, contracts_by_feature.get(feature_id, []),
                    language=self.language)
                if result is not None:
                    features.append(result)
                continue

            prompt = self.renderer.render(
                cot_template(self.target.test_template(), self.prompt_strategy),
                {
                    "feature_id": unit["feature_id"],
                    "name": unit["name"],
                    "description": unit["description"],
                    "requirements": unit["requirements"],
                    "stack": self.stack,
                },
            )

            prompt = f"{prompt}\n\n{_FEATURE_TESTS_OUTPUT_CONTRACT}"

            # One structured call per feature. Schema via with_structured_output, reinforced in the
            # prompt for weaker models that sometimes answer in prose instead of tool args. The model
            # returns both the structured `cases` oracle and a runnable `test_source` module.
            result = None
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    result = self.llm.with_structured_output(FeatureTests).invoke(prompt)
                except Exception as exc:
                    last_error = exc
                if result is not None and getattr(result, "cases", None) is not None:
                    break
                if attempt < 3:
                    time.sleep(float(attempt))
            else:
                if last_error is not None:
                    raise RuntimeError(
                        f"test generation failed for feature {unit['feature_id']}"
                    ) from last_error
                raise RuntimeError(
                    f"test generation returned no structured FeatureTests for feature {unit['feature_id']}"
                )

            # The spec owns feature_id; don't trust the LLM to echo it correctly.
            features.append(
                FeatureTests(
                    feature_id=unit["feature_id"],
                    cases=list(result.cases),
                    test_source=result.test_source,
                )
            )

        return GeneratedTests(features=features)

    def run(self, ir: dict, output_dir: Path = Path("outputs/generated")) -> GeneratedTests:
        """Generate tests from the spec and write them out as runnable pytest modules."""
        tests = self.generate(ir)
        self.write_files(tests, output_dir / ir.get("project_name", "project") / "tests")
        return tests

    def repair_compile_error(self, ir: dict, feature_test: dict, diagnostics: str) -> FeatureTests:
        """Repair one generated Java test source using compiler diagnostics only.

        This preserves the structured black-box cases exactly; only the runnable-style JUnit source
        is rewritten so the static Java compile check can include tests.
        """
        normalize_ir_features(ir)
        PlanningAgent.normalize_ir_planning(ir)
        feature_id = str(feature_test.get("feature_id", ""))
        unit = next((u for u in feature_units(ir) if str(u.get("feature_id")) == feature_id), None)
        if unit is None:
            raise ValueError(f"no feature spec found for generated test feature {feature_id}")

        result: TestSourceRepair = self.llm.with_structured_output(TestSourceRepair).invoke(
            self.renderer.render(
                cot_template("repair_compile_errors_java_tests.j2", self.prompt_strategy),
                {
                    "feature_id": feature_id,
                    "name": unit.get("name", ""),
                    "description": unit.get("description", ""),
                    "requirements": unit.get("requirements", []),
                    "stack": self.stack,
                    "diagnostics": diagnostics,
                    "prior_source": feature_test.get("test_source", ""),
                },
            )
        )
        return FeatureTests(
            feature_id=feature_id,
            cases=list(feature_test.get("cases", [])),
            test_source=result.test_source,
        )

    @staticmethod
    def write_files(tests: GeneratedTests, output_dir: Path, language: str = "python") -> list[Path]:
        """Write one runnable-style test module per feature under output_dir (pytest for python,
        a JUnit class for java).

        Pure persistence of what the agent already produced — this writer reads only the
        spec-derived `tests`, never any implementation. The `test_source` field holds the test
        module source (python or, for java, a JUnit class). Because the agent cannot see the code,
        import/class names follow conventions and may need binding adjustment — the structured
        `cases` remain the canonical, code-independent oracle (see schema/tests.py).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for feature in tests.features:
            slug = feature.feature_id.replace(".", "_")
            source = feature.test_source.strip()
            if language == "java":
                from src.cleanroom.utils.java_packager import java_class_name
                if not source:
                    source = (f"// No JUnit source generated for feature {feature.feature_id}; "
                              f"{len(feature.cases)} case(s) recorded in the IR.\n")
                cls = java_class_name(source, f"Feature_{slug}Test")
                dest = output_dir / f"{cls}.java"
                dest.write_text(source + "\n")
            elif language == "javascript":
                if not source:
                    source = (f"// No Jest source generated for feature {feature.feature_id}; "
                              f"{len(feature.cases)} case(s) recorded in the IR.\n")
                dest = output_dir / f"feature_{slug}.test.js"
                dest.write_text(source + "\n")
            else:
                if not source:
                    source = (
                        f"# No pytest source was generated for feature {feature.feature_id}.\n"
                        f"# {len(feature.cases)} black-box case(s) are recorded in the IR "
                        f"under generated_tests.\n"
                    )
                header = (
                    f'"""Spec-derived pytest module for feature {feature.feature_id}.\n\n'
                    f"Generated black-box from the specification ({len(feature.cases)} case(s)); "
                    "the Test Agent never sees the implementation, so import paths follow\n"
                    "sensible module conventions and may need adjustment to bind to the concrete\n"
                    "generated package.\n"
                    '"""\n'
                )
                dest = output_dir / f"test_feature_{slug}.py"
                dest.write_text(header + "\n" + source + "\n")
            written.append(dest)
        return written


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.agents.test.agent <enriched_ir.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as fh:
        enriched_ir = json.load(fh)

    generated_tests = TestAgent().run(enriched_ir)
    total = 0
    for feature in generated_tests.features:
        print(f"Feature {feature.feature_id}: {len(feature.cases)} test case(s)")
        total += len(feature.cases)
    print(f"\n{len(generated_tests.features)} features, {total} test cases total.")
    print(f"pytest modules (test_feature_*.py) written under "
          f"outputs/generated/{enriched_ir.get('project_name', 'project')}/tests/")
