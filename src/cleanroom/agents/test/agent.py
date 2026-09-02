"""Test Agent.

Derives black-box test cases from the SPECIFICATION, one feature at a time. Generation is
driven by a deepagents agent (``agents/deep/generation.py``): one agent per feature, briefed
with that feature's per-FR contract sheets in its own virtual filesystem, recording each case
through ``submit_test_case`` and the runnable module through ``submit_test_source``. There is
no single-shot "deterministic" driver any more — the agent path is the only path.

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
  * The generating agent runs on a StateBackend seeded with contract sheets and its own
    /tests pool only, so no implementation exists in its filesystem to be found.

Do not let this agent peek at the generated code to "make better tests" — that would
destroy the spec/code separation that is the whole point of the pipeline.
================================================================================
"""

import json
import sys
from pathlib import Path

from src.cleanroom.agents.planning.agent import PlanningAgent
from src.cleanroom.agents.test.schema.tests import FeatureTests, GeneratedTests, TestSourceRepair
from src.cleanroom.agents.test.tools.spec_reader import feature_units
from src.cleanroom.targets import get_target
from src.cleanroom.utils.ir import normalize_ir_features
from src.cleanroom.utils.llm_client import get_llm
from src.cleanroom.utils.prompt_renderer import PromptRenderer


class TestAgent:
    # Keep pytest from collecting this class as a test (its name starts with "Test").
    __test__ = False

    def __init__(self, llm=None, stack: str = "python", language: str = "python") -> None:
        # `llm` is injectable only so tests can avoid network calls; it is NOT a
        # channel for implementation data. Defaults to the shared cost-control model client.
        # NOTE: generate() does NOT use this client — the deep driver builds its own agent on
        # get_llm(). It still backs repair_compile_error below.
        self.llm = llm if llm is not None else get_llm()
        self.renderer = PromptRenderer()
        # Target stack (spec/run-level, NOT implementation): shapes the emitted pytest
        # module — plain function calls (python) vs TestClient HTTP requests (fastapi) —
        # and the failure oracle (ValueError vs HTTPException). Knowing the stack is not
        # seeing code, so isolation is preserved, exactly as in the Code/Planning agents.
        self.stack = stack
        # The language target picks the test template + framework (pytest vs JUnit).
        self.language = language
        self.target = get_target(language, stack)

    def generate(self, ir: dict) -> GeneratedTests:
        """Generate tests for every feature from the spec only. Input is the spec.

        One deepagents agent per feature, briefed from the planner's CONTRACTS — exactly the
        sheets the code and proof agents get, so all three derive from byte-identical input.
        The agent's filesystem never contains generated code; see agents/deep/generation.py.
        """
        from src.cleanroom.agents.deep.generation import deep_generate_tests  # noqa: PLC0415

        normalize_ir_features(ir)
        PlanningAgent.normalize_ir_planning(ir)

        contracts_by_feature: dict[str, list[dict]] = {}
        for c in (ir.get("planning") or {}).get("contracts", []) or []:
            contracts_by_feature.setdefault(str(c["feature_id"]), []).append(c)

        features: list[FeatureTests] = []
        for unit in feature_units(ir):  # the ONLY IR read: features/requirements
            feature_id = str(unit["feature_id"])
            result, _metrics = deep_generate_tests(
                ir, feature_id, contracts_by_feature.get(feature_id, []),
                language=self.language)
            if result is not None:
                features.append(result)

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
                "repair_compile_errors_java_tests.j2",
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
