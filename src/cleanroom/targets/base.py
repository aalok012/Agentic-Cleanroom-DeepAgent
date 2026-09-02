"""The default (Python) language target.

The base class *is* the Python behavior (python | fastapi sub-stacks); `JavaTarget` overrides the
language-specific pieces. All imports of agent/packager/runner code are lazy (inside methods) to
avoid an import cycle — the agents import this module at top level.
"""

from __future__ import annotations

from pathlib import Path


class LanguageTarget:
    language = "python"
    file_ext = ".py"
    test_framework = "pytest"

    # --- which Jinja template each agent renders ---
    # Generation (code/test) has no template: those stages are agents that own their
    # own prompts in agents/deep/generation.py. What is left is the adapter and the
    # test-informed recovery regen, which are still single rendered calls.
    def adapter_template(self) -> str:
        """Thin glue over a PROVED Dafny core (FastAPI/Python path)."""
        return "generate_adapter.j2"

    def feedback_template(self) -> str:
        """Test-informed regeneration in the recovery loop (Python path)."""
        return "regenerate_with_feedback.j2"

    # --- Dafny-core adapter (proved features ship from Dafny + thin glue) ---
    def adapter_file_path(self, feature_id: str) -> str:
        """Path (relative to the code dir) for a feature's proved-core adapter file."""
        return f"controllers/f{feature_id.replace('.', '_')}_adapter.py"

    def stage_cores(self, app_dir: Path, project_dir: Path, modules: list[str]) -> dict:
        """Stage the compiled Dafny cores (+ any shim) into an assembled app so adapters resolve.

        Default = the FastAPI/Python path (compiled ``<module>-py`` dirs + ``dafny_marshal``)."""
        from src.cleanroom.utils.dafny_project import stage_dafny_cores

        return stage_dafny_cores(Path(app_dir), Path(project_dir), modules)

    # --- certification ---
    def oracle_name(self, stack: str) -> str:
        return "http" if stack == "fastapi" else "executable"

    def package_sample(self, generated_code: dict, code_dir: Path, stack: str) -> "Path | None":
        """Lay a code sample out for its oracle. Returns the assembled app dir (fastapi) or
        None (flat function tree). Mirrors the long-standing CertificationAgent._write_sample."""
        from src.cleanroom.agents.code.agent import CodeAgent
        from src.cleanroom.agents.code.schema.code import GeneratedCode
        from src.cleanroom.utils.packager import build_runnable_package

        if stack == "fastapi":
            return build_runnable_package(generated_code, code_dir)
        CodeAgent.write_files(GeneratedCode(**generated_code), code_dir)
        return None

    def package_tests(self, generated_tests: dict, code_dir: Path, stack: str) -> list[Path]:
        """Lay generated test source out where the target's compiler/build tool can see it."""
        return []

    def run_case(self, code_dir: Path, plan: dict, case: dict, stack: str, timeout: float):
        """Execute one spec-derived test case against a laid-out sample → (ok, reason)."""
        from src.cleanroom.agents.evaluation.runner import run_case, run_case_http

        if stack == "fastapi":
            return run_case_http(code_dir, plan.get("file_path", ""), case, timeout=max(timeout, 30.0))
        return run_case(code_dir, plan.get("file_path", ""), plan.get("signature", ""),
                        case, timeout=timeout)

    def oracle_available(self) -> bool:
        """Whether the executable oracle can run (Python always can)."""
        return True
