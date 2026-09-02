"""The Java language target — plain Java + JUnit5.

v1 scope: one stack (plain Java service classes, no web framework), JUnit5 tests, and an
executable oracle that compiles implementation + test sources with ``javac``. JUnit execution is
still deferred; the JUnit jar is required when generated test sources are present. The Dafny-core
*adapter* path stays Python-only for now.
"""

from __future__ import annotations

from pathlib import Path

from src.cleanroom.targets.base import LanguageTarget


class JavaTarget(LanguageTarget):
    language = "java"
    file_ext = ".java"
    test_framework = "junit5"

    def adapter_template(self) -> str:
        # Be explicit: don't silently fall back to the Python/FastAPI adapter prompt.
        raise NotImplementedError("Java Dafny-core adapter is not supported in v1 "
                                  "(adapter shipping is FastAPI-only).")

    def feedback_template(self) -> str:
        raise NotImplementedError("Java test-informed recovery is not supported in v1 "
                                  "(the recovery loop is Python-only).")

    def stage_cores(self, app_dir, project_dir, modules: list[str]) -> dict:
        # Plain Java has no adapter path (adapter_mode is never true for it), so nothing to stage.
        return {"staged": [], "missing": list(modules), "note": "plain-java stack has no adapter"}

    def oracle_name(self, stack: str) -> str:
        return "java"

    def package_sample(self, generated_code: dict, code_dir: Path, stack: str) -> "Path | None":
        from src.cleanroom.utils.java_packager import write_java_sources

        write_java_sources(generated_code, code_dir)
        return None

    def package_tests(self, generated_tests: dict, code_dir: Path, stack: str) -> list[Path]:
        from src.cleanroom.utils.java_packager import write_java_test_sources

        return write_java_test_sources(generated_tests, code_dir)

    def run_case(self, code_dir: Path, plan: dict, case: dict, stack: str, timeout: float):
        from src.cleanroom.agents.evaluation.runner import run_case_java

        return run_case_java(code_dir, plan, case, timeout=timeout)

    def oracle_available(self) -> bool:
        from src.cleanroom.utils.java_tooling import java_available

        return java_available()
