"""The Spring Boot language target — Java + Spring Web (one web stack under language=java).

This is the Java analog of the Python ``fastapi`` stack: the Code Agent emits one self-contained
``@RestController`` per functional requirement, and a MECHANICAL packager assembles them into a
runnable Maven Spring Boot project. Spring's classpath **component scanning** is the exact analog
of the FastAPI router auto-discovery the Python packager relies on — drop ``@RestController``
classes anywhere under the base package and ``@SpringBootApplication`` registers them, so the
packager never writes per-feature wiring and the Code/Test isolation guarantee is preserved.

v1 scope: in-memory state (no JPA/DB), so each FR class stays self-contained and compiles cleanly;
the oracle is a Maven test-compile check that degrades gracefully when the toolchain is absent
(mirroring the plain-Java ``javac`` compile-check). Full MockMvc execution is a documented
follow-up.
"""

from __future__ import annotations

from pathlib import Path

from src.cleanroom.targets.java import JavaTarget


class SpringBootTarget(JavaTarget):
    language = "java"
    file_ext = ".java"
    test_framework = "junit5-spring"

    def adapter_template(self) -> str:
        # Thin Spring @RestController glue over a feature PROVED in Dafny and translated to Java.
        return "generate_adapter_spring.j2"

    def adapter_file_path(self, feature_id: str) -> str:
        return f"controllers/F{feature_id.replace('.', '_')}Adapter.java"

    def stage_cores(self, code_dir, project_dir, modules: list[str]) -> dict:
        from src.cleanroom.utils.dafny_project import stage_dafny_cores_java

        return stage_dafny_cores_java(Path(code_dir), Path(project_dir), modules)

    def oracle_name(self, stack: str) -> str:
        return "spring-build"

    def package_sample(self, generated_code: dict, code_dir: Path, stack: str) -> "Path | None":
        from src.cleanroom.utils.spring_packager import build_spring_project

        return build_spring_project(generated_code, code_dir)

    def package_tests(self, generated_tests: dict, code_dir: Path, stack: str) -> list[Path]:
        from src.cleanroom.utils.spring_packager import write_spring_tests

        return write_spring_tests(generated_tests, code_dir)

    def run_case(self, code_dir: Path, plan: dict, case: dict, stack: str, timeout: float):
        from src.cleanroom.agents.evaluation.runner import run_case_spring

        return run_case_spring(code_dir, plan, case, timeout=max(timeout, 180.0))

    def oracle_available(self) -> bool:
        from src.cleanroom.utils.maven_tooling import spring_oracle_available

        return spring_oracle_available()
