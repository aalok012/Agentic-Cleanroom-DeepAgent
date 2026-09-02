"""The JavaScript language target — Node.js + Express + SQLite, mirroring the Python FastAPI stack.

The Code Agent emits one self-contained module per functional requirement, each exporting a PURE
function (stdlib-only, no side effects at import) named exactly as the contract. A MECHANICAL
packager assembles them into a runnable Express app backed by a SQLite key/value store. Tests are
Jest modules; the certification oracle runs the spec-derived cases *in-process* via Node (the JS
analog of the Python executable oracle) — it loads the pure function file and calls it directly, so
no `npm install` is needed for pass@k.

v1 scope: like the Java target, proved features are NOT shipped as Dafny-core adapters (the proof
tier still runs for the verification metric); every feature goes through codegen + cert.
"""

from __future__ import annotations

from pathlib import Path

from src.cleanroom.targets.base import LanguageTarget


class JsTarget(LanguageTarget):
    language = "javascript"
    file_ext = ".js"
    test_framework = "jest"

    def adapter_template(self) -> str:
        raise NotImplementedError("JS Dafny-core adapter is not supported in v1 "
                                  "(adapter shipping is FastAPI-only).")

    def feedback_template(self) -> str:
        raise NotImplementedError("JS test-informed recovery is not supported in v1 "
                                  "(the recovery loop is Python-only).")

    def stage_cores(self, app_dir, project_dir, modules: list[str]) -> dict:
        # No adapter path for JS in v1, so nothing to stage.
        return {"staged": [], "missing": list(modules), "note": "express stack has no adapter"}

    def oracle_name(self, stack: str) -> str:
        return "javascript"

    def package_sample(self, generated_code: dict, code_dir: Path, stack: str) -> "Path | None":
        from src.cleanroom.utils.js_packager import write_js_sources

        return write_js_sources(generated_code, code_dir)

    def run_case(self, code_dir: Path, plan: dict, case: dict, stack: str, timeout: float):
        from src.cleanroom.agents.evaluation.runner import run_case_js

        return run_case_js(code_dir, plan, case, timeout=max(timeout, 30.0))

    def oracle_available(self) -> bool:
        from src.cleanroom.utils.js_tooling import node_available

        return node_available()
