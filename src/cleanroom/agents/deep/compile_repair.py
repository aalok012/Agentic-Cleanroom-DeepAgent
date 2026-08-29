"""Deepagents driver for the Java compile-repair loop (Stage 5b).

Same contract as ``agents/code/compile_repair.run_java_compile_repair`` — same keyword
arguments, same metrics dict — so ``run_pipeline`` can swap drivers without any change to
its reporting. The difference is who drives the iteration:

  deterministic : build -> parse diagnostics -> ONE repair call per broken file -> rebuild,
                  repeated for a fixed number of rounds.
  deepagent     : build -> parse diagnostics -> hand the broken sources to an agent that
                  reads, edits, and calls ``compile_check`` itself until the build is green
                  or its step budget runs out.

==========================  ISOLATION (unchanged)  ==========================
``compile_repair`` may feed the Code Agent compiler diagnostics ONLY — never test cases,
expected outputs, or runtime verdicts. That is preserved structurally here, not by prompt:

  * The code agent's virtual filesystem is seeded with generated CODE files only. Test
    sources are never written into it, so ``ls``/``grep``/``read_file`` cannot surface one.
  * ``compile_check`` returns diagnostics filtered to the seeded code files. Errors in test
    sources are reported to the code agent as a COUNT with no content.
  * Test-source repair runs as a SEPARATE agent invocation with its own filesystem, seeded
    with test sources only. It is never a subagent of the code agent — a deepagents subagent
    shares the parent's filesystem state, which would merge the two pools.
=============================================================================
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from src.cleanroom.agents.code.compile_repair import (
    _diagnostics_for_file,
    _generated_source_map,
    _generated_test_map,
    _map_errors,
    _parse_compile_errors,
    _rebuild_project,
    _run_compile_check,
)
from src.cleanroom.agents.deep.runtime import (
    CODE_ROOT,
    TEST_ROOT,
    agent_step_count,
    build_agent,
    deep_max_steps,
    final_files,
    index_from_virtual_path,
    invoke_agent,
    read_virtual_files,
    seed_files,
    virtual_path,
)

if TYPE_CHECKING:
    from src.cleanroom.agents.code.agent import CodeAgent
    from src.cleanroom.agents.test.agent import TestAgent

CODE_PROMPT = """\
You are repairing COMPILE ERRORS in generated Java sources.

Rules:
* Work only in {root}/. Those files are the generated sources; there is nothing else.
* Fix ONLY what the compiler complains about. Do not add features, rename public methods,
  change method signatures the rest of the project calls, or "improve" working code.
* Every file must keep implementing the same requirement it implements now.
* You have compiler diagnostics and nothing else — no tests, no expected outputs, no
  runtime results. Do not guess at runtime behaviour; fix types, imports, syntax, and
  missing/incorrect declarations.

Loop: read the failing file, edit it, then call `compile_check` to rebuild and get fresh
diagnostics. Repeat until `compile_check` reports success or you can make no further
progress. Call `compile_check` after each batch of edits — it is the only source of truth.
"""

TEST_PROMPT = """\
You are repairing COMPILE ERRORS in generated JUnit test sources.

Rules:
* Work only in {root}/. Those files are the generated tests.
* Fix compilation only — imports, types, syntax, JUnit annotations. NEVER weaken, delete,
  or trivially satisfy an assertion, and never change what a test asserts. The test suite is
  a frozen oracle; changing its expectations would invalidate the run.
* If a test cannot compile without changing what it asserts, leave it and say so.

Loop: edit, then call `compile_check` for fresh diagnostics. Repeat until it reports success
or you can make no further progress.
"""


def run_java_compile_repair_deep(
    *,
    code_agent: CodeAgent,
    ir: dict,
    code_dir: Path,
    stack: str,
    generated_tests: dict | None = None,
    test_agent: TestAgent | None = None,
    dafny_proj: Path | None = None,
    adapter_modules: dict[str, str] | None = None,
    max_rounds: int = 2,
    timeout: float = 240.0,
) -> dict:
    """Agent-driven Java compile repair. Mutates ``ir`` in place; returns run metrics.

    ``max_rounds`` keeps its meaning as an effort dial: 0 disables the stage, and it scales
    the agent's step budget rather than counting fixed repair rounds.
    """
    adapter_modules = adapter_modules or {}
    generated_tests = generated_tests if generated_tests is not None else (ir.get("generated_tests") or {})
    if generated_tests and "generated_tests" not in ir:
        ir["generated_tests"] = generated_tests

    metrics: dict = {
        "driver": "deepagent",
        "max_rounds": max(0, int(max_rounds)),
        "attempts": [],
        "repaired_files": [],
        "ok": False,
        "skipped": False,
        "reason": "",
        "unmapped_errors": [],
        "agent": {},
    }
    if metrics["max_rounds"] <= 0:
        metrics["skipped"] = True
        metrics["reason"] = "disabled"
        return metrics
    if not ir.get("generated_code"):
        metrics["skipped"] = True
        metrics["reason"] = "no generated_code"
        return metrics

    def rebuild() -> Path:
        return _rebuild_project(
            code_agent=code_agent,
            ir=ir,
            code_dir=Path(code_dir),
            stack=stack,
            generated_tests=generated_tests,
            dafny_proj=dafny_proj,
            adapter_modules=adapter_modules,
        )

    def check(project_dir: Path):
        """Compile once and split the diagnostics across the two isolation pools."""
        result = _run_compile_check(project_dir, stack, timeout=timeout)
        errors = _parse_compile_errors(result.diagnostics, project_dir)
        source_map = _generated_source_map(ir.get("generated_code") or {}, project_dir, stack)
        test_map = _generated_test_map(generated_tests or {}, project_dir, stack)
        mapped_code, mapped_tests, unmapped = _map_errors(errors, source_map, test_map)
        return result, errors, mapped_code, mapped_tests, unmapped

    # --- initial build + compile: identical to the deterministic driver -------------
    project_dir = rebuild()
    chk, errors, mapped_code, mapped_tests, unmapped = check(project_dir)
    metrics["attempts"].append(_attempt(0, chk, errors, mapped_code, mapped_tests, unmapped))
    if chk.ok or chk.skipped:
        metrics["ok"] = chk.ok
        metrics["skipped"] = chk.skipped
        metrics["reason"] = chk.reason
        return metrics
    if not mapped_code and not mapped_tests:
        metrics["reason"] = "compile failed, but no generated Java source could be mapped"
        metrics["unmapped_errors"] = [e.as_dict() for e in unmapped[:20]]
        return metrics

    steps = deep_max_steps() * max(1, metrics["max_rounds"])

    # --- pool 1: generated CODE (compiler diagnostics only) ------------------------
    if mapped_code:
        agent_metrics = _repair_pool(
            ir=ir,
            pool="code",
            root=CODE_ROOT,
            prompt=CODE_PROMPT.format(root=CODE_ROOT),
            items=_code_items(ir, mapped_code),
            rebuild=rebuild,
            check=check,
            commit=lambda idx, content: _commit_code(ir, idx, content),
            diagnostics_for=lambda project_dir, chk, mc, mt: _code_diagnostics(
                project_dir, chk, mc, mt),
            max_steps=steps,
        )
        metrics["agent"]["code"] = agent_metrics
        metrics["repaired_files"].extend(agent_metrics.pop("repaired", []))

    # --- pool 2: generated TESTS (separate agent, separate filesystem) -------------
    project_dir = rebuild()
    chk, errors, mapped_code, mapped_tests, unmapped = check(project_dir)
    if mapped_tests and test_agent is None:
        metrics["reason"] = "compile failed in generated tests, but no test repair agent was provided"
        metrics["unmapped_errors"] = [e.as_dict() for errs in mapped_tests.values() for e in errs][:20]
    elif mapped_tests:
        agent_metrics = _repair_pool(
            ir=ir,
            pool="test",
            root=TEST_ROOT,
            prompt=TEST_PROMPT.format(root=TEST_ROOT),
            items=_test_items(generated_tests, mapped_tests),
            rebuild=rebuild,
            check=check,
            commit=lambda idx, content: _commit_test(ir, generated_tests, idx, content),
            diagnostics_for=lambda project_dir, chk, mc, mt: _test_diagnostics(
                project_dir, chk, mt),
            max_steps=steps,
        )
        metrics["agent"]["test"] = agent_metrics
        metrics["repaired_files"].extend(agent_metrics.pop("repaired", []))

    # --- final verdict --------------------------------------------------------------
    project_dir = rebuild()
    chk, errors, mapped_code, mapped_tests, unmapped = check(project_dir)
    metrics["attempts"].append(
        _attempt(len(metrics["attempts"]), chk, errors, mapped_code, mapped_tests, unmapped))
    metrics["ok"] = chk.ok
    metrics["skipped"] = chk.skipped
    metrics["reason"] = chk.reason if (chk.ok or chk.skipped) else (
        metrics["reason"] or chk.reason or "compile failed after agent repair")
    if not chk.ok:
        metrics["unmapped_errors"] = [e.as_dict() for e in unmapped[:20]]
    return metrics


# --- pool runner -------------------------------------------------------------------
def _repair_pool(*, ir, pool, root, prompt, items, rebuild, check, commit,
                 diagnostics_for, max_steps) -> dict:
    """Run ONE agent over ONE isolation pool.

    ``items`` is ``{artifact_index: (display_name, content)}``. The agent's virtual
    filesystem is seeded with exactly those files and nothing else.
    """
    paths = {virtual_path(root, idx, name): idx for idx, (name, _) in items.items()}
    seeded = seed_files({p: items[idx][1] for p, idx in paths.items()})
    committed: dict[int, str] = {}
    checks: list[dict] = []

    @tool
    def compile_check() -> str:
        """Rebuild the project with your current edits and return fresh compiler diagnostics.

        Call this after editing. It is the only way to see whether a fix worked.
        """
        current = read_virtual_files(list(paths))
        for path, content in current.items():
            committed[paths[path]] = content
            commit(paths[path], content)
        project_dir = rebuild()
        chk, errors, mapped_code, mapped_tests, unmapped = check(project_dir)
        checks.append({"ok": chk.ok, "skipped": chk.skipped, "error_count": len(errors)})
        if chk.skipped:
            return f"Compile check unavailable: {chk.reason}. Stop and leave the sources as they are."
        if chk.ok:
            return "SUCCESS: the project compiles. Stop now — do not make further edits."
        return diagnostics_for(project_dir, chk, mapped_code, mapped_tests)

    agent = build_agent([compile_check], prompt, temperature=0.0, name=f"compile-repair-{pool}")
    listing = "\n".join(f"- {p}" for p in sorted(paths))
    opening = (
        f"{len(paths)} generated file(s) fail to compile:\n{listing}\n\n"
        "Call `compile_check` first to see the current diagnostics, then repair them."
    )

    state = invoke_agent(agent, opening, seeded, max_steps=max_steps)

    # The agent's last edits may post-date its final compile_check — commit them too.
    for path, content in final_files(state).items():
        idx = paths.get(path)
        if idx is None:
            idx = index_from_virtual_path(path)   # agent renamed/created a file
        if idx is None or idx not in items:
            continue
        if committed.get(idx) != content:
            committed[idx] = content
            commit(idx, content)

    repaired = [
        {"round": 1, "kind": pool, "path": name, "index": idx}
        for idx, (name, original) in items.items()
        if idx in committed and committed[idx] != original
    ]
    return {
        "pool": pool,
        "files_seeded": len(paths),
        "files_changed": len(repaired),
        "compile_checks": len(checks),
        "agent_steps": agent_step_count(state),
        "max_steps": max_steps,
        "repaired": repaired,
    }


# --- pool wiring: code --------------------------------------------------------------
def _code_items(ir: dict, mapped_code: dict) -> dict[int, tuple[str, str]]:
    files = (ir.get("generated_code") or {}).get("files") or []
    out: dict[int, tuple[str, str]] = {}
    for idx in sorted(mapped_code):
        if idx >= len(files):
            continue
        f = files[idx]
        name = f"{str(f.get('fr_id') or f.get('feature_id') or idx).replace('.', '_')}.java"
        out[idx] = (name, f.get("content") or "")
    return out


def _commit_code(ir: dict, index: int, content: str) -> None:
    files = (ir.get("generated_code") or {}).get("files") or []
    if 0 <= index < len(files):
        files[index] = {**files[index], "content": content}


def _strip_test_diagnostics(diagnostics: str, test_errors: list) -> str:
    """Remove every diagnostic block belonging to a generated TEST source.

    Necessary because ``_diagnostics_for_file`` appends a raw tail of the compiler output,
    and javac echoes the offending SOURCE LINE beneath each error. Without this filter a
    failing ``assertEquals(42, ...)`` would put its expected value in front of the Code
    Agent — precisely the leak this stage is supposed to make impossible.

    A javac/maven diagnostic is a block: an anchored ``path:line: error:`` line followed by
    the echoed source and caret. We drop a block once its anchor names a test path, and
    resume at the next anchor that does not.
    """
    test_names = {str(e.path) for e in test_errors}
    test_names.update(e.path.name for e in test_errors)
    if not test_names or not diagnostics:
        return diagnostics or ""
    anchor = re.compile(r"^(?:\[ERROR\]\s+)?(?P<path>\S+?\.java)[:\[]")
    kept: list[str] = []
    skipping = False
    for line in diagnostics.splitlines():
        m = anchor.match(line.strip())
        if m:
            path = m.group("path")
            skipping = any(path.endswith(name) or name.endswith(path) for name in test_names)
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _code_diagnostics(project_dir, chk, mapped_code, mapped_tests) -> str:
    """Diagnostics for the CODE agent: generated-code errors in full, test errors as a
    bare count. Leaking a test source into this message would break the isolation the
    stage exists to preserve."""
    test_errors = [e for errs in mapped_tests.values() for e in errs]
    scrubbed = _strip_test_diagnostics(chk.diagnostics, test_errors)
    parts = [_diagnostics_for_file(scrubbed, errs, project_dir)
             for _, errs in sorted(mapped_code.items())]
    if mapped_tests:
        parts.append(
            f"\n[{len(test_errors)} further error(s) are in generated TEST sources. They are "
            "repaired separately and are deliberately not shown to you. Ignore them.]"
        )
    if not parts:
        parts.append(scrubbed[-4000:] or chk.reason)
    return "\n\n".join(parts)[-12000:]


# --- pool wiring: tests -------------------------------------------------------------
def _test_items(generated_tests: dict, mapped_tests: dict) -> dict[int, tuple[str, str]]:
    features = (generated_tests or {}).get("features") or []
    out: dict[int, tuple[str, str]] = {}
    for idx in sorted(mapped_tests):
        if idx >= len(features):
            continue
        feature = features[idx]
        name = f"Feature_{str(feature.get('feature_id') or idx).replace('.', '_')}Test.java"
        out[idx] = (name, feature.get("test_source") or "")
    return out


def _commit_test(ir: dict, generated_tests: dict, index: int, content: str) -> None:
    features = (generated_tests or {}).get("features") or []
    if 0 <= index < len(features):
        features[index] = {**features[index], "test_source": content}
        ir["generated_tests"] = generated_tests


def _test_diagnostics(project_dir, chk, mapped_tests) -> str:
    parts = [_diagnostics_for_file(chk.diagnostics, errs, project_dir)
             for _, errs in sorted(mapped_tests.items())]
    if not parts:
        parts.append(chk.diagnostics[-4000:] or chk.reason)
    return "\n\n".join(parts)[-12000:]


# --- metrics ------------------------------------------------------------------------
def _attempt(round_idx, chk, errors, mapped_code, mapped_tests, unmapped) -> dict:
    return {
        "round": round_idx,
        "ok": chk.ok,
        "skipped": chk.skipped,
        "command": chk.command,
        "reason": chk.reason,
        "error_count": len(errors),
        "code_files": len(mapped_code),
        "test_files": len(mapped_tests),
        "unmapped_errors": [e.as_dict() for e in unmapped[:8]],
    }
