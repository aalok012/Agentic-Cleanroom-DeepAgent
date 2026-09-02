"""Substitute the full-toolset agents into the existing pipeline.

Rather than fork ``run_pipeline`` (1200 lines of stage orchestration, checkpointing, packaging
and metrics), this swaps the SIX generation seams and lets the rest of the pipeline run
untouched. Two things follow from that, both wanted:

  * every metric the pipeline already computes — verification_pass_ratio, PassVer@1,
    test_case_pass_ratio, tokens, latency, cost — is produced the same way, so a full-toolset
    run is directly comparable to a clean-room run rather than merely similar;
  * the arm is a decorator over the pipeline, so it cannot drift out of sync with it.

The six seams:

    SpecAgent._contract_feature                     (single-shot -> agent, NEW)
    DependencyAnalyzer._infer_semantic_edges        (single-shot -> agent, NEW)
    deep.planning.deep_design_feature               (agent -> agent, disk backend + shell)
    deep.generation.deep_generate_code                       "
    deep.generation.deep_generate_tests                      "
    deep.generation.deep_generate_dafny                      "

Patching, not editing, is deliberate: the clean-room drivers keep their StateBackend and stay
the default, and ``tests/test_isolation.py`` keeps guarding them. Nothing here runs unless a
caller opts in.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.cleanroom.experiments import agents as full
from src.cleanroom.experiments.full_toolset import RunLog


def install_full_toolset(root: Path, log: RunLog | None = None, *,
                         model: str | None = None) -> RunLog:
    """Point every generation seam at the full-toolset agents. Returns the shared RunLog.

    ``root`` is the shared on-disk backend directory — ONE for the whole run, which is what
    removes isolation: every agent reads and writes the same tree.
    """
    log = log or RunLog()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: spec contracts -----------------------------------------------------
    from src.cleanroom.agents.spec_agent import agent as spec_mod

    def _contract_feature(self, feature_name, feature_id, frs):
        return full.spec_contracts(root, feature_name, feature_id, frs, log, model=model)

    spec_mod.SpecAgent._contract_feature = _contract_feature

    # --- Stage 2: semantic FR edges --------------------------------------------------
    from src.cleanroom.agents.dependency import agent as dep_mod
    from src.cleanroom.utils.ir import feature_id_of

    def _infer_semantic_edges(self, feature, within):
        frs = feature.get("functional_requirements", []) or []
        edges = full.dependency_edges(
            root, feature.get("name", ""), feature_id_of(feature), frs, log, model=model)
        # The caller trusts its own `within` set; keep that contract exactly as the
        # deterministic path does so an agent cannot inject a cross-feature edge.
        return [(s, t) for s, t in edges if s in within and t in within and s != t]

    dep_mod.DependencyAnalyzer._infer_semantic_edges = _infer_semantic_edges

    # --- Stage 3: planning -----------------------------------------------------------
    from src.cleanroom.agents.deep import planning as planning_mod

    def deep_design_feature(feature_name, fr_order, text_by_id, contracts_by_fr, *,
                            stack="python", return_metrics=False, **_kw):
        plans = full.plan_feature(root, feature_name, list(fr_order), text_by_id,
                                  contracts_by_fr, log, stack=stack, model=model)
        return (plans, {"driver": "full-toolset"}) if return_metrics else plans

    planning_mod.deep_design_feature = deep_design_feature

    # --- Stages 4-6: code / test / proof ---------------------------------------------
    from src.cleanroom.agents.deep import generation as gen_mod

    def deep_generate_code(ir, contracts, *, language="Python", **_kw):
        files = full.generate_code(root, ir, contracts, log, language=language, model=model)
        return files, {"driver": "full-toolset", "frs_requested": len(contracts),
                       "frs_generated": len(files)}

    def deep_generate_tests(ir, feature_id, contracts, *, language="Python", **_kw):
        result = full.generate_tests(root, ir, feature_id, contracts, log,
                                     language=language, model=model)
        return result, {"driver": "full-toolset",
                        "cases": len(result.cases) if result else 0,
                        "has_source": bool(result and result.test_source.strip())}

    def deep_generate_dafny(ir, feature_id, contracts, *, verifier=None, **_kw):
        out = full.generate_dafny(root, ir, feature_id, contracts, log,
                                  verifier=verifier, model=model)
        row = log.rows[-1] if log.rows else {}
        return out, {"driver": "full-toolset",
                     "has_source": bool((out.get("source") or "").strip()),
                     "verify_calls": row.get("verify_calls", 0),
                     "verified": row.get("verified", False)}

    gen_mod.deep_generate_code = deep_generate_code
    gen_mod.deep_generate_tests = deep_generate_tests
    gen_mod.deep_generate_dafny = deep_generate_dafny

    # The agents import these by value at call time (`from ... import deep_generate_code`
    # inside the method body), so patching the module attribute is enough — but the Dafny
    # agent's proof CACHE would otherwise hand back clean-room proofs for a full-toolset run
    # and silently mix the arms. Give this arm its own cache namespace.
    _isolate_proof_cache(root)
    return log


def _isolate_proof_cache(root: Path) -> None:
    """Keep this arm's proof cache separate from the clean-room arm's.

    ``DafnyAgent`` caches a proved feature by a signature over model + prompt + contracts.
    That signature does not include the arm, so without this a clean-room proof would be
    reused for a full-toolset run (or vice versa) and the comparison would be silently
    contaminated by results the other arm produced.
    """
    from src.cleanroom.agents.dafny import agent as dafny_mod

    original = dafny_mod.DafnyAgent._feature_sig

    def _feature_sig(self, ir, feature_id, mod):
        return "fulltoolset-" + original(self, ir, feature_id, mod)

    dafny_mod.DafnyAgent._feature_sig = _feature_sig


def write_tool_report(log: RunLog, output_dir: Path, project: str = "project") -> Path:
    """Write the per-agent tool-call breakdown next to the run's other artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{project}_tool_calls.json"
    dest.write_text(json.dumps(log.as_dict(), indent=2))
    return dest


def format_tool_table(log: RunLog) -> str:
    """The per-agent breakdown as a text table for the run's stdout summary."""
    per_agent = log.by_agent()
    if not per_agent:
        return "(no agent invocations recorded)"
    lines = [f"{'agent':<12} {'runs':>5} {'turns':>6} {'tools':>6} {'secs':>8}  tools used",
             "-" * 78]
    for name, agg in sorted(per_agent.items()):
        used = ", ".join(f"{t}×{n}" for t, n in agg["tools"].items()) or "(none)"
        lines.append(f"{name:<12} {agg['invocations']:>5} {agg['assistant_turns']:>6} "
                     f"{agg['tool_calls_total']:>6} {agg['seconds']:>8.1f}  {used}")
    return "\n".join(lines)
