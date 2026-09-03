"""End-to-end Agentic Cleanroom pipeline.

Runs the full pipeline on a single SRS document and saves the enriched IR:

    SRS.xml
      -> [1] Spec        FR parse + behavioral contracts    [parse: deterministic, contracts: LLM]
      -> [2] Dependency  nested graph (feature + FR level)  [regex deterministic, inner: LLM semantic]
      -> [3] Planning    impl metadata per FR contract      [LLM: signature/layer; docstring deterministic]
      -> [4a] Proof tier   prove feature logic in Dafny     [Dafny verifier + configured LLM, opt-in via --prove]
      -> [4] Code        adapters over proved cores + full code for the rest  [LLM: one call per contract]
      -> [5] Test        black-box test cases               [LLM: one call per feature]
      -> [6] Certification  pass@k for UN-proved features   [executable oracle, opt-in via --certify]

Per-feature guarantee (FastAPI + --prove): a feature PROVED in [4a] ships its LOGIC as the compiled
Dafny core, and the Code Agent writes only a thin adapter over it (Stage 4); everything else gets
full code certified by pass@k in [6] (prove-or-test fallback). The proof tier runs BEFORE codegen
so the Code Agent knows which features ship from Dafny.

Talks to any OpenAI-compatible /v1 endpoint. Set LLM_BASE_URL (self-hosted vLLM/Ollama/TGI)
or OPENAI_API_KEY (api.openai.com) in .env.
The proof tier additionally needs a `dafny` binary on PATH/$DAFNY.

Every optional agent has its own ON/OFF switch, so any "arm" is just a flag combination — there
is no separate baseline script. Required agents (Spec, Planning, Code) are the irreducible
spec→code task and have no switch; Dependency / Proof / Test / Certify / Recovery toggle freely.

Usage:
    # full pipeline
    uv run python run_pipeline.py "data/srs/dineout_srs.xml" --prove --certify --samples 3
    # certify only, no recovery
    uv run python run_pipeline.py "data/srs/dineout_srs.xml" --certify --no-recovery
    # minimal arm: spec → code only (no deps/proof/test/cert)
    uv run python run_pipeline.py "data/srs/dineout_srs.xml" --no-dependency --no-test
    # convenience preset (== --no-prove --no-recovery --no-llm-deps --temperature 0)
    uv run python run_pipeline.py "data/srs/dineout_srs.xml" --baseline --certify
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing project modules — some read API keys / model env vars at import time.
load_dotenv()

# --- Agents (one per pipeline stage) ---
from src.cleanroom.agents.evaluation.agent import CertificationAgent
from src.cleanroom.agents.code.agent import CodeAgent
from src.cleanroom.agents.dafny.agent import DafnyAgent
from src.cleanroom.agents.dependency.agent import DependencyAnalyzer
from src.cleanroom.agents.planning.agent import PlanningAgent
from src.cleanroom.agents.spec_agent.agent import SpecAgent
from src.cleanroom.agents.test.agent import TestAgent

# --- Utilities, metrics, config ---
from src.cleanroom.llms.callbacks.metric import GLOBAL_METRICS
from src.cleanroom.utils.code_stats import code_stats
from src.cleanroom.utils.cost import estimate_cost_by_model
from src.cleanroom.config import LANGUAGES, RunConfig
from src.cleanroom.utils.llm_client import DEFAULT_MODEL, get_llm, llm_api_key_configured
from src.cleanroom.utils.dafny_project import compile_dafny, scaffold_dafny_project, summarize_dafny_java_api
from src.cleanroom.utils.packager import build_runnable_package
from src.cleanroom.utils.stack_select import select_stack
from src.cleanroom.utils.ir import normalize_ir
from src.cleanroom.utils.run_record import make_run_id, record_run
from src.cleanroom.utils.usage_log import append_usage_log, metrics_from_globals

def _banner(step: str, title: str) -> None:
    print(f"\n{'=' * 70}\n  {step}  {title}\n{'=' * 70}")


def _inject_empty_dependency(ir: dict) -> dict:
    """Dependency agent OFF (--no-dependency): install a valid but EMPTY graph so Planning
    (which requires `dependency_graph`) still runs, with no edges/prerequisites — every feature
    is independent and ordered as it appears in the spec."""
    fids = [str(f.get("id")) for f in ir.get("features", [])]
    ir["dependency_graph"] = {"nodes": fids, "edges": [], "build_order": fids, "cycles": []}
    for f in ir.get("features", []):
        f["fr_order"] = [str(r.get("id")) for r in f.get("functional_requirements", [])]
        f["fr_edges"] = []
        f["fr_cycles"] = []
    return ir


def _timed(name: str, stages: list, fn):
    """Run fn(), recording (and printing) its wall-clock time and LLM tokens consumed."""
    in0, out0, calls0 = GLOBAL_METRICS.snapshot()
    t0 = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - t0
    in1, out1, calls1 = GLOBAL_METRICS.snapshot()
    calls = calls1 - calls0
    stages.append(
        {
            "name": name,
            "seconds": round(seconds, 3),
            "input_tokens": in1 - in0,
            "output_tokens": out1 - out0,
            "calls": calls,
        }
    )
    extra = f"  ·  {(in1 - in0) + (out1 - out0):,} tokens, {calls} call(s)" if calls else ""
    print(f"  ⏱  done in {seconds:.2f}s{extra}")
    return result


def _finalize_metrics(metrics: dict, stages: list) -> None:
    in_tot, out_tot, calls = GLOBAL_METRICS.snapshot()
    cost, by_model = estimate_cost_by_model(GLOBAL_METRICS.calls)
    # Accurate per-model cost; show all models used if a run is intentionally mixed.
    metrics["model"] = "+".join(sorted(by_model)) if len(by_model) > 1 else (GLOBAL_METRICS.model or DEFAULT_MODEL)
    metrics["total_seconds"] = round(sum(s["seconds"] for s in stages), 3)
    metrics["tokens"] = {"input": in_tot, "output": out_tot, "total": in_tot + out_tot, "calls": calls}
    metrics["cost_usd"] = cost
    metrics["cost_by_model"] = by_model


def _compute_summary(metrics: dict, ir: dict) -> dict:
    """Roll the per-stage metrics up into the headline research metrics — ALL at FR granularity
    (the FR is the atomic, independently-certifiable unit; a proved feature certifies all its FRs)."""
    d = metrics.get("dafny") or {}
    cert = metrics.get("certification") or {}
    rec = metrics.get("recovery") or {}
    tok = metrics.get("tokens") or {}
    calls = tok.get("calls", 0) or 0

    # feature id -> its FR ids, and the full FR universe
    fr_by_feature: dict[str, list[str]] = {}
    all_fr_ids: set[str] = set()
    for f in ir.get("features", []):
        ids = [str(r.get("id")) for r in f.get("functional_requirements", [])]
        fr_by_feature[str(f.get("id"))] = ids
        all_fr_ids.update(ids)
    total_frs = len(all_fr_ids)

    # FR-level sets: a proved feature certifies ALL its FRs; pass@1 is already per-FR.
    proved_fr_ids = {fr for fid in (metrics.get("proved_feature_ids") or [])
                     for fr in fr_by_feature.get(fid, [])}
    pass1_fr_ids = set(cert.get("pass_at_1_fr_ids") or [])
    certified_fr_ids = proved_fr_ids | pass1_fr_ids

    # avg verification iterations PER FR: each FR inherits its feature's proof rounds.
    wr = wn = 0
    for fe in d.get("features", []):
        n = len(fr_by_feature.get(str(fe.get("feature_id")), []))
        wr += fe.get("rounds", 0) * n
        wn += n
    n_tested = cert.get("n_total_tested_features", 0)
    loops_run = rec.get("loops_run", 0)

    return {
        # FRs covered by a discharged proof / all FRs
        "verification_pass_ratio": round(len(proved_fr_ids) / total_frs, 4) if total_frs else 0.0,
        # fraction of individual test-case executions that passed
        "test_case_pass_ratio": round(cert.get("aggregate_case_pass_rate", 0.0), 4),
        # FRs certified by PROOF or by passing pass@1 (set union, dedupes overlap) / all FRs
        "PassVer@1": round(len(certified_fr_ids) / total_frs, 4) if total_frs else 0.0,
        # avg generate->verify->revise rounds, FR-weighted
        "avg_verification_iteration": round(wr / wn, 2) if wn else 0.0,
        # avg test-track iterations = initial pass + recovery passes (Python recovery loop)
        "avg_test_iteration": round(1 + loops_run, 2) if (cert and n_tested) else 0.0,
        "avg_input_token": round(tok.get("input", 0) / calls, 1) if calls else 0.0,
        "avg_output_token": round(tok.get("output", 0) / calls, 1) if calls else 0.0,
        "total_time": round(metrics.get("total_seconds", 0.0), 2),
    }


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 70)
    print("  RUN METRICS")
    print("=" * 70)
    rows = [
        ("verification_pass_ratio", f"{s['verification_pass_ratio']:.4f}"),
        ("test_case_pass_ratio", f"{s['test_case_pass_ratio']:.4f}"),
        ("PassVer@1", f"{s['PassVer@1']:.4f}"),
        ("avg_verification_iteration", f"{s['avg_verification_iteration']:.2f}"),
        ("avg_test_iteration", f"{s['avg_test_iteration']:.2f}"),
        ("avg_input_token", f"{s['avg_input_token']:.1f}"),
        ("avg_output_token", f"{s['avg_output_token']:.1f}"),
        ("total_time", f"{s['total_time']:.2f}s"),
    ]
    for name, val in rows:
        print(f"  {name:<28}: {val:>12}")
    print("=" * 70)


# --- Stage-level checkpointing (opt-in via --resume) -------------------------
# The pipeline is a long, expensive sequence (proof tier + codegen + cert can each be
# many LLM calls). A checkpoint is written AFTER every completed stage so a failure in a
# later stage can resume without re-spending the earlier stages' tokens. We persist the
# enriched `ir`, the `metrics` (incl. the per-stage timing/token table), and a full snapshot
# of GLOBAL_METRICS (input/output/calls) so the resumed run's totals + cost stay accurate.
CKPT_STAGES = ("spec", "dependency", "planning", "proof", "code", "test", "certification")


def _ckpt_path(output_dir: Path, srs_path: Path) -> Path:
    return output_dir / f"{srs_path.stem}_ckpt.json"


def _save_ckpt(path: Path, completed: list, ir: dict, metrics: dict) -> None:
    payload = {
        "completed": list(completed),
        "ir": ir,
        "metrics": metrics,
        "global_metrics": {
            "input_tokens": GLOBAL_METRICS.input_tokens,
            "output_tokens": GLOBAL_METRICS.output_tokens,
            "latency_ms": GLOBAL_METRICS.latency_ms,
            "model": GLOBAL_METRICS.model,
            "calls": GLOBAL_METRICS.calls,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)   # atomic: a crash mid-write never corrupts a good checkpoint


def _load_ckpt(path: Path) -> dict | None:
    if not path.exists():
        return None
    ckpt = json.loads(path.read_text())
    gm = ckpt.get("global_metrics") or {}
    # Restore the process-wide token accumulator so resumed totals include prior stages.
    GLOBAL_METRICS.input_tokens = gm.get("input_tokens", 0)
    GLOBAL_METRICS.output_tokens = gm.get("output_tokens", 0)
    GLOBAL_METRICS.latency_ms = gm.get("latency_ms", 0.0)
    GLOBAL_METRICS.model = gm.get("model", "") or ""
    GLOBAL_METRICS.calls = list(gm.get("calls", []))
    return ckpt


def run(srs_path: Path, output_dir: Path, cfg: RunConfig) -> tuple[dict, dict]:
    stages: list = []
    language = cfg.language
    stack = cfg.stack
    certify, samples, prove = cfg.certify, cfg.samples, cfg.prove
    prove_target, max_cert_loops = cfg.prove_target, cfg.max_cert_loops
    metrics: dict = {"srs": srs_path.name, "stack": stack, "language": language,
                     "stages": stages, "config": cfg.as_dict()}
    # Per-stage models: each agent gets its OWN client so a mixed-model run prices correctly
    # (estimate_cost_by_model reads the per-call model recorded on GLOBAL_METRICS). Generation
    # stages share cfg.temperature; certification samples at cfg.cert_temperature for pass@k diversity.
    spec_llm = get_llm(model=cfg.spec_model, temperature=cfg.temperature)
    dep_llm = get_llm(model=cfg.dependency_model, temperature=cfg.temperature)
    plan_llm = get_llm(model=cfg.planning_model, temperature=cfg.temperature)
    code_llm = get_llm(model=cfg.code_model, temperature=cfg.temperature)
    test_llm = get_llm(model=cfg.test_model, temperature=cfg.temperature)
    cert_llm = get_llm(model=cfg.cert_model, temperature=cfg.cert_temperature)
    on = cfg.agents_enabled()
    arm = " ".join(f"{name}={'on' if state else 'off'}" for name, state in on.items())
    print(f"  Agents: {arm}")
    if cfg.baseline:
        print("  [BASELINE preset] proof OFF · recovery OFF · regex-only deps · temperature 0")

    # --- Checkpoint bookkeeping (opt-in via --resume) ------------------------
    # On --resume, adopt the persisted metrics/per-stage table + IR and skip completed stages.
    ckpt_path = _ckpt_path(output_dir, srs_path)
    ckpt = _load_ckpt(ckpt_path) if cfg.resume else None
    done: set = set(ckpt.get("completed", [])) if ckpt else set()
    completed: list = list(ckpt.get("completed", [])) if ckpt else []
    ir_ckpt: dict = (ckpt or {}).get("ir") or {}
    if ckpt:
        metrics = ckpt.get("metrics") or metrics      # persisted timing/token table + sub-metrics
        stages = metrics.setdefault("stages", stages)  # _timed appends new stages after the loaded ones
        stack = ir_ckpt.get("stack", stack)
        print(f"\n  [RESUME] checkpoint loaded — skipping completed stages: {sorted(done) or '(none)'}")

    def _checkpoint(name: str, _ir: dict) -> None:
        if name not in completed:
            completed.append(name)
        _save_ckpt(ckpt_path, completed, _ir, metrics)

    # --- Stage 1: Spec (deterministic FR parse, then LLM behavioral contracts) -
    # The SRSReader extracts functional requirements only (section-aware, no LLM); the
    # contract phase then writes a design-by-contract spec per FR (one LLM call/feature).
    _banner("[1]", "Spec Agent — parsing the SRS (FR-only) and writing behavioral contracts")
    if "spec" in done:
        ir = ir_ckpt
        stack = ir.get("stack", stack)
        n_fr = sum(len(f["functional_requirements"]) for f in ir["features"])
        print(f"  [skip — resumed] {ir.get('project_name')}: {len(ir['features'])} feature(s), "
              f"FR={n_fr}, stack={stack}")
    else:
        spec = SpecAgent(llm=spec_llm)
        # output_dir=None: skip the redundant <stem>_ir.json dump (subset of full_ir.json).
        ir_obj = _timed("spec_parse", stages, lambda: spec.run(srs_path, output_dir=None))
        ir = ir_obj.model_dump()
        ir = _timed("spec_contracts", stages,
                    lambda: spec.synthesize_contracts(ir, output_dir=output_dir))
        n_fr = sum(len(f["functional_requirements"]) for f in ir["features"])
        print(f"Project   : {ir['project_name']}")
        print(f"Features  : {len(ir['features'])}  (FR={n_fr})")
        print(f"Contracts : {len(ir['contracts'])}")

        # --- Resolve the target stack (auto-select from the spec unless pinned) -----
        # Done here, after contracts exist, so the classifier can read preconditions. The
        # chosen stack drives planning signatures, codegen shape, packaging and certification.
        if language == "java":
            # Java sub-stack resolved in RunConfig (plain 'java' or 'spring'); never auto-select.
            stack = stack if stack in ("java", "spring") else "java"
            _java_desc = ("spring (Spring Boot web), tests: JUnit+MockMvc" if stack == "spring"
                          else "java (plain), tests: JUnit")
            print(f"Language  : java   (stack: {_java_desc})")
        elif language == "javascript":
            stack = "express"
            print("Language  : javascript   (stack: express — Node/Express + SQLite, tests: Jest)")
        elif stack == "auto":
            stack, reason = select_stack(ir)
            print(f"Stack     : {stack}  (auto-selected — {reason})")
        else:
            print(f"Stack     : {stack}  (pinned via --stack)")
        metrics["stack"] = stack
        metrics["language"] = language
        ir["stack"] = stack
        ir["language"] = language
        _checkpoint("spec", ir)

    # --- Stage 2: Dependency (nested; LLM infers semantic FR->FR edges) -------
    _banner("[2]", "Dependency Agent — building the nested dependency graph")
    if "dependency" in done:
        print("  [skip — resumed] dependency graph loaded from checkpoint.")
    else:
        if not cfg.run_dependency:
            # Agent OFF (--no-dependency): empty graph, no edges/prerequisites. No LLM call.
            print("  Dependency agent OFF (--no-dependency) — empty graph, features independent.")
            ir = _inject_empty_dependency(ir)
        else:
            # cfg.llm_deps=False (e.g. --baseline) → deterministic regex-only edges.
            ir = _timed("dependency", stages,
                        lambda: DependencyAnalyzer(llm=dep_llm if cfg.llm_deps else None).enrich(ir, output_dir=output_dir))
        graph = ir["dependency_graph"]
        print(f"Outer nodes : {len(graph['nodes'])}  edges: {len(graph['edges'])}")
        print(f"Build order : {' -> '.join(graph['build_order']) or '(none)'}")
        print(f"Cycles      : {graph['cycles'] or 'none'}")
        inner_edges = sum(len(f.get("fr_edges", [])) for f in ir["features"])
        inner_cycles = sum(len(f.get("fr_cycles", [])) for f in ir["features"])
        print(f"Inner (FR)  : {inner_edges} edge(s), {inner_cycles} cycle(s) across features")
        _checkpoint("dependency", ir)

    # --- Stage 3: Planning (per-FR contracts) --------------------------------
    _banner("[3]", "Planning Agent — generating per-FR contracts (signature + docstring + path)")
    if "planning" in done:
        print(f"  [skip — resumed] planning loaded: {len(ir['planning']['contracts'])} contract(s).")
    else:
        ir = _timed("planning", stages, lambda: PlanningAgent(llm=plan_llm, stack=stack).enrich(ir, output_dir=output_dir))
        contracts = ir["planning"]["contracts"]
        print(f"Contracts   : {len(contracts)}  (in dependency order)")
        for c in contracts[:12]:
            print(f"  [{c['fr_id']}] ({c['mvc_layer']}) {c['signature']}")
            print(f"        -> {c['file_path']}   prereqs={c['prerequisite_fr_ids']}")
        if len(contracts) > 12:
            print(f"  … and {len(contracts) - 12} more (see {ir['project_name']}_planning.json)")
        _checkpoint("planning", ir)

    # --- Stage 4a: Proof tier (Dafny) — prove each feature's logic where possible -----
    # Runs BEFORE code generation so the Code Agent knows which features ship from Dafny. A PROVED
    # feature's logic ships as the compiled Dafny core (the Code Agent writes only a thin adapter
    # over it); everything else gets full code + pass@k. Degrades gracefully: no --prove or no dafny
    # binary => 0 proved => normal codegen for all features.
    proved_feature_ids: set[str] = set()
    proved_modules: dict[str, str] = {}        # feature_id -> Dafny module base (e.g. "F4_1")
    proved_sources: dict[str, str] = {}        # feature_id -> Dafny source
    proved_compiled_ids: set[str] = set()      # proved features whose Dafny ALSO translated to native
    proved_java_apis: dict[str, dict] = {}      # feature_id -> actual translated Java API summary
    proj = None
    if "proof" in done:
        # Reconstruct the proof-stage locals from the persisted IR/metrics (NO tokens): the proved
        # set + Dafny sources live in ir['generated_dafny']; the compiled set in metrics['compile'].
        _banner("[4a]", "Dafny Proof Agent — [skip — resumed]")
        gdaf = ir.get("generated_dafny")
        if gdaf:
            feats = gdaf.get("features", [])
            proved_feature_ids = {f["feature_id"] for f in feats if f.get("verified")}
            proved_modules = {f["feature_id"]: f["module"] for f in feats if f.get("verified")}
            proved_sources = {f["feature_id"]: f.get("dafny_source", "") for f in feats if f.get("verified")}
            proved_compiled_ids = {c["feature_id"] for c in (metrics.get("compile") or {}).get("compiled", [])}
            project_dir = output_dir / "generated" / ir["project_name"] / "dafny_proof"
            # Reuse the EXISTING dafny project if present — scaffold_dafny_project() rmtree's it,
            # which would destroy the already-compiled cores (out/<module>-*) that adapter staging
            # and certification need. Only scaffold from scratch when nothing is there.
            proj = project_dir if project_dir.exists() else scaffold_dafny_project(project_dir)
            if prove_target == "java":
                proved_java_apis = {fid: summarize_dafny_java_api(proj, proved_modules[fid])
                                    for fid in sorted(proved_compiled_ids) if fid in proved_modules}
            print(f"  proved {len(proved_feature_ids)}, compiled {len(proved_compiled_ids)}; "
                  f"dafny project re-scaffolded at {project_dir}")
        else:
            print("  nothing to restore (no Dafny output captured in checkpoint).")
    elif prove:
        _banner("[4a]", "Dafny Proof Agent — proving feature logic (formal verification)")
        from src.cleanroom.utils.dafny_verify import dafny_available
        if not dafny_available():
            print("  dafny binary not found — skipping proof tier (every feature falls to code+test).")
        else:
            project_dir = output_dir / "generated" / ir["project_name"] / "dafny_proof"
            proj = _timed("scaffold", stages, lambda: scaffold_dafny_project(project_dir))
            gen = _timed("dafny", stages,
                         lambda: DafnyAgent(proj, model=cfg.proof_model, max_rounds=cfg.prove_rounds).generate(ir))
            proved_feature_ids = {f.feature_id for f in gen.features if f.verified}
            proved_modules = {f.feature_id: f.module for f in gen.features if f.verified}
            proved_sources = {f.feature_id: f.dafny_source for f in gen.features if f.verified}
            ntot, nver = len(gen.features), gen.n_verified
            n_axiom = sum(1 for f in gen.features if f.verified and f.axioms)
            ax_note = f"  ({n_axiom} via assumed axioms)" if n_axiom else ""
            print(f"Proved    : {nver}/{ntot} features{ax_note}  →  {sorted(proved_feature_ids) or '(none)'}")
            for f in gen.features:
                label = "PROVED " if f.verified else "unproved"
                axt = f"  [{len(f.axioms)} assumed axiom(s)]" if f.verified and f.axioms else ""
                print(f"  [{f.feature_id}] {label}  {f.module}  ({f.rounds}r){axt}")
            metrics["dafny"] = {
                "model": cfg.proof_model, "n_features": ntot, "n_verified": nver,
                "n_proved_with_axioms": n_axiom,
                "verification_rate": round(nver / ntot, 3) if ntot else 0.0,
                "features": [{"feature_id": f.feature_id, "module": f.module,
                              "verified": f.verified, "rounds": f.rounds,
                              "axioms": len(f.axioms)} for f in gen.features],
            }
            comp = _timed("compile", stages, lambda: compile_dafny(proj, gen, target=prove_target))
            metrics["compile"] = comp
            proved_compiled_ids = {c["feature_id"] for c in comp.get("compiled", [])}
            if prove_target == "java":
                proved_java_apis = {
                    fid: summarize_dafny_java_api(proj, proved_modules[fid])
                    for fid in sorted(proved_compiled_ids)
                    if fid in proved_modules
                }
            print(f"Compiled  : {len(comp['compiled'])}/{nver} proved features to {comp.get('target', prove_target)}")
            ir = {**ir, "generated_dafny": gen.model_dump(),
                  "proved_feature_ids": sorted(proved_feature_ids)}
        _checkpoint("proof", ir)

    # Adapter shipping (Dafny core + thin glue) ships proved features FROM Dafny and excludes them
    # from pass@k. Supported on FastAPI (compiled-to-Python core + Python adapter) and Spring
    # (translated-to-Java core staged into the Maven build + a @RestController adapter). On the
    # plain python/java stacks --prove is proof-only and code/test proceed normally for every
    # feature. For Spring we only adapt features whose Dafny ALSO translated to Java successfully —
    # a proved-but-untranslated feature would leave its adapter with no core to call (and the
    # build-check compiles the whole project), so those fall back to normal codegen + cert.
    if stack == "fastapi":
        adapter_feature_ids = set(proved_feature_ids)
    elif stack == "spring":
        adapter_feature_ids = set(proved_feature_ids) & proved_compiled_ids
    else:
        adapter_feature_ids = set()
    adapter_mode = bool(adapter_feature_ids)
    cert_skip = set(adapter_feature_ids)
    # JS (express) ships no Dafny-core adapter, but — like FastAPI/Spring — a PROVED feature's
    # correctness is already guaranteed by the proof, so exclude it from certification too. This
    # makes test_case_pass_ratio / PassVer@1 cover the SAME unproved set across all languages.
    # (adapter_mode stays False: the proved feature's code is still generated normally, we just
    # don't re-test it — JS has no adapter to ship it from.)
    if stack == "express":
        cert_skip = set(proved_feature_ids)

    # --- Stage 4: Code (LLM) — adapters over proved cores + full code for the rest ----
    # code_agent / code_dir are defined unconditionally (the Test, compile-repair, cert and
    # recovery stages all reference them) even when the codegen itself is skipped on resume.
    code_agent = CodeAgent(llm=code_llm, stack=stack, language=language)
    code_dir = output_dir / "generated" / ir["project_name"]
    if "code" in done:
        _banner("[4]", "Code Agent — [skip — resumed]")
        print(f"  loaded {len(ir.get('generated_code', {}).get('files', []))} generated file(s) "
              "from checkpoint (packaged app already on disk).")
    else:
        _banner("[4]", "Code Agent — generating MVC code from the spec")
        if adapter_mode:
            from src.cleanroom.agents.code.schema.code import GeneratedCode
            adapters = []
            for fid in sorted(adapter_feature_ids):
                adapters.append(_timed(
                    "adapter", stages,
                    lambda fid=fid: code_agent.generate_adapter(
                        ir,
                        fid,
                        proved_modules[fid],
                        proved_sources[fid],
                        java_api=proved_java_apis.get(fid),
                    )))
            full = _timed("code", stages,
                          lambda: code_agent.generate(ir, skip_feature_ids=adapter_feature_ids))
            generated_code = GeneratedCode(files=adapters + full.files)
            print(f"  {len(adapters)} proved feature(s) shipped as Dafny-core adapters; "
                  f"{len(full.files)} file(s) generated for the rest.")
        else:
            generated_code = _timed("code", stages, lambda: code_agent.generate(ir))
        ir = {**ir, "generated_code": generated_code.model_dump()}
        metrics["code"] = code_stats(ir["generated_code"])
        stat = metrics["code"]
        if language == "java":
            metrics["code"]["language"] = "Java"   # code_stats labels by Python AST; override
        elif language == "javascript":
            metrics["code"]["language"] = "JavaScript"   # code_stats labels by Python AST; override
        if stack == "fastapi":
            app_dir = build_runnable_package(ir["generated_code"], code_dir, ir=ir)
            metrics["code"]["runnable_app"] = str(app_dir)
            if adapter_mode and proj is not None:
                staged = code_agent.target.stage_cores(
                    Path(app_dir), proj, [proved_modules[fid] for fid in sorted(adapter_feature_ids)])
                metrics["code"]["dafny_cores"] = staged
                print(f"  Staged proved Dafny cores {staged['staged']} + shim into {staged['cores_dir']}")
            print(f"{stat['files']} file(s); {stat['lines_of_code']} LOC ({stat['language']}); "
                  f"layers={stat['files_per_layer']}")
            print(f"Assembled runnable FastAPI app: {app_dir}  (cd \"{code_dir}\" && python -m app)")
        else:
            # python (flat functions), plain java (src/*.java), or spring (a Maven project) — the
            # target picks the layout.
            proj_dir = code_agent.target.package_sample(ir["generated_code"], code_dir, stack)
            print(f"Wrote {stat['files']} file(s); {stat['lines_of_code']} LOC "
                  f"({stat['language']}); layers={stat['files_per_layer']}")
            if stack == "spring":
                metrics["code"]["runnable_app"] = str(proj_dir or code_dir)
                if adapter_mode and proj is not None:
                    staged = code_agent.target.stage_cores(
                        proj_dir or code_dir, proj,
                        [proved_modules[fid] for fid in sorted(adapter_feature_ids)])
                    metrics["code"]["dafny_cores"] = staged
                    print(f"  Staged translated Dafny Java cores {staged['staged']} into "
                          f"{staged.get('java_src', proj_dir)}")
                print(f"Assembled runnable Spring Boot project: {proj_dir or code_dir}  "
                      f"(cd \"{proj_dir or code_dir}\" && mvn spring-boot:run)")
        _checkpoint("code", ir)

    # --- Stage 5: Test (LLM) -------------------------------------------------
    # The Test Agent reads only ir['features']; the generated_code in the IR is
    # structurally ignored — isolation is preserved.
    generated_tests_dict: dict | None = None
    test_agent_runner: TestAgent | None = None
    if "test" in done:
        _banner("[5]", "Test Agent — [skip — resumed]")
        generated_tests_dict = ir.get("generated_tests")
        if cfg.run_test:
            test_agent_runner = TestAgent(llm=test_llm, stack=stack, language=language)
        print(f"  loaded {len((generated_tests_dict or {}).get('features', []))} tested "
              "feature(s) from checkpoint (test modules already on disk).")
    elif not cfg.run_test:
        # Agent OFF (--no-test): no test artifacts. If --certify is also set, the cert stage
        # will synthesize ephemeral tests on the fly (a weaker, non-persisted oracle).
        _banner("[5]", "Test Agent — SKIPPED (--no-test)")
        note = "  certification will synthesize ephemeral tests." if certify else "  (no certification will run.)"
        print(f"  Test agent OFF — no test modules written.{note}")
        _checkpoint("test", ir)
    else:
        _banner("[5]", "Test Agent — deriving black-box tests from the spec")
        test_agent_runner = TestAgent(llm=test_llm, stack=stack, language=language)
        generated_tests = _timed("test", stages, lambda: test_agent_runner.generate(ir))
        generated_tests_dict = generated_tests.model_dump()
        ir = {**ir, "generated_tests": generated_tests_dict}
        if language == "java":
            written_tests = code_agent.target.package_tests(generated_tests_dict, code_dir, stack)
            tests_dir = code_dir / ("src/test/java" if stack == "spring" else "src")
        else:
            tests_dir = output_dir / "generated" / ir["project_name"] / "tests"
            written_tests = TestAgent.write_files(generated_tests, tests_dir, language=language)
        total_cases = sum(len(f.cases) for f in generated_tests.features)
        metrics["tests"] = {"features": len(generated_tests.features), "cases": total_cases}
        _fw = ("JUnit+MockMvc" if stack == "spring" else "JUnit") if language == "java" else (
            "Jest" if language == "javascript" else "pytest")
        print(f"{len(generated_tests.features)} feature(s), {total_cases} test case(s).")
        print(f"Wrote {len(written_tests)} {_fw} module(s) to {tests_dir}")
        _checkpoint("test", ir)

    # --- Stage 5b: Java static compile repair --------------------------------
    # This is compile-diagnostic feedback only: no test cases, expected outputs, or runtime
    # verdicts are fed to the Code Agent. It makes the Java source tree buildable before pass@k.
    if language == "java" and cfg.max_compile_repair_loops > 0:
        _banner("[5b]", f"Java compile repair — static build check "
                        f"(≤ {cfg.max_compile_repair_loops} repairs · {cfg.repair_driver})")
        if cfg.uses_deep_agent():
            from src.cleanroom.agents.deep.compile_repair import (
                run_java_compile_repair_deep as run_java_compile_repair,
            )
        else:
            from src.cleanroom.agents.code.compile_repair import run_java_compile_repair

        adapter_modules = {
            fid: proved_modules[fid]
            for fid in sorted(adapter_feature_ids)
            if fid in proved_modules
        }
        compile_repair = _timed(
            "java_compile_repair",
            stages,
            lambda: run_java_compile_repair(
                code_agent=code_agent,
                ir=ir,
                code_dir=code_dir,
                stack=stack,
                generated_tests=generated_tests_dict,
                test_agent=test_agent_runner,
                dafny_proj=proj if adapter_mode else None,
                adapter_modules=adapter_modules,
                max_rounds=cfg.max_compile_repair_loops,
                timeout=max(240.0, cfg.case_timeout),
            ),
        )
        metrics["java_compile_repair"] = compile_repair
        repaired = compile_repair.get("repaired_files") or []
        if repaired:
            prev_code = metrics.get("code") or {}
            next_code = code_stats(ir["generated_code"])
            next_code["language"] = "Java"
            if prev_code.get("runnable_app"):
                next_code["runnable_app"] = prev_code["runnable_app"]
            if prev_code.get("dafny_cores"):
                next_code["dafny_cores"] = prev_code["dafny_cores"]
            metrics["code"] = next_code
        if compile_repair.get("skipped") and compile_repair.get("ok"):
            print(f"  Java static compile partially checked: {compile_repair.get('reason', 'tooling missing')}")
        elif compile_repair.get("skipped"):
            print(f"  Java compile repair skipped: {compile_repair.get('reason', 'unknown')}")
        elif compile_repair.get("ok"):
            print(f"  Java static compile passed after {len(compile_repair.get('attempts', []))} check(s); "
                  f"repaired {len(repaired)} generated file(s).")
        else:
            print(f"  Java static compile still failing after "
                  f"{len(compile_repair.get('attempts', []))} check(s): "
                  f"{compile_repair.get('reason', 'compile failed')}")
            for err in (compile_repair.get("unmapped_errors") or [])[:3]:
                print(f"    unmapped: {err.get('path')}:{err.get('line')} {err.get('message')}")

    # --- Stage 6: Certification (optional) — pass@k for the UN-proved (test fallback) --
    from src.cleanroom.targets import get_target as _get_target
    if "certification" in done:
        _banner("[6]", "Certification — [skip — resumed]")
        proved_feature_ids = set(ir.get("proved_feature_ids") or proved_feature_ids)
        print(f"  loaded certification from checkpoint: "
              f"{(metrics.get('certification') or {}).get('aggregate_pass_at', {})}")
    elif certify and not _get_target(language, stack).oracle_available():
        _banner("[6]", "Certification — SKIPPED (no executable oracle for this target)")
        print(f"  No executable oracle for language='{language}' (e.g. javac/JDK missing) — "
              f"code + tests were generated, but pass@k is unavailable.")
    elif certify:
        skip_note = f" · skipping {len(cert_skip)} proved" if cert_skip else ""
        _banner("[6]", f"Certification Agent — pass@k (n={samples} samples){skip_note}")
        cert = _timed(
            "certification",
            stages,
            lambda: CertificationAgent(
                code_llm=cert_llm,
                n=samples,
                k_values=cfg.k_values,
                stack=stack,
                language=language,
                case_timeout=cfg.case_timeout,
                skip_feature_ids=cert_skip,
                dafny_proj=proj if adapter_mode else None,
                dafny_modules=[proved_modules[fid] for fid in sorted(adapter_feature_ids)] if adapter_mode else None,
            ).run(ir),
        )
        ir = {**ir, "certification": cert.model_dump()}
        metrics["certification"] = {
            "n": cert.n,
            "k_values": cert.k_values,
            "aggregate_pass_at": cert.aggregate_pass_at,
            "aggregate_case_pass_rate": cert.aggregate_case_pass_rate,
            "n_tested_frs": len(cert.frs),
            "n_proved_features": len(proved_feature_ids),
            "n_total_tested_features": len(cert.features),
            # FR ids passing pass@1 — unioned with proved features' FRs for FR-level PassVer@1.
            "pass_at_1_fr_ids": sorted(
                fr.fr_id for fr in cert.frs if fr.pass_at.get("pass@1", 0.0) >= 1.0),
        }
        if not cert.frs:
            print("  (no features to test — all were certified by the proof tier)")
        else:
            for k, v in cert.aggregate_pass_at.items():
                print(f"  {k}: {v:.3f}  (over {len(cert.frs)} un-proved FR(s))")

        # --- Stage 6b: Recovery loop (opt-in via --max-cert-loops) ----------------
        # For each feature still failing pass@1, re-prove (escalated rounds) and, if it stays
        # unproved, regenerate its code WITH the failing test cases (the contained, user-approved
        # clean-room break), then re-certify — up to `max_cert_loops` times.
        from src.cleanroom.agents.recovery.loop import RecoveryLoop, failing_feature_ids
        # Recovery's re-prove + test-informed regen path is Python-specific in v1.
        if max_cert_loops > 0 and language == "python" and cert.features and failing_feature_ids(cert):
            _banner("[6b]", f"Recovery loop — re-prove + test-informed regen (≤ {max_cert_loops} passes)")
            loop = RecoveryLoop(
                ir,
                stack=stack,
                proj=proj,
                proved_feature_ids=set(proved_feature_ids),
                proved_modules=proved_modules,
                proved_sources=proved_sources,
                code_dir=code_dir,
                adapter_mode=adapter_mode,
                max_loops=max_cert_loops,
                prove_target=prove_target,
                # Live per-iteration metrics, flushed each loop → survives an early stop.
                iter_log=output_dir / f"{ir['project_name']}_recovery_iters.jsonl",
                repair_driver=cfg.repair_driver,
            )
            recovery = _timed("recovery", stages, lambda: loop.run(cert))
            cert = recovery["result"]
            proved_feature_ids = set(loop.proved_feature_ids)
            ir = {**ir, "certification": cert.model_dump(),
                  "proved_feature_ids": sorted(proved_feature_ids),   # final set incl. recovery re-proofs
                  "recovery": {k: v for k, v in recovery.items() if k != "result"}}
            metrics["certification"]["aggregate_pass_at"] = cert.aggregate_pass_at
            metrics["certification"]["aggregate_case_pass_rate"] = cert.aggregate_case_pass_rate
            metrics["certification"]["n_proved_features"] = len(proved_feature_ids)
            metrics["certification"]["n_total_tested_features"] = len(cert.features)
            metrics["certification"]["pass_at_1_fr_ids"] = sorted(
                fr.fr_id for fr in cert.frs if fr.pass_at.get("pass@1", 0.0) >= 1.0)
            metrics["recovery"] = {
                "max_loops": max_cert_loops,
                "loops_run": len(recovery["loops"]),
                "repaired_by_proof": recovery["repaired_by_proof"],
                "repaired_by_tests": recovery["repaired_by_tests"],
                "uncertified": recovery["uncertified"],
                "labels": recovery["labels"],
                "loops": recovery["loops"],
            }
            print(f"  recovery: +{len(recovery['repaired_by_proof'])} re-proved, "
                  f"+{len(recovery['repaired_by_tests'])} repaired-with-tests, "
                  f"{len(recovery['uncertified'])} uncertified  ·  "
                  f"pass@1 now {cert.aggregate_pass_at.get('pass@1', 0.0):.3f}")
            for fid, lab in sorted(recovery["labels"].items()):
                print(f"    [{fid}] {lab}")

    if certify and "certification" not in done:
        _checkpoint("certification", ir)

    _finalize_metrics(metrics, stages)
    metrics["proved_feature_ids"] = sorted(proved_feature_ids)   # final set (incl. recovery)
    metrics["summary"] = _compute_summary(metrics, ir)
    _print_summary(metrics["summary"])
    # A fully-completed run clears its checkpoint so the next invocation starts fresh.
    ckpt_path.unlink(missing_ok=True)
    return ir, metrics


def write_run_report(metrics: dict, dest: Path) -> None:
    lines = [
        "# Run report",
        "",
        f"_SRS: {metrics['srs']} · language: {metrics.get('language', 'python')} · "
        f"model: {metrics.get('model', 'n/a')}_",
        "",
    ]
    s = metrics.get("summary") or {}
    if s:
        lines += ["## Run metrics", ""]
        for k in ("verification_pass_ratio", "test_case_pass_ratio", "PassVer@1",
                  "avg_verification_iteration", "avg_test_iteration",
                  "avg_input_token", "avg_output_token", "total_time"):
            lines.append(f"- **{k}**: {s.get(k)}")
        lines.append("")
    lines += [
        "## Timing & tokens (per stage)",
        "",
        "| stage | seconds | input tok | output tok | calls |",
        "|---|---|---|---|---|",
    ]
    for s in metrics["stages"]:
        lines.append(f"| {s['name']} | {s['seconds']:.2f} | {s['input_tokens']:,} | {s['output_tokens']:,} | {s['calls']} |")
    tok = metrics.get("tokens", {"input": 0, "output": 0, "total": 0, "calls": 0})
    lines += [
        f"| **total** | **{metrics.get('total_seconds', 0):.2f}** | **{tok['input']:,}** | **{tok['output']:,}** | **{tok['calls']}** |",
        "",
        "## Tokens & cost",
        "",
        f"- total tokens: {tok['total']:,}  (input {tok['input']:,} / output {tok['output']:,})",
        f"- LLM calls: {tok['calls']}",
        f"- estimated cost: ${metrics.get('cost_usd', 0):.4f}",
    ]
    by_model = metrics.get("cost_by_model") or {}
    if len(by_model) > 1:
        lines.append("- per-model (accurate mixed-model cost):")
        for m, agg in sorted(by_model.items()):
            lines.append(f"  - {m}: {agg['calls']} call(s), "
                         f"{agg['input'] + agg['output']:,} tok, ${agg['cost_usd']:.4f}")
    if metrics.get("code"):
        c = metrics["code"]
        stack = ", ".join(c["third_party_libraries"]) or "Python standard library only"
        lines += [
            "",
            "## Generated code",
            "",
            f"- language: {c['language']}",
            f"- files: {c['files']} · lines of code: {c['lines_of_code']:,}",
            f"- files per layer: {c['files_per_layer']}",
            f"- tech stack: {stack}",
        ]
    if metrics.get("tests"):
        t = metrics["tests"]
        lines += ["", "## Generated tests", "", f"- features: {t['features']} · test cases: {t['cases']}"]
    if metrics.get("java_compile_repair"):
        r = metrics["java_compile_repair"]
        status = (
            "partial" if r.get("ok") and r.get("skipped")
            else "passed" if r.get("ok")
            else "skipped" if r.get("skipped")
            else "failed"
        )
        lines += [
            "",
            "## Java compile repair",
            "",
            f"- status: {status}",
            f"- checks: {len(r.get('attempts') or [])} · repaired files: {len(r.get('repaired_files') or [])}",
            f"- reason: {r.get('reason', '')}",
        ]
    if metrics.get("certification"):
        cert = metrics["certification"]
        lines += ["", "## Certification (pass@k)", "", f"- samples n={cert['n']}"]
        lines += [f"- {k}: {v:.3f}" for k, v in cert["aggregate_pass_at"].items()]
        if "aggregate_case_pass_rate" in cert:
            lines.append(f"- case pass rate: {cert['aggregate_case_pass_rate']:.3f}")
    if metrics.get("recovery"):
        r = metrics["recovery"]
        lines += [
            "",
            "## Recovery loop (re-prove + test-informed regen)",
            "",
            f"- passes run: {r['loops_run']} / {r['max_loops']} max",
            f"- re-proved (now ship from Dafny): {r['repaired_by_proof'] or '(none)'}",
            f"- repaired with test feedback (clean-room relaxed): {r['repaired_by_tests'] or '(none)'}",
            f"- still UNCERTIFIED: {r['uncertified'] or '(none)'}",
            "",
            "Final per-feature certification label:",
        ]
        for fid, lab in sorted(r["labels"].items()):
            lines.append(f"  - {fid}: {lab}")
    if metrics.get("dafny"):
        d = metrics["dafny"]
        lines += [
            "",
            "## Dafny proof tier (formal verification)",
            "",
            f"- model: {d.get('model', 'n/a')}",
            f"- features: {d['n_features']} · proved: {d['n_verified']} · rate: {d['verification_rate']:.3f}",
        ]
        if d.get("n_proved_with_axioms"):
            lines.append(f"- of which {d['n_proved_with_axioms']} rely on assumed axioms (not fully discharged)")
        for f in d["features"]:
            mark = "PROVED" if f["verified"] else "unproved (→ test track)"
            axt = f" [{f['axioms']} assumed axiom(s)]" if f.get("axioms") else ""
            lines.append(f"  - {f['feature_id']} ({f['module']}): {mark} ({f['rounds']}r){axt}")
    if metrics.get("compile"):
        c = metrics["compile"]
        lines += [
            "",
            f"## Native compile (dafny translate {c.get('target', 'py')})",
            "",
            f"- compiled: {len(c.get('compiled', []))} · failed: {len(c.get('failed', []))}",
        ]
    dest.write_text("\n".join(lines) + "\n")


RECORD_FILE = Path("Full_Pipeline_run_metric_record_with_differentModels.md")


def append_metric_record(
    metrics: dict, cfg: RunConfig, command: str, status: str, dest: Path | None = None
) -> None:
    """Append one structured section per run to the cross-model comparison ledger.

    Captures: which models drove each stage, the exact command + pipeline flags, the
    headline summary metrics (PassVer@1 etc.), the Dafny proof results per feature, the
    pass@k certification, the per-recovery-loop progress (targeted/reproved/still-failing/
    pass@1), and the final per-feature outcome label (PROVED / TESTED / UNCERTIFIED).
    Appended for every run — completed or failed — so models can be A/B'd from one file.
    """
    from datetime import datetime

    # Ledger path: env override (PIPELINE_LEDGER_FILE) lets parallel runs each write their OWN
    # ledger so concurrent appends never interleave; a wrapper merges them afterwards. Default
    # is the shared repo-root ledger for normal single runs.
    if dest is None:
        dest = Path(os.getenv("PIPELINE_LEDGER_FILE") or RECORD_FILE)

    s = metrics.get("summary") or {}
    models = (metrics.get("config") or {}).get("models") or cfg.models_used()
    agents = (metrics.get("config") or {}).get("agents") or {}
    tok = metrics.get("tokens") or {}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L: list[str] = []

    if not dest.exists():
        L += [
            "# Full pipeline — run metrics across different models",
            "",
            "_Auto-appended after every `run_pipeline.py` run. One section per run._",
            "",
        ]

    L += [
        "---",
        "",
        f"## Run {metrics.get('run_id', ts)} — {status.upper()}",
        "",
        f"- **When**: {ts}",
        f"- **SRS**: {metrics.get('srs', '?')}  ·  **language**: {metrics.get('language', '?')}"
        f"  ·  **stack**: {metrics.get('stack', '?')}",
        f"- **Command**: `{command}`",
        "",
        "### Models used (per stage)",
        "",
        "| stage | model |",
        "|---|---|",
    ]
    for stage in ("spec", "dependency", "planning", "code", "test", "proof", "cert"):
        if stage in models:
            L.append(f"| {stage} | `{models[stage]}` |")

    arm = ", ".join(f"{k}={'on' if v else 'off'}" for k, v in agents.items())
    L += [
        "",
        "### Pipeline flags",
        "",
        f"- agents: {arm}",
        f"- certify={cfg.certify} · samples={cfg.samples} · k={list(cfg.k_values)} · "
        f"prove={cfg.prove} · max_cert_loops={cfg.max_cert_loops} · "
        f"max_compile_repair_loops={cfg.max_compile_repair_loops} · "
        f"repair_driver={cfg.repair_driver}",
        f"- temperature={cfg.temperature} · cert_temperature={cfg.cert_temperature} · "
        f"prove_rounds={cfg.prove_rounds} · baseline={cfg.baseline}",
        "",
    ]

    if s:
        L += [
            "### Run metrics (summary)",
            "",
            "| metric | value |",
            "|---|---|",
            f"| PassVer@1 | {s.get('PassVer@1')} |",
            f"| verification_pass_ratio | {s.get('verification_pass_ratio')} |",
            f"| test_case_pass_ratio | {s.get('test_case_pass_ratio')} |",
            f"| avg_verification_iteration | {s.get('avg_verification_iteration')} |",
            f"| avg_test_iteration | {s.get('avg_test_iteration')} |",
            f"| avg_input_token | {s.get('avg_input_token')} |",
            f"| avg_output_token | {s.get('avg_output_token')} |",
            f"| total_time (s) | {s.get('total_time')} |",
            "",
            "#### Copy-paste row (TSV — paste directly under your spreadsheet header)",
            "",
            "```text",
            "verification_pass_ratio\ttest_case_pass_ratio\tavg_input_token\t"
            "avg_output_token\ttotal_time\tavg_verification_iter\tavg_test_iter\tPassVer@1",
            "\t".join(
                str(s.get(k, ""))
                for k in (
                    "verification_pass_ratio",
                    "test_case_pass_ratio",
                    "avg_input_token",
                    "avg_output_token",
                    "total_time",
                    "avg_verification_iteration",
                    "avg_test_iteration",
                    "PassVer@1",
                )
            ),
            "```",
            "",
        ]
    L += [
        f"- tokens: {tok.get('total', 0):,} (in {tok.get('input', 0):,} / "
        f"out {tok.get('output', 0):,}) · calls {tok.get('calls', 0)}",
        f"- estimated cost: ${metrics.get('cost_usd', 0):.4f}",
        "",
    ]
    by_model = metrics.get("cost_by_model") or {}
    if len(by_model) > 1:
        L.append("- cost by model:")
        for m, agg in sorted(by_model.items()):
            L.append(
                f"  - `{m}`: {agg['calls']} call(s), "
                f"{agg['input'] + agg['output']:,} tok, ${agg['cost_usd']:.4f}"
            )
        L.append("")

    d = metrics.get("dafny") or {}
    if d:
        L += [
            "### Dafny proof tier",
            "",
            f"- proved {d.get('n_verified', 0)}/{d.get('n_features', 0)} features "
            f"(rate {d.get('verification_rate', 0)})",
            "",
            "| feature | result | rounds | axioms |",
            "|---|---|---|---|",
        ]
        for f in d.get("features", []):
            mark = "PROVED" if f["verified"] else "unproved"
            L.append(f"| {f['feature_id']} | {mark} | {f['rounds']} | {f.get('axioms', 0)} |")
        L.append("")

    cert = metrics.get("certification") or {}
    if cert:
        L += ["### Certification (pass@k)", ""]
        for k, v in (cert.get("aggregate_pass_at") or {}).items():
            L.append(f"- {k}: {v:.3f}")
        L += [
            f"- case pass rate: {cert.get('aggregate_case_pass_rate', 0):.3f}",
            f"- pass@1 FR ids: {cert.get('pass_at_1_fr_ids') or '(none)'}",
            "",
        ]

    crep = metrics.get("java_compile_repair") or {}
    if crep:
        crep_status = (
            "partial" if crep.get("ok") and crep.get("skipped")
            else "passed" if crep.get("ok")
            else "skipped" if crep.get("skipped")
            else "failed"
        )
        L += [
            "### Java compile repair",
            "",
            f"- status: {crep_status}",
            f"- checks: {len(crep.get('attempts') or [])}",
            f"- repaired files: {len(crep.get('repaired_files') or [])}",
            f"- reason: {crep.get('reason', '')}",
            "",
        ]

    rec = metrics.get("recovery") or {}
    if rec:
        L += [
            "### Recovery loops (full metrics per iteration)",
            "",
            "| loop | targeted | reproved | still failing | pass@1 | PassVer@1 | case_pass | proved FRs | certified FRs |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for lp in rec.get("loops", []):
            tot = lp.get("total_frs", "?")
            L.append(
                f"| {lp['loop']} | {lp['targeted']} | {lp['reproved'] or '—'} | "
                f"{lp['still_failing'] or '—'} | {lp['pass_at_1']} | "
                f"{lp.get('passver_at_1', '—')} | {lp.get('case_pass_rate', '—')} | "
                f"{lp.get('proved_frs', '—')}/{tot} | {lp.get('certified_frs', '—')}/{tot} |"
            )
        L += [
            "",
            f"- re-proved (now ship from Dafny): {rec.get('repaired_by_proof') or '(none)'}",
            f"- repaired-with-tests (clean-room relaxed): {rec.get('repaired_by_tests') or '(none)'}",
            f"- still UNCERTIFIED: {rec.get('uncertified') or '(none)'}",
            "",
        ]

    labels = dict(rec.get("labels") or {})
    if not labels:  # no recovery this run — synthesize from proved set
        for fid in sorted(metrics.get("proved_feature_ids") or []):
            labels[fid] = "PROVED"
    if labels:
        L += ["### Final per-feature outcome", "", "| feature | outcome |", "|---|---|"]
        for fid, lab in sorted(labels.items()):
            L.append(f"| {fid} | {lab} |")
        L.append("")
    L += [f"- proved feature ids: {metrics.get('proved_feature_ids') or '(none)'}", ""]

    with dest.open("a") as fh:
        fh.write("\n".join(L) + "\n")


def _persist_run_artifacts(
    metrics: dict,
    *,
    status: str,
    output_dir: Path,
    srs_stem: str,
) -> dict:
    """Write per-run record, update RUN_RESULTS.md and API_USAGE.md."""
    run_id = make_run_id(metrics.get("srs") or srs_stem)
    metrics["run_id"] = run_id
    runs_dir = output_dir / "runs"
    recorded = record_run(metrics, status=status, runs_dir=runs_dir)
    usage_log = append_usage_log(
        metrics,
        status=status,
        run_id=run_id,
        result_path=recorded["markdown"],
    )
    recorded["usage_log"] = usage_log
    return recorded


def main() -> None:
    # ===== CLI definition: target, per-agent ON/OFF switches, per-stage models, tuning knobs =====
    parser = argparse.ArgumentParser(description="Run the full Agentic Cleanroom pipeline on an SRS document.")
    parser.add_argument("srs_path", type=Path, help="Path to the SRS XML document")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to write artifacts")
    parser.add_argument("--language", default="python", choices=list(LANGUAGES),
                        help="Target language for code + tests (default: python). "
                             "'java' → Java code + JUnit tests. With --language java the Java "
                             "sub-stack is 'java' (plain) by default, or 'spring' for Spring Boot.")
    # --- Per-agent ON/OFF switches (compose any arm). Required agents — Spec, Planning, Code —
    #     have no switch: they are the irreducible spec→code task given to every arm. ---
    parser.add_argument("--dependency", action=argparse.BooleanOptionalAction, default=True,
                        help="Dependency agent (FR graph). --no-dependency skips it (empty graph; "
                             "features independent, spec order).")
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=True,
                        help="Test agent (black-box tests). --no-test skips test generation.")
    parser.add_argument("--recovery", action=argparse.BooleanOptionalAction, default=True,
                        help="Recovery loop after certification (needs --certify). "
                             "--no-recovery is equivalent to --max-cert-loops 0.")
    parser.add_argument("--repair-driver", choices=("deterministic", "deepagent"),
                        default="deterministic",
                        help="Who drives the repair loops (Java compile repair, recovery regen). "
                             "'deterministic' (default) = the fixed-round loops that produced the "
                             "recorded results; 'deepagent' = a LangChain deepagents agent plans and "
                             "re-checks on its own (needs the `deepagents` package).")
    parser.add_argument("--certify", action=argparse.BooleanOptionalAction, default=False,
                        help="Run the pass@k certification stage (--no-certify to skip; default off).")
    parser.add_argument("--samples", type=int, default=1,
                        help="Code samples per FR for certification (default: 1 → pass@1 only). "
                             "Pass a higher N with explicit --k for a multi-sample estimate.")
    parser.add_argument("--k", type=int, nargs="+", default=None, metavar="K",
                        help="pass@k values to report (default: 1). e.g. --k 1 3")
    parser.add_argument("--stack", default="auto", choices=["auto", "python", "fastapi", "java", "spring", "express"],
                        help="Sub-stack within a language. Python: 'auto' (default, from the SRS), "
                             "'python', or 'fastapi'. Java: 'java' (plain, default) or 'spring' "
                             "(Spring Boot web). 'auto' under --language java resolves to plain java.")
    # Model selection. The simplest path is --model, which runs EVERY stage (proof included) on one
    # model. The per-stage flags below override individual stages and always win over --model.
    # With no model flags at all: every stage uses DEFAULT_MODEL, and the proof tier uses DAFNY_MODEL.
    parser.add_argument("--model", default=None,
                        help="Single model for ALL stages (proof included) — shorthand for setting "
                             "every --*-model at once. Per-stage flags below override it.")
    parser.add_argument("--spec-model", default=None, help="Override model for the Spec/contracts stage.")
    parser.add_argument("--dependency-model", default=None, help="Override model for the Dependency stage.")
    parser.add_argument("--planning-model", default=None, help="Override model for the Planning stage.")
    parser.add_argument("--code-model", default=None, help="Override model for the Code Agent.")
    parser.add_argument("--test-model", default=None, help="Override model for the Test Agent.")
    parser.add_argument("--proof-model", default=None, help="Override model for the Dafny proof tier (needs a strong model).")
    parser.add_argument("--cert-model", default=None, help="Override model for certification samples / recovery regen.")
    parser.add_argument("--prove", action=argparse.BooleanOptionalAction, default=False,
                        help="Dafny proof tier: prove each feature's logic where possible "
                             "(needs a `dafny` binary + the configured proof model). Proved features compile to native "
                             "code; with --certify the rest fall to the pass@k test track. "
                             "--no-prove (default) skips it.")
    parser.add_argument("--prove-target", default=None,
                        choices=["py", "cs", "js", "go", "java", "cpp", "rs"],
                        help="dafny translate target for proved features "
                             "(default: follows --language — py for python, java for java).")
    parser.add_argument("--max-cert-loops", type=int, default=2,
                        help="Recovery loop: max times to re-prove + regenerate-with-test-feedback "
                             "+ re-certify features that fail pass@1 (default: 2; needs --certify; "
                             "0 disables). NOTE: the test-feedback regen deliberately relaxes "
                             "clean-room isolation for those failing features only.")
    parser.add_argument("--max-compile-repair-loops", type=int, default=2,
                        help="Java only: max static compile-repair passes before certification "
                             "(default: 2; 0 disables). Uses compiler diagnostics only, not "
                             "runtime test outcomes.")
    # Per-agent tuning knobs.
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature for spec/planning/code/test (default: 0.0).")
    parser.add_argument("--cert-temperature", type=float, default=0.4,
                        help="Sampling temperature for certification code samples (pass@k diversity; default: 0.4).")
    parser.add_argument("--case-timeout", type=float, default=10.0,
                        help="Per-test-case execution timeout in seconds (default: 10.0).")
    parser.add_argument("--prove-rounds", type=int, default=6,
                        help="Dafny generate→verify→revise rounds per feature (default: 6).")
    parser.add_argument("--llm-deps", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the LLM to infer semantic FR→FR edges in the dependency stage "
                             "(--no-llm-deps = deterministic regex edges only).")
    parser.add_argument("--baseline", action="store_true",
                        help="Control run: forces proof OFF, recovery OFF, regex-only dependencies, "
                             "and temperature 0 — a plain spec→code→test(→pass@k) baseline to A/B "
                             "against the full prove-or-test pipeline.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the per-stage checkpoint (outputs/<stem>_ckpt.json): reload "
                             "the IR + metrics + token accounting and SKIP every stage already "
                             "completed, continuing from the one that failed. A successful run clears "
                             "the checkpoint. Run the SAME command with --resume after a failure.")
    parser.add_argument("--debug-llm-once", action="store_true",
                        help="Print the first LLM prompt, raw model response, and reasoning-token "
                             "usage fields before structured-output parsing. Debug only.")
    args = parser.parse_args()
    command = "python run_pipeline.py " + " ".join(sys.argv[1:])

    if not args.srs_path.exists():
        parser.error(f"SRS file not found: {args.srs_path}")

    if args.debug_llm_once:
        os.environ["CLEANROOM_LLM_DEBUG_ONCE"] = "1"

    if not llm_api_key_configured():
        print(
            "ERROR: No LLM endpoint configured. Set LLM_BASE_URL (self-hosted server) "
            "or OPENAI_API_KEY (api.openai.com) in .env."
        )
        sys.exit(1)

    # ===== Execute: run the pipeline → persist artifacts → append the model-comparison ledger =====
    # On success we write the IR, run report, per-run record, and the cross-model metric ledger.
    # On failure we still persist whatever partial metrics were collected (status="failed").
    metrics: dict | None = None
    cfg = RunConfig.from_args(args)
    try:
        ir, metrics = run(args.srs_path, args.output_dir, cfg)
        ir = {**ir, "metrics": metrics}
        normalize_ir(ir)

        dest = args.output_dir / f"{args.srs_path.stem}_full_ir.json"
        dest.write_text(json.dumps(ir, indent=2))
        report = args.output_dir / f"{args.srs_path.stem}_run_report.md"
        write_run_report(metrics, report)

        recorded = _persist_run_artifacts(
            metrics,
            status="complete",
            output_dir=args.output_dir,
            srs_stem=args.srs_path.stem,
        )
        append_metric_record(metrics, cfg, command, status="complete")
        _banner("DONE", "Run complete")
        print(f"IR        : {dest}")
        print(f"Report    : {report}")
        print(f"Run record: {recorded['markdown']}")
        print(f"Run JSON  : {recorded['json']}")
        print(f"Run index : {recorded['index']}")
        print(f"Usage log : {recorded['usage_log']}")
        per_stage = "   ".join(f"{s['name']}={s['seconds']:.1f}s" for s in metrics["stages"] if s["seconds"] >= 0.01)
        print(f"Per stage : {per_stage}")
        print(f"Time      : {metrics['total_seconds']:.2f}s   Tokens: {metrics['tokens']['total']:,}   Cost: ${metrics['cost_usd']:.4f}")
    except Exception:
        partial = (
            metrics
            if metrics is not None
            else metrics_from_globals(
                srs=args.srs_path.name,
                stack=cfg.stack,
                stages=[],
                config=cfg.as_dict(),
            )
        )
        if metrics is not None:
            _finalize_metrics(partial, partial.get("stages", []))
        recorded = _persist_run_artifacts(
            partial,
            status="failed",
            output_dir=args.output_dir,
            srs_stem=args.srs_path.stem,
        )
        append_metric_record(partial, cfg, command, status="failed")
        print(f"Run record: {recorded['markdown']}  (partial)")
        print(f"Usage log : {recorded['usage_log']}  (partial tokens recorded)")
        raise


if __name__ == "__main__":
    main()
