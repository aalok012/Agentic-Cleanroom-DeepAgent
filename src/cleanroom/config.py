"""Central run configuration — every pipeline knob, populated from the CLI.

One object threaded through `run()` so each stage can pick its own model, the target language,
pass@k, etc. Defaults preserve today's behavior (all stages on `DEFAULT_MODEL`, proof on
`DAFNY_MODEL`, language=python). See `RunConfig.from_args` for the CLI mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.cleanroom.utils.llm_client import DAFNY_MODEL, DEFAULT_MODEL

LANGUAGES = ("python", "java", "javascript")


@dataclass
class RunConfig:
    # --- target ---
    language: str = "python"          # python | java  (the top-level axis the user picks)
    stack: str = "auto"               # python sub-stack: auto | python | fastapi (ignored for java)

    # --- prompt strategy ---
    # 'baseline' = the original prompts (unchanged); 'cot' = parallel Chain-of-Thought variants
    # that make each agent reason step-by-step BEFORE emitting its (unchanged) structured output;
    # 'mot' = Module-of-Thought variants that decompose into private helpers, implement, then
    # compose the public entry (Planning/Code/Test only; other stages fall back to the CoT prompt).
    prompt_strategy: str = "baseline"

    # --- per-stage models (each stage can run a different model) ---
    spec_model: str = DEFAULT_MODEL
    dependency_model: str = DEFAULT_MODEL
    planning_model: str = DEFAULT_MODEL
    code_model: str = DEFAULT_MODEL
    test_model: str = DEFAULT_MODEL
    proof_model: str = DAFNY_MODEL
    cert_model: str = DEFAULT_MODEL

    # --- certification / proof ---
    certify: bool = False
    samples: int = 1
    k_values: tuple[int, ...] = (1,)
    prove: bool = False
    prove_target: str = "py"
    max_cert_loops: int = 2
    max_compile_repair_loops: int = 2

    # --- repair driver ---
    # 'deterministic' (default) = the fixed-round loops that produced every existing result;
    # 'deepagent' = LangChain deepagents drives the compile-repair and recovery-regen steps,
    # planning and re-checking on its own. Opt-in so recorded runs stay reproducible.
    repair_driver: str = "deterministic"

    # --- generation driver ---
    # 'deterministic' (default) = one structured call per contract/feature, as every recorded
    # result was produced. 'deepagent' = the planning, code, test and proof stages each run as
    # a deepagents agent that reads its own briefing and submits through validated tools.
    # Opt-in, so both arms stay runnable for comparison.
    gen_driver: str = "deterministic"

    # --- per-agent ON/OFF switches (compose any arm; required agents Spec/Planning/Code
    #     have no switch — they are the irreducible spec->code task given to every arm) ---
    run_dependency: bool = True       # Dependency agent (semantic FR graph). Off => empty graph.
    run_test: bool = True             # Test agent (black-box tests). Off => no test generation.
    run_recovery: bool = True         # Recovery loop after certification (off == max_cert_loops 0).

    # --- per-agent knobs ---
    temperature: float = 0.0          # generation temperature: spec / planning / code / test
    cert_temperature: float = 0.4     # sampling temperature for certification code samples (pass@k diversity)
    case_timeout: float = 10.0        # per-test-case execution timeout (s)
    prove_rounds: int = 6             # Dafny generate->verify->revise rounds per feature
    llm_deps: bool = True             # use the LLM to infer semantic FR->FR edges (else regex-only)
    baseline: bool = False            # control run: no proof, no recovery, regex-only deps, temp 0
    resume: bool = False              # resume from the per-stage checkpoint, skipping completed stages

    def models_used(self) -> dict[str, str]:
        """Stage -> model, for the run report / debugging."""
        return {
            "spec": self.spec_model, "dependency": self.dependency_model,
            "planning": self.planning_model, "code": self.code_model,
            "test": self.test_model, "proof": self.proof_model, "cert": self.cert_model,
        }

    def agents_enabled(self) -> dict[str, bool]:
        """Which agents are ON for this run (required ones are always True)."""
        return {
            "spec": True, "dependency": self.run_dependency, "planning": True,
            "proof": self.prove, "code": True, "test": self.run_test,
            "certify": self.certify,
            # recovery only fires after a real certification pass
            "recovery": self.certify and self.run_recovery and self.max_cert_loops > 0,
            "compile_repair": self.language == "java" and self.max_compile_repair_loops > 0,
        }

    def uses_deep_agent(self) -> bool:
        """True when the deepagents-driven repair loops replace the deterministic ones."""
        return self.repair_driver == "deepagent"

    def uses_deep_generation(self) -> bool:
        """True when planning/code/test/proof run as deepagents rather than single calls."""
        return self.gen_driver == "deepagent"

    def as_dict(self) -> dict:
        return {
            "language": self.language, "stack": self.stack, "models": self.models_used(),
            "prompt_strategy": self.prompt_strategy,
            "agents": self.agents_enabled(),
            "run_dependency": self.run_dependency, "run_test": self.run_test,
            "run_recovery": self.run_recovery,
            "certify": self.certify, "samples": self.samples, "k_values": list(self.k_values),
            "prove": self.prove, "prove_target": self.prove_target,
            "max_cert_loops": self.max_cert_loops,
            "max_compile_repair_loops": self.max_compile_repair_loops,
            "repair_driver": self.repair_driver,
            "gen_driver": self.gen_driver,
            "temperature": self.temperature,
            "cert_temperature": self.cert_temperature, "case_timeout": self.case_timeout,
            "prove_rounds": self.prove_rounds, "llm_deps": self.llm_deps, "baseline": self.baseline,
        }

    @classmethod
    def from_args(cls, args) -> RunConfig:
        """Build from an argparse Namespace. Java fixes its own stack; --prove-target follows the
        language unless overridden. `--baseline` is a control preset: it forces proof OFF, recovery
        OFF, regex-only dependencies, and temperature 0 (other flags still apply)."""
        language = getattr(args, "language", "python")
        baseline = getattr(args, "baseline", False)
        # Java sub-stacks: plain `java` (default) or `spring` (Spring Boot web). Anything else the
        # user might pass (python/fastapi/auto) is not a Java stack, so it falls back to plain java.
        java_stack = getattr(args, "stack", "java")
        java_stack = java_stack if java_stack in ("java", "spring") else "java"
        # JS has a single stack (Node/Express + SQLite); Java picks plain/spring; python keeps its
        # raw --stack (auto|python|fastapi).
        if language == "java":
            resolved_stack = java_stack
        elif language == "javascript":
            resolved_stack = "express"
        else:
            resolved_stack = getattr(args, "stack", "auto")
        # --prove-target follows the language: java->java, else py. JS ships no Dafny-core adapter
        # (like Java), so its proved cores are unused — compiling to py is safe and the proof-tier
        # verification_rate (the recorded metric) is target-independent.
        prove_target = getattr(args, "prove_target", None) or ("java" if language == "java" else "py")
        k_values = tuple(getattr(args, "k", None) or (1,))
        # Per-agent switches (baseline forces the optional ones into their control state).
        run_dependency = getattr(args, "dependency", True)
        run_test = getattr(args, "test", True)
        run_recovery = False if baseline else getattr(args, "recovery", True)
        # Recovery off (explicit, or baseline) => 0 loops, regardless of --max-cert-loops.
        max_loops = 0 if (baseline or not run_recovery) else getattr(args, "max_cert_loops", 2)
        # Model resolution precedence (highest first): explicit per-stage flag -> --model -> stage
        # default. `--model X` thus runs every stage (proof included) on X; a per-stage flag still
        # overrides just that stage. With neither, stages fall back to DEFAULT_MODEL (proof: DAFNY_MODEL).
        one = getattr(args, "model", None)
        base = one or DEFAULT_MODEL          # default for all non-proof stages
        proof_base = one or DAFNY_MODEL       # proof keeps its stronger default unless --model is set
        return cls(
            language=language,
            stack=resolved_stack,
            prompt_strategy=getattr(args, "prompt_strategy", "baseline") or "baseline",
            spec_model=getattr(args, "spec_model", None) or base,
            dependency_model=getattr(args, "dependency_model", None) or base,
            planning_model=getattr(args, "planning_model", None) or base,
            code_model=getattr(args, "code_model", None) or base,
            test_model=getattr(args, "test_model", None) or base,
            proof_model=getattr(args, "proof_model", None) or proof_base,
            cert_model=getattr(args, "cert_model", None) or base,
            run_dependency=run_dependency,   # baseline keeps the deterministic graph (regex deps)
            run_test=run_test,
            run_recovery=run_recovery,
            certify=getattr(args, "certify", False),
            samples=getattr(args, "samples", 1),
            k_values=k_values,
            prove=False if baseline else getattr(args, "prove", False),
            prove_target=prove_target,
            max_cert_loops=max_loops,
            max_compile_repair_loops=getattr(args, "max_compile_repair_loops", 2),
            repair_driver=getattr(args, "repair_driver", None) or "deterministic",
            gen_driver=getattr(args, "gen_driver", None) or "deterministic",
            temperature=0.0 if baseline else getattr(args, "temperature", 0.0),
            cert_temperature=getattr(args, "cert_temperature", 0.4),
            case_timeout=getattr(args, "case_timeout", 10.0),
            prove_rounds=getattr(args, "prove_rounds", 6),
            llm_deps=False if baseline else getattr(args, "llm_deps", True),
            baseline=baseline,
            resume=getattr(args, "resume", False),
        )
