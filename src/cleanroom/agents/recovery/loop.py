"""Recovery loop — the prove-or-test fallback's last resort (Stage 4 ↔ Stage 6 cycle).

Runs ONLY when ``--certify`` is on (it needs the pass@k signal to know what failed). For each
feature still failing ``pass@1`` after the first certification it does, per iteration ``i``:

  (a) RE-PROVE it in Dafny with escalated rounds (6 + 2·i). If it now verifies and the run is
      in FastAPI adapter mode, the feature ships from the compiled Dafny core (thin adapter) and
      leaves the test pool entirely.
  (b) REGENERATE its code WITH the failing test cases fed back, at escalated temperature
      (0.4 + 0.2·i). This is the DELIBERATE, CONTAINED clean-room break the user approved: only
      here does code generation ever see tests, and only after a feature has already failed both
      proof and the clean-room first pass.
  (c) RE-CERTIFY (n=1, pass@1) the regenerated features against the SAME frozen spec-derived test
      suite (tests are never regenerated — the oracle must not move).

Outcome labels per feature, surfaced in the run report so we never overclaim:
  PROVED · TESTED (clean-room) · TESTED (repaired-with-tests) · UNCERTIFIED.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.cleanroom.agents.code.agent import CodeAgent
from src.cleanroom.agents.code.schema.code import GeneratedCode
from src.cleanroom.agents.dafny.agent import DafnyAgent
from src.cleanroom.agents.dafny.schema.dafny import GeneratedDafny
from src.cleanroom.agents.evaluation.agent import CertificationAgent
from src.cleanroom.agents.evaluation.schema.certification import CertificationResult
from src.cleanroom.utils.dafny_project import compile_dafny, stage_dafny_cores
from src.cleanroom.utils.llm_client import DAFNY_MODEL, get_llm
from src.cleanroom.utils.packager import build_runnable_package

BASE_ROUNDS = 6
BASE_TEMP = 0.4


def failing_feature_ids(result: CertificationResult) -> set[str]:
    """Features whose pass@1 < 1.0 (the user-chosen failing signal). Proved features were
    skipped by certification, so they never appear here."""
    return {f.feature_id for f in result.features if f.pass_at.get("pass@1", 0.0) < 1.0}


class RecoveryLoop:
    def __init__(
        self,
        ir: dict,
        *,
        stack: str,
        proj: Path | None,
        proved_feature_ids: set[str],
        proved_modules: dict[str, str],
        proved_sources: dict[str, str],
        code_dir: Path,
        adapter_mode: bool,
        max_loops: int = 2,
        prove_target: str = "py",
        dafny_model: str = DAFNY_MODEL,
        iter_log: Path | None = None,
        prompt_strategy: str = "baseline",
        repair_driver: str = "deterministic",
    ) -> None:
        self.ir = ir
        self.stack = stack
        # Strategy for the re-prove / regen / re-certify agents below — keep it identical to the
        # run's strategy so the recovery pass uses the same prompts as the first pass.
        self.prompt_strategy = prompt_strategy
        self.proj = Path(proj) if proj else None
        self.proved_feature_ids = set(proved_feature_ids)
        self.proved_modules = dict(proved_modules)
        self.proved_sources = dict(proved_sources)
        self.code_dir = Path(code_dir)
        self.adapter_mode = adapter_mode
        self.max_loops = max_loops
        self.prove_target = prove_target
        self.dafny_model = dafny_model
        # 'deterministic' = one independent regeneration call per failing FR (the recorded
        # behaviour). 'deepagent' = one agent holds every failing FR of the feature at once,
        # so a failure caused by two FRs disagreeing about a shared shape is fixable. The
        # re-certification in step (c) stays OUTSIDE the agent either way — the agent never
        # gets to iterate against the oracle it is scored on.
        self.repair_driver = repair_driver
        self.regen_metrics: list[dict] = []
        # If set, each iteration's full metrics are flushed here as a JSON line the moment it
        # finishes — so you can stop the run early (Ctrl-C / kill) and keep the per-iteration data.
        self.iter_log = Path(iter_log) if iter_log else None
        # Adapters are spec+Dafny only (no test feedback), so a base-temperature agent writes them.
        self._adapter_agent = CodeAgent(llm=get_llm(temperature=0.0), stack=stack,
                                        prompt_strategy=prompt_strategy)

    # --- public ----------------------------------------------------------------
    def run(self, initial_result: CertificationResult) -> dict:
        """Drive the loop from the Stage-6 result. Returns an outcome dict and mutates ``ir``."""
        result = initial_result
        first_failing = failing_feature_ids(result)
        loops: list[dict] = []
        repaired_by_proof: set[str] = set()
        repaired_by_tests: set[str] = set()

        for i in range(self.max_loops):
            failing = failing_feature_ids(result)
            if not failing:
                break
            rounds = BASE_ROUNDS + 2 * i
            temp = round(BASE_TEMP + 0.2 * i, 2)
            print(f"\n  ── recovery loop {i + 1}/{self.max_loops} · targeting {sorted(failing)} "
                  f"· dafny rounds={rounds} · code temp={temp} ──")

            # (a) re-prove (escalated rounds); newly-proved features ship from Dafny (adapter mode).
            newly_proved = self._reprove(sorted(failing), rounds)
            if newly_proved:
                repaired_by_proof |= newly_proved
                print(f"     re-proved: {sorted(newly_proved)} → shipped as Dafny-core adapter(s)")

            # (b) regenerate the still-failing features WITH failing test cases (clean-room break).
            still = failing - newly_proved
            if still:
                new_files = self._regenerate(still, result.failures, temp)
                self._swap_files(new_files)
                print(f"     regenerated {len(new_files)} file(s) for {sorted(still)} with test feedback"
                      f" ({self.repair_driver})")

            self._rebuild_app()

            # (c) re-certify (n=1, pass@1) against the frozen test suite.
            result = self._certify()
            now_failing = failing_feature_ids(result)
            repaired_by_tests |= (still - now_failing)
            # Full metric snapshot for THIS iteration (not just pass@1) — so the per-iteration
            # progression of every headline metric is recorded, printed, and laddered in the ledger.
            snap = self._snapshot(result)
            loops.append({
                "loop": i + 1, "dafny_rounds": rounds, "code_temp": temp,
                "targeted": sorted(failing), "reproved": sorted(newly_proved),
                "still_failing": sorted(now_failing),
                "pass_at_1": snap["pass_at"].get("pass@1", 0.0),
                **snap,
            })
            print(f"     iter {i + 1} metrics: PassVer@1={snap['passver_at_1']}  "
                  f"pass@1={snap['pass_at'].get('pass@1', 0.0)}  "
                  f"case_pass_rate={snap['case_pass_rate']}  "
                  f"proved_frs={snap['proved_frs']}/{snap['total_frs']}  "
                  f"certified_frs={snap['certified_frs']}/{snap['total_frs']}")
            # Flush THIS iteration to disk now, so stopping early keeps every completed iteration.
            if self.iter_log is not None:
                self.iter_log.parent.mkdir(parents=True, exist_ok=True)
                with self.iter_log.open("a") as fh:
                    fh.write(json.dumps(loops[-1]) + "\n")
                    fh.flush()
            if not now_failing:
                break

        final_failing = failing_feature_ids(result)
        return {
            "result": result,
            "loops": loops,
            "repair_driver": self.repair_driver,
            "regen_metrics": self.regen_metrics,
            "proved_feature_ids": sorted(self.proved_feature_ids),
            "labels": self._labels(first_failing, repaired_by_proof, repaired_by_tests, final_failing),
            "repaired_by_proof": sorted(repaired_by_proof),
            "repaired_by_tests": sorted(repaired_by_tests),
            "uncertified": sorted(final_failing),
        }

    # --- per-iteration metrics snapshot ---------------------------------------
    def _snapshot(self, result: CertificationResult) -> dict:
        """All headline metrics for the pipeline state AFTER one recovery iteration, at FR
        granularity (the atomic unit): pass@k, case-pass-rate, count of FRs covered by a proof,
        and PassVer@1 (FRs that are proved OR pass pass@1, deduped) over the full FR universe."""
        fr_by_feature: dict[str, list[str]] = {}
        all_fr: set[str] = set()
        for f in self.ir.get("features", []):
            ids = [str(r.get("id")) for r in f.get("functional_requirements", [])]
            fr_by_feature[str(f.get("id"))] = ids
            all_fr.update(ids)
        total = len(all_fr) or 1
        proved_fr = {fr for fid in self.proved_feature_ids for fr in fr_by_feature.get(fid, [])}
        pass1_fr = {fr.fr_id for fr in result.frs if fr.pass_at.get("pass@1", 0.0) >= 1.0}
        certified_fr = proved_fr | pass1_fr
        return {
            "pass_at": {k: round(v, 3) for k, v in result.aggregate_pass_at.items()},
            "case_pass_rate": round(result.aggregate_case_pass_rate, 3),
            "proved_frs": len(proved_fr),
            "certified_frs": len(certified_fr),
            "total_frs": total,
            "passver_at_1": round(len(certified_fr) / total, 3),
        }

    # --- (b) regenerate with test feedback -------------------------------------
    def _regenerate(self, feature_ids: set[str], failures: list[dict], temp: float) -> list:
        """Re-emit the failing features' code WITH their failing cases (the clean-room break).

        Both drivers return the same thing — a list of GeneratedFile whose identity fields come
        from the planner's contract — so ``_swap_files`` is unchanged.
        """
        if self.repair_driver == "deepagent":
            from src.cleanroom.agents.deep.recovery import deep_regenerate_with_test_feedback

            files, metrics = deep_regenerate_with_test_feedback(
                self.ir, feature_ids, failures, temperature=temp)
            self.regen_metrics.append(metrics)
            return files
        regen = CodeAgent(llm=get_llm(temperature=temp), stack=self.stack,
                          prompt_strategy=self.prompt_strategy)
        return regen.regenerate_with_test_feedback(self.ir, feature_ids, failures)

    # --- (a) re-prove ----------------------------------------------------------
    def _reprove(self, feature_ids: list[str], rounds: int) -> set[str]:
        """Re-prove failing features with escalated rounds; ship newly-proved ones from Dafny."""
        if not (self.proj and self.adapter_mode):
            return set()  # shipping a proved core is a FastAPI/adapter-mode concept only.
        agent = DafnyAgent(self.proj, model=self.dafny_model, max_rounds=rounds,
                           prompt_strategy=self.prompt_strategy)
        proved_now: list = []
        for fid in feature_ids:
            feat = agent.generate_feature(self.ir, fid)
            if feat.verified:
                self.proved_feature_ids.add(fid)
                self.proved_modules[fid] = feat.module
                self.proved_sources[fid] = feat.dafny_source
                proved_now.append(feat)
        if not proved_now:
            return set()
        compile_dafny(self.proj, GeneratedDafny(features=proved_now), target=self.prove_target)
        for feat in proved_now:
            self._ship_as_adapter(feat.feature_id, feat.module, feat.dafny_source)
        return {f.feature_id for f in proved_now}

    def _ship_as_adapter(self, fid: str, module: str, source: str) -> None:
        """Replace a feature's full-code files with a thin adapter over its proved Dafny core."""
        files = self.ir["generated_code"]["files"]
        files[:] = [f for f in files if f.get("feature_id") != fid]
        adapter = self._adapter_agent.generate_adapter(self.ir, fid, module, source)
        files.append(adapter.model_dump())

    # --- (b) swap regenerated code --------------------------------------------
    def _swap_files(self, new_files: list) -> None:
        files = self.ir["generated_code"]["files"]
        new_by_fr = {f.fr_id: f for f in new_files}
        seen: set[str] = set()
        for i, f in enumerate(files):
            if f.get("fr_id") in new_by_fr:
                files[i] = new_by_fr[f["fr_id"]].model_dump()
                seen.add(f["fr_id"])
        for fr_id, nf in new_by_fr.items():
            if fr_id not in seen:
                files.append(nf.model_dump())

    def _rebuild_app(self) -> None:
        """Reassemble the on-disk runnable app so final artifacts reflect the repairs."""
        if self.stack != "fastapi":
            CodeAgent.write_files(GeneratedCode(**self.ir["generated_code"]), self.code_dir)
            return
        app_dir = build_runnable_package(self.ir["generated_code"], self.code_dir, ir=self.ir)
        if self.adapter_mode and self.proj and self.proved_feature_ids:
            stage_dafny_cores(Path(app_dir), self.proj, self._proved_module_list())

    # --- (c) re-certify --------------------------------------------------------
    def _certify(self) -> CertificationResult:
        agent = CertificationAgent(
            code_llm=get_llm(temperature=0.0),
            n=1,                       # pass@1 on the repaired pipeline sample
            stack=self.stack,
            prompt_strategy=self.prompt_strategy,
            skip_feature_ids=self.proved_feature_ids if self.adapter_mode else set(),
            dafny_proj=self.proj if self.adapter_mode else None,
            dafny_modules=self._proved_module_list() if self.adapter_mode else None,
        )
        return agent.certify(self.ir)

    # --- labels & helpers ------------------------------------------------------
    def _proved_module_list(self) -> list[str]:
        return [self.proved_modules[fid] for fid in sorted(self.proved_feature_ids)
                if fid in self.proved_modules]

    def _labels(self, first_failing, by_proof, by_tests, final_failing) -> dict[str, str]:
        labels: dict[str, str] = {}
        for f in self.ir.get("features", []):
            fid = str(f.get("id"))
            if fid in self.proved_feature_ids:
                labels[fid] = "PROVED" + (" (recovery)" if fid in by_proof else "")
            elif fid in final_failing:
                labels[fid] = "UNCERTIFIED"
            elif fid in by_tests:
                labels[fid] = "TESTED (repaired-with-tests)"
            elif fid in first_failing:
                labels[fid] = "TESTED (repaired-with-tests)"
            else:
                labels[fid] = "TESTED (clean-room)"
        return labels
