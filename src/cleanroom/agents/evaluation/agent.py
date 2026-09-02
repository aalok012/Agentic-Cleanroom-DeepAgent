"""Evaluation Agent — pass@k with executable oracle (EvalPlus-style rigor).

The ONE stage where spec-derived CODE and spec-derived TESTS are brought together,
and only AFTER both are frozen. It feeds nothing back into generation; it only measures.

Metric: pass@k (Chen et al., 2021). Scoring unit is each FR (functional requirement),
macro-averaged across FRs — analogous to HumanEval tasks.

Oracle: an isolated subprocess executes each structured test case (inputs_json /
expected_json) against the candidate code. No LLM judge unless execution cannot run.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from statistics import mean

from src.cleanroom.agents.code.agent import CodeAgent
from src.cleanroom.agents.code.schema.code import GeneratedCode, GeneratedFile
from src.cleanroom.agents.evaluation.metric import pass_at_k
from src.cleanroom.targets import get_target
from src.cleanroom.agents.planning.agent import PlanningAgent
from src.cleanroom.agents.evaluation.schema.certification import (
    CertificationResult,
    FeatureCertification,
    FRCertification,
)
from src.cleanroom.agents.test.agent import TestAgent
from src.cleanroom.utils.ir import feature_id_of, normalize_generated_tests, normalize_ir_features, normalize_test_case, requirement_text
from src.cleanroom.utils.llm_client import get_llm


class CertificationAgent:
    def __init__(
        self,
        code_llm=None,
        n: int = 3,
        k_values: tuple[int, ...] | None = None,
        temperature: float = 0.4,
        stack: str = "python",
        language: str = "python",
        use_pipeline_sample: bool = True,
        case_timeout: float = 10.0,
        skip_feature_ids: set[str] | None = None,
        dafny_proj: "Path | None" = None,
        dafny_modules: list[str] | None = None,
    ) -> None:
        self.n = max(1, n)
        # Features already PROVED by the Dafny tier — certified by proof, so excluded from pass@k.
        self.skip_feature_ids = set(skip_feature_ids or ())
        # When proved features ship as Dafny-core adapters, every sample app the oracle boots needs
        # the compiled cores + shim staged or the adapters' `import dafny_marshal` fails at boot.
        self.dafny_proj = Path(dafny_proj) if dafny_proj else None
        self.dafny_modules = list(dafny_modules or [])
        # pass@1 only by default (don't generate extra samples just to compute pass@3);
        # callers can still pass k_values explicitly if they want a multi-sample estimate.
        self.k_values = list(k_values or [1])
        self.k_values = [k for k in self.k_values if k <= self.n] or [1]
        self.code_llm = code_llm if code_llm is not None else get_llm(temperature=temperature)
        self.stack = stack
        self.language = language
        self.target = get_target(language, stack)
        self.use_pipeline_sample = use_pipeline_sample
        self.case_timeout = case_timeout

    def certify(self, ir: dict) -> CertificationResult:
        if not (ir.get("planning") or {}).get("contracts"):
            raise ValueError("CertificationAgent requires an IR with 'planning.contracts'.")

        normalize_ir_features(ir)
        PlanningAgent.normalize_ir_planning(ir)
        tests = ir.get("generated_tests") or TestAgent(
            stack=self.stack, language=self.language
        ).generate(ir).model_dump()
        normalize_generated_tests(tests)
        cases_by_fr = self._cases_by_fr(tests)
        plans = {c["fr_id"]: c for c in ir["planning"]["contracts"]}
        feature_names = {feature_id_of(f): f.get("name", "") for f in ir.get("features", [])}

        # Drop FRs whose feature was already PROVED by the Dafny tier — they're certified by
        # proof and need no statistical testing (the prove-or-test fallback).
        if self.skip_feature_ids:
            cases_by_fr = {
                fr_id: cases for fr_id, cases in cases_by_fr.items()
                if plans.get(fr_id, {}).get("feature_id") not in self.skip_feature_ids
            }

        samples = self._collect_samples(ir)
        req_text = self._requirement_text_map(ir)

        diagnostics: list[dict] = []
        fr_results: list[FRCertification] = []

        for fr_id, cases in sorted(cases_by_fr.items(), key=lambda x: x[0]):
            plan = plans.get(fr_id, {})
            passing_samples = 0
            cases_passed = 0

            for sample_idx, sample in enumerate(samples):
                sample_pass = True
                with tempfile.TemporaryDirectory(prefix="cleanroom_cert_") as tmp:
                    code_dir = Path(tmp) / "code"
                    self._write_sample(sample, code_dir, tests)
                    for case in cases:
                        ok, reason = self.target.run_case(
                            code_dir, plan, case, self.stack, self.case_timeout
                        )
                        diagnostics.append({
                            "fr_id": fr_id,
                            "feature_id": plan.get("feature_id", ""),
                            "feature_name": feature_names.get(plan.get("feature_id"), ""),
                            "requirement_id": case.get("requirement_id", fr_id),
                            "sample": sample_idx,
                            "description": case.get("description", ""),
                            "inputs": case.get("inputs", case.get("inputs_json", "")),
                            "expected": case.get("expected", case.get("expected_json", "")),
                            "requirement_text": req_text.get(fr_id, ""),
                            "verdict": "pass" if ok else "fail",
                            "reason": reason,
                        })
                        if ok:
                            cases_passed += 1
                        else:
                            sample_pass = False
                if sample_pass and cases:
                    passing_samples += 1

            cases_total = len(cases) * len(samples)
            pass_at = {f"pass@{k}": pass_at_k(len(samples), passing_samples, k) for k in self.k_values}
            fr_results.append(
                FRCertification(
                    fr_id=fr_id,
                    feature_id=plan.get("feature_id", ""),
                    n_samples=len(samples),
                    n_test_cases=len(cases),
                    passing_samples=passing_samples,
                    pass_at=pass_at,
                    cases_passed=cases_passed,
                    cases_total=cases_total,
                    case_pass_rate=(cases_passed / cases_total if cases_total else 0.0),
                )
            )

        aggregate_pass_at = {
            f"pass@{k}": (mean(fr.pass_at.get(f"pass@{k}", 0.0) for fr in fr_results) if fr_results else 0.0)
            for k in self.k_values
        }
        total_cases = sum(fr.cases_total for fr in fr_results)
        total_passed = sum(fr.cases_passed for fr in fr_results)
        aggregate_case_rate = total_passed / total_cases if total_cases else 0.0

        features_out = self._aggregate_features(fr_results, feature_names)
        model = getattr(self.code_llm, "model_name", getattr(self.code_llm, "model", ""))
        result = CertificationResult(
            model=str(model),
            oracle=self.target.oracle_name(self.stack),
            n=len(samples),
            k_values=self.k_values,
            frs=fr_results,
            features=features_out,
            aggregate_pass_at=aggregate_pass_at,
            aggregate_case_pass_rate=aggregate_case_rate,
            failures=[d for d in diagnostics if d["verdict"] != "pass"],
        )
        self._print_diagnostics(ir, diagnostics, result)
        return result

    def run(self, ir: dict, output_dir: Path = Path("outputs/generated")) -> CertificationResult:
        result = self.certify(ir)
        self.write_report(result, output_dir / ir.get("project_name", "project"))
        return result

    def _collect_samples(self, ir: dict) -> list[GeneratedCode]:
        code_agent = CodeAgent(llm=self.code_llm, stack=self.stack, language=self.language)
        samples: list[GeneratedCode] = []

        if self.use_pipeline_sample and ir.get("generated_code"): 
            samples.append(GeneratedCode(**ir["generated_code"]))

        need = self.n - len(samples)
        if need <= 0:
            return samples[: self.n]

        for _ in range(need):
            samples.append(code_agent.generate(ir))
        return samples[: self.n]

    def _write_sample(self, sample: GeneratedCode, code_dir: Path, tests: dict | None = None) -> None:
        """Lay a sample out for its oracle via the language target (flat tree / FastAPI app /
        Java sources). For a fastapi sample with Dafny-core adapters, also stage the compiled
        cores + the marshalling shim so the app boots (otherwise `import dafny_marshal` fails).
        Targets that statically compile generated tests can lay those sources out too."""
        app_dir = self.target.package_sample(sample.model_dump(), code_dir, self.stack)
        if app_dir is not None and self.dafny_proj and self.dafny_modules:
            # Route through the target so the FastAPI (compiled -py + shim) and Spring (translated
            # Java cores into src/main/java) staging each use their own layout.
            self.target.stage_cores(Path(app_dir), self.dafny_proj, self.dafny_modules)
        self.target.package_tests(tests or {}, Path(app_dir) if app_dir is not None else code_dir,
                                  self.stack)

    @staticmethod
    def _cases_by_fr(tests: dict) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for feature in tests.get("features", []):
            for case in feature.get("cases", []):
                case = normalize_test_case(case)
                rid = case.get("requirement_id")
                if rid:
                    out.setdefault(rid, []).append(case)
        return out

    @staticmethod
    def _aggregate_features(fr_results: list[FRCertification], feature_names: dict) -> list[FeatureCertification]:
        by_feature: dict[str, list[FRCertification]] = {}
        for fr in fr_results:
            by_feature.setdefault(fr.feature_id, []).append(fr)

        features: list[FeatureCertification] = []
        for fid, frs in sorted(by_feature.items()):
            n_cases = sum(fr.n_test_cases for fr in frs)
            cases_passed = sum(fr.cases_passed for fr in frs)
            cases_total = sum(fr.cases_total for fr in frs)
            n_samples = frs[0].n_samples if frs else 0
            passing_samples = min(fr.passing_samples for fr in frs) if frs else 0
            k_keys = frs[0].pass_at.keys() if frs else []
            pass_at = {
                key: mean(fr.pass_at.get(key, 0.0) for fr in frs) if frs else 0.0
                for key in k_keys
            }
            features.append(
                FeatureCertification(
                    feature_id=fid,
                    name=feature_names.get(fid, ""),
                    n_samples=n_samples,
                    n_test_cases=n_cases,
                    passing_samples=passing_samples,
                    pass_at=pass_at,
                    cases_passed=cases_passed,
                    cases_total=cases_total,
                    case_pass_rate=(cases_passed / cases_total if cases_total else 0.0),
                )
            )
        return features

    @staticmethod
    def _requirement_text_map(ir: dict) -> dict[str, str]:
        out: dict[str, str] = {}
        for feature in ir.get("features", []):
            for r in feature.get("functional_requirements", []):
                out[r["id"]] = requirement_text(r)
        return out

    @staticmethod
    def _trim(text: str, limit: int = 160) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _print_diagnostics(self, ir: dict, diagnostics: list[dict], result: CertificationResult) -> None:
        print("\n" + "=" * 70)
        print("  FUNCTIONAL REQUIREMENTS  (spec coverage)")
        print("=" * 70)
        for feature in ir.get("features", []):
            fid = feature_id_of(feature) or "?"
            print(f"\n[{fid}] {feature.get('name', '')}")
            for r in feature.get("functional_requirements", []):
                print(f"    FR  [{r['id']}] {self._trim(requirement_text(r))}")

        print("\n" + "=" * 70)
        print("  TEST RESULTS  (executable oracle per spec-derived test case)")
        print("=" * 70)
        for d in diagnostics:
            mark = "PASS" if d["verdict"] == "pass" else "FAIL"
            print(f"  [{mark}] {d['fr_id']:<12} sample={d['sample']}  {self._trim(d['description'], 60)}")

        fails = [d for d in diagnostics if d["verdict"] != "pass"]
        if fails:
            print("\n" + "=" * 70)
            print("  === FAILURES ===")
            print("=" * 70)
            for d in fails[:25]:
                print(f"\n[FAIL] {d['fr_id']}  sample={d['sample']}")
                print(f"  test:     {self._trim(d['description'], 200)}")
                print(f"  inputs:   {self._trim(str(d['inputs']), 200)}")
                print(f"  expected: {self._trim(str(d['expected']), 200)}")
                print(f"  reason:   {self._trim(d['reason'], 200)}")
            if len(fails) > 25:
                print(f"\n  … and {len(fails) - 25} more failures")

        total = len(diagnostics)
        n_pass = sum(1 for d in diagnostics if d["verdict"] == "pass")
        k = self.k_values[-1]
        score = result.aggregate_pass_at.get(f"pass@{k}", 0.0)
        print("\n" + "=" * 70)
        print(f"  SUMMARY: {n_pass} passed / {total - n_pass} failed / {total} total case executions")
        print(f"  Case pass rate: {result.aggregate_case_pass_rate:.3f}")
        print(f"  Aggregate pass@{k} (macro-avg over FRs): {score:.3f}")
        print(f"  Oracle: {result.oracle}")
        print("=" * 70 + "\n")

    @staticmethod
    def write_report(result: CertificationResult, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ks = result.k_values
        lines = [
            "# Certification report — pass@k (executable oracle)",
            "",
            f"_Model: {result.model or 'n/a'} · samples n={result.n} · "
            f"FRs certified: {len(result.frs)} · oracle: {result.oracle}_",
            "",
            "## Aggregate (macro-average over FRs)",
        ]
        lines += [f"- **pass@{k}**: {result.aggregate_pass_at.get(f'pass@{k}', 0.0):.3f}" for k in ks]
        lines += [
            f"- **case pass rate**: {result.aggregate_case_pass_rate:.3f}",
            "",
            "## Per FR (HumanEval-style task unit)",
            "",
            "| fr_id | feature | cases | c/n | case rate | " + " | ".join(f"pass@{k}" for k in ks) + " |",
            "|---|---|---|---|---|" + "---|" * len(ks),
        ]
        for fr in result.frs:
            cells = " | ".join(f"{fr.pass_at.get(f'pass@{k}', 0.0):.3f}" for k in ks)
            lines.append(
                f"| {fr.fr_id} | {fr.feature_id} | {fr.n_test_cases} | "
                f"{fr.passing_samples}/{fr.n_samples} | {fr.case_pass_rate:.3f} | {cells} |"
            )
        lines += [
            "",
            "## Per feature (rolled up)",
            "",
            "| feature | name | cases | case rate | " + " | ".join(f"pass@{k}" for k in ks) + " |",
            "|---|---|---|---|" + "---|" * len(ks),
        ]
        for f in result.features:
            cells = " | ".join(f"{f.pass_at.get(f'pass@{k}', 0.0):.3f}" for k in ks)
            lines.append(
                f"| {f.feature_id} | {f.name} | {f.n_test_cases} | "
                f"{f.case_pass_rate:.3f} ({f.cases_passed}/{f.cases_total}) | {cells} |"
            )
        dest = output_dir / "certification_report.md"
        dest.write_text("\n".join(lines) + "\n")
        return dest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.cleanroom.agents.evaluation.agent <full_ir.json>")
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as fh:
        enriched_ir = json.load(fh)

    agent = CertificationAgent()
    result = agent.run(enriched_ir)
    for fr in result.frs:
        ks = "  ".join(f"pass@{k}={fr.pass_at[f'pass@{k}']:.3f}" for k in result.k_values)
        print(f"  {fr.fr_id}  c/n={fr.passing_samples}/{fr.n_samples}  {ks}")
    print("\nAggregate: " + "  ".join(
        f"pass@{k}={result.aggregate_pass_at[f'pass@{k}']:.3f}" for k in result.k_values
    ))
