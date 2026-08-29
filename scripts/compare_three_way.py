#!/usr/bin/env python3
"""Three-way A/B/C: pipeline(baseline) vs CoT vs MoT DeepSeek runs (same pipeline, three prompt
strategies). Extends scripts/compare_cot_vs_pipeline.py from 2 arms to 3.

Sources, keyed by (srs, language):
  * pipeline (baseline prompts) -> experiment_metrics.csv   (deepseek rows only)
  * cot                         -> cot_deepseek_results.csv
  * mot                         -> mot_deepseek_results.csv  IF present, else assembled directly
                                   from outputs/mot/<srs>/<lang>/*full_ir.json (no emit step needed)

Emits three_way_deepseek.csv. For each metric it writes the three arm values plus three deltas:
    pipeline_X, cot_X, mot_X,
    d_cot_X     = cot - pipeline   (identical to the old delta_X in cot_vs_pipeline_deepseek.csv),
    d_mot_X     = mot - cot        (the incremental effect of MoT decomposition over CoT),
    d_total_X   = mot - pipeline   (total gain of MoT over the baseline pipeline).
Columns are grouped per metric, matching the spreadsheet's pipeline_/cot_/delta_ grouping.

Also prints an aggregate summary (per-arm means + a MoT-vs-CoT win/tie/loss tally on the two
headline metrics) and writes three_way_summary_deepseek.csv. Run from anywhere:
    python scripts/compare_three_way.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cleanroom.utils.llm_client import DEFAULT_MODEL  # noqa: E402
BASELINE_CSV = ROOT / "experiment_metrics.csv"
COT_CSV = ROOT / "cot_deepseek_results.csv"
MOT_CSV = ROOT / "mot_deepseek_results.csv"
MOT_OUTPUTS = ROOT / "outputs" / "mot"
SRS_DIR = ROOT / "data" / "srs"
LANGS = ["python", "java", "javascript"]

# Which rows of the baseline ledger count as the "pipeline" arm. Defaults to the model
# under test; override with --model-filter (use "deepseek" for the archived paper runs).
MODEL_FILTER = os.getenv("EXPERIMENT_MODEL") or DEFAULT_MODEL

OUT = ROOT / "three_way_deepseek.csv"
OUT_SUMMARY = ROOT / "three_way_summary_deepseek.csv"

# metric -> (baseline column, cot/mot column). Only total_tokens differs by source.
METRICS = {
    "verification_pass_ratio": ("verification_pass_ratio", "verification_pass_ratio"),
    "test_case_pass_ratio": ("test_case_pass_ratio", "test_case_pass_ratio"),
    "avg_input_token": ("avg_input_token", "avg_input_token"),
    "avg_output_token": ("avg_output_token", "avg_output_token"),
    "total_time": ("total_time", "total_time"),
    "avg_verification_iter": ("avg_verification_iter", "avg_verification_iter"),
    "avg_test_iter": ("avg_test_iter", "avg_test_iter"),
    "PassVer@1": ("PassVer@1", "PassVer@1"),
    "total_tokens": ("tokens_total", "total_tokens"),
}
# metrics where the summary should report a mean and a MoT-vs-CoT comparison
HEADLINE = ("verification_pass_ratio", "PassVer@1")
# maps full_ir metrics.summary keys -> our result column names (mirrors emit_cot_results.py)
_SUMMARY_MAP = {
    "verification_pass_ratio": "verification_pass_ratio",
    "test_case_pass_ratio": "test_case_pass_ratio",
    "avg_input_token": "avg_input_token",
    "avg_output_token": "avg_output_token",
    "total_time": "total_time",
    "avg_verification_iteration": "avg_verification_iter",
    "avg_test_iteration": "avg_test_iter",
    "PassVer@1": "PassVer@1",
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_csv(path: Path, model_filter: str | None) -> dict[tuple[str, str], dict]:
    """Rows keyed by (srs, language). `model_filter` keeps only rows whose `model`
    column contains that substring — the baseline ledger holds every model ever run,
    so the pipeline arm must be narrowed to the one under test."""
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for r in csv.DictReader(f):
            if model_filter and model_filter.lower() not in (r.get("model", "").lower()):
                continue
            out[(r.get("srs", ""), r.get("language", ""))] = r
    return out


def _load_mot_from_outputs() -> dict[tuple[str, str], dict]:
    """Fallback: build the mot rows straight from outputs/mot/<srs>/<lang>/*full_ir.json."""
    out: dict[tuple[str, str], dict] = {}
    if not MOT_OUTPUTS.exists():
        return out
    srs_stems = ([p.stem for p in sorted(SRS_DIR.glob("*.xml"))]
                 if SRS_DIR.exists() else [p.name for p in sorted(MOT_OUTPUTS.iterdir()) if p.is_dir()])
    for stem in srs_stems:
        for lang in LANGS:
            fir = next(iter((MOT_OUTPUTS / stem / lang).glob("*full_ir.json")), None)
            if fir is None:
                continue
            try:
                m = (json.loads(fir.read_text(encoding="utf-8")).get("metrics") or {})
            except (json.JSONDecodeError, OSError):
                continue
            summary = m.get("summary") or {}
            row = {col: summary.get(src, "") for src, col in _SUMMARY_MAP.items()}
            row["total_tokens"] = (m.get("tokens") or {}).get("total", "")
            out[(stem, lang)] = row
    return out


def _load_mot() -> tuple[dict[tuple[str, str], dict], str]:
    rows = _load_csv(MOT_CSV, model_filter=None)
    if rows:
        return rows, f"{MOT_CSV.name}"
    rows = _load_mot_from_outputs()
    if rows:
        return rows, f"{MOT_OUTPUTS.relative_to(ROOT)}/ (assembled directly; no emit step)"
    return {}, "(no MoT data found — run ./run_mot_matrix.sh first)"


def _mean(values: list) -> float | None:
    nums = [n for n in (_num(v) for v in values) if n is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def main() -> None:
    global MODEL_FILTER
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-filter", default=MODEL_FILTER,
        help="substring matched against the baseline ledger's `model` column "
             "(default: the model under test; pass 'deepseek' to reproduce the archived runs)")
    MODEL_FILTER = ap.parse_args().model_filter

    base = _load_csv(BASELINE_CSV, model_filter=MODEL_FILTER)
    cot = _load_csv(COT_CSV, model_filter=None)
    mot, mot_src = _load_mot()

    cols = ["srs", "language"]
    for m in METRICS:
        cols += [f"pipeline_{m}", f"cot_{m}", f"mot_{m}",
                 f"d_cot_{m}", f"d_mot_{m}", f"d_total_{m}"]

    # A three-way comparison is only meaningful for cells present in all three arms.
    # Keep incomplete cells out of the emitted CSV so downstream tables do not contain blanks.
    keys = sorted(set(base) & set(cot) & set(mot))
    omitted = sorted((set(base) | set(cot) | set(mot)) - set(keys))
    rows = []
    for srs, lang in keys:
        b, c, mo = base.get((srs, lang), {}), cot.get((srs, lang), {}), mot.get((srs, lang), {})
        row = {"srs": srs, "language": lang}
        for m, (bcol, ccol) in METRICS.items():
            bv, cv, mv = b.get(bcol, ""), c.get(ccol, ""), mo.get(ccol, "")
            row[f"pipeline_{m}"], row[f"cot_{m}"], row[f"mot_{m}"] = bv, cv, mv
            nb, nc, nm = _num(bv), _num(cv), _num(mv)
            row[f"d_cot_{m}"] = round(nc - nb, 4) if (nb is not None and nc is not None) else ""
            row[f"d_mot_{m}"] = round(nm - nc, 4) if (nc is not None and nm is not None) else ""
            row[f"d_total_{m}"] = round(nm - nb, 4) if (nb is not None and nm is not None) else ""
        rows.append(row)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- aggregate summary ----
    arms = {"pipeline": (base, 0), "cot": (cot, 1), "mot": (mot, 1)}
    summary_rows = []
    for m, (bcol, ccol) in METRICS.items():
        rec = {"metric": m}
        for arm, (data, which) in arms.items():
            col = bcol if which == 0 else ccol
            rec[f"mean_{arm}"] = _mean([r.get(col, "") for r in data.values()])
        summary_rows.append(rec)
    with OUT_SUMMARY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean_pipeline", "mean_cot", "mean_mot"])
        w.writeheader()
        w.writerows(summary_rows)

    # ---- console report ----
    print(f"wrote {OUT.name}  ({len(rows)} complete (srs,language) triples)")
    print(f"  pipeline rows: {len(base)} (experiment_metrics.csv, model~{MODEL_FILTER!r})")
    print(f"  cot rows:      {len(cot)} ({COT_CSV.name})")
    print(f"  mot rows:      {len(mot)}  source: {mot_src}")
    if omitted:
        print(f"  omitted rows:  {len(omitted)} missing at least one arm")
    print(f"wrote {OUT_SUMMARY.name}\n")
    print(f"{'metric':24s} {'pipeline':>10s} {'cot':>10s} {'mot':>10s}")
    for rec in summary_rows:
        def fmt(x):
            return f"{x:.4f}" if isinstance(x, float) else "  n/a"
        print(f"{rec['metric']:24s} {fmt(rec['mean_pipeline']):>10s} "
              f"{fmt(rec['mean_cot']):>10s} {fmt(rec['mean_mot']):>10s}")

    # MoT vs CoT win/tie/loss on the headline metrics (cells where both are numeric)
    if mot:
        print("\nMoT vs CoT (per-cell, higher = better):")
        for m in HEADLINE:
            ccol = METRICS[m][1]
            win = tie = loss = 0
            for k in set(cot) & set(mot):
                nc, nm = _num(cot[k].get(ccol)), _num(mot[k].get(ccol))
                if nc is None or nm is None:
                    continue
                win += nm > nc
                tie += nm == nc
                loss += nm < nc
            print(f"  {m:24s} win={win}  tie={tie}  loss={loss}")
    else:
        print("\n(no MoT cells yet — run ./run_mot_matrix.sh, then re-run this script)")


if __name__ == "__main__":
    main()
