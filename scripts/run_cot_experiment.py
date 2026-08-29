#!/usr/bin/env python3
"""Autonomous CoT experiment driver — DeepSeek only, all SRS x 3 languages.

Runs the full pipeline with --prompt-strategy cot and model deepseek/deepseek-v3.2 for every
(SRS x language) cell, with:
  * per-cell TIMEOUT (kills a hung cell instead of stalling the whole batch),
  * IDEMPOTENT resume (a cell with a complete run record is skipped),
  * multi-ROUND RETRY (failed/incomplete cells are retried up to MAX_ROUNDS),
  * the 3 languages of one SRS run CONCURRENTLY; SRS groups run sequentially.

After every round it (re)writes:
  * cot_deepseek_results.csv  — the requested columns, one row per (srs, language):
        srs, language, verification_pass_ratio, test_case_pass_ratio, avg_input_token,
        avg_output_token, total_time, avg_verification_iter, avg_test_iter, PassVer@1, total_tokens
  * <srs_stem>_results_cot_deepseek.csv — per-SRS, full experiment_metrics.csv column set.

Safe to re-run: completed cells are skipped, so it converges.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.cleanroom.utils.llm_client import DEFAULT_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRS_DIR = ROOT / "data" / "srs"
BASE = ROOT / "outputs" / "cot"
# Model under test. Defaults to whatever .env configures (LLM_MODEL), so this driver
# follows the served model instead of the retired OpenRouter slug it was written against.
MODEL = os.getenv("EXPERIMENT_MODEL") or DEFAULT_MODEL
FLAGS = ["--prove", "--certify", "--max-cert-loops", "4", "--prove-rounds", "4"]
LANGS = [("python", "fastapi"), ("java", "spring"), ("javascript", "express")]
CELL_TIMEOUT = int(os.getenv("COT_CELL_TIMEOUT", "3900"))   # seconds per cell (65 min)
MAX_ROUNDS = int(os.getenv("COT_MAX_ROUNDS", "4"))
COMBINED_CSV = ROOT / "cot_deepseek_results.csv"
COMBINED_JSON = ROOT / "cot_deepseek_results.json"

COMBINED_COLUMNS = [
    "srs", "language", "verification_pass_ratio", "test_case_pass_ratio",
    "avg_input_token", "avg_output_token", "total_time",
    "avg_verification_iter", "avg_test_iter", "PassVer@1", "total_tokens",
]
# metrics.summary key -> combined CSV column
SUMMARY_MAP = {
    "verification_pass_ratio": "verification_pass_ratio",
    "test_case_pass_ratio": "test_case_pass_ratio",
    "avg_input_token": "avg_input_token",
    "avg_output_token": "avg_output_token",
    "total_time": "total_time",
    "avg_verification_iteration": "avg_verification_iter",
    "avg_test_iteration": "avg_test_iter",
    "PassVer@1": "PassVer@1",
}


def cell_dir(stem: str, lang: str) -> Path:
    return BASE / stem / lang


def is_complete(stem: str, lang: str) -> bool:
    d = cell_dir(stem, lang) / "runs"
    for j in d.glob("*.json"):
        try:
            if json.loads(j.read_text(encoding="utf-8")).get("status") == "complete":
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def launch(srs: Path, lang: str, stack: str) -> subprocess.Popen:
    out = cell_dir(srs.stem, lang)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PIPELINE_LEDGER_FILE"] = str(out / "ledger.md")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv", "run", "python", "run_pipeline.py", str(srs),
        "--prompt-strategy", "cot", "--model", MODEL,
        "--language", lang, "--stack", stack,
        *FLAGS, "--output-dir", str(out),
    ]
    log = open(out / "run.log", "w")
    log.write(f"$ {' '.join(cmd)}\n\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    p._cot_log = log  # type: ignore[attr-defined]
    p._cot_start = time.time()  # type: ignore[attr-defined]
    p._cot_name = f"{srs.stem}/{lang}"  # type: ignore[attr-defined]
    return p


def run_group(srs: Path) -> None:
    """Run the 3 languages of one SRS concurrently, each with its own timeout."""
    procs: list[subprocess.Popen] = []
    for lang, stack in LANGS:
        if is_complete(srs.stem, lang):
            print(f"      skip    {srs.stem}/{lang}  (already complete)", flush=True)
            continue
        print(f"      start   {srs.stem}/{lang}", flush=True)
        procs.append(launch(srs, lang, stack))
    while procs:
        time.sleep(10)
        still: list[subprocess.Popen] = []
        for p in procs:
            if p.poll() is not None:
                p._cot_log.close()  # type: ignore[attr-defined]
                rc = p.returncode
                print(f"      {'done ' if rc == 0 else 'EXIT'+str(rc)}   {p._cot_name}", flush=True)
            elif time.time() - p._cot_start > CELL_TIMEOUT:  # type: ignore[attr-defined]
                p.kill()
                p._cot_log.write(f"\n\n!! KILLED after {CELL_TIMEOUT}s timeout\n")  # type: ignore[attr-defined]
                p._cot_log.close()  # type: ignore[attr-defined]
                print(f"      TIMEOUT {p._cot_name}  (killed after {CELL_TIMEOUT}s)", flush=True)
            else:
                still.append(p)
        procs = still


def _build_rows() -> list[dict]:
    rows: list[dict] = []
    for srs in sorted(SRS_DIR.glob("*.xml")):
        for lang, _stack in LANGS:
            d = cell_dir(srs.stem, lang)
            row = {c: "" for c in COMBINED_COLUMNS}
            row["srs"], row["language"] = srs.stem, lang
            fir = next(iter(d.glob("*full_ir.json")), None)
            if fir is not None:
                try:
                    m = (json.loads(fir.read_text(encoding="utf-8")).get("metrics") or {})
                except (json.JSONDecodeError, OSError):
                    m = {}
                summary = m.get("summary") or {}
                for src, col in SUMMARY_MAP.items():
                    row[col] = summary.get(src, "")
                row["total_tokens"] = (m.get("tokens") or {}).get("total", "")
            row["status"] = "complete" if is_complete(srs.stem, lang) else "incomplete"
            rows.append(row)
    return rows


def emit_results() -> None:
    """Write BOTH outputs (refreshed after every SRS): the requested-column CSV and a JSON mirror."""
    rows = _build_rows()
    with COMBINED_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COMBINED_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    payload = {
        "model": MODEL, "strategy": "cot", "flags": " ".join(FLAGS),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "columns": COMBINED_COLUMNS,
        "cells": rows,
    }
    COMBINED_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    done = sum(1 for r in rows if r.get("status") == "complete")
    print(f"  wrote {COMBINED_CSV.name} + {COMBINED_JSON.name}  ({done}/{len(rows)} cells complete)", flush=True)


def emit_per_srs_csv(srs: Path) -> None:
    csv_out = ROOT / f"{srs.stem}_results_cot_deepseek.csv"
    subprocess.run(
        ["uv", "run", "python", "scripts/collect_metrics.py",
         "--root", str(BASE / srs.stem), "--out", str(csv_out)],
        cwd=str(ROOT), check=False,
    )


def main() -> None:
    srs_files = sorted(SRS_DIR.glob("*.xml"))
    print(f"=== CoT experiment START {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"model={MODEL} strategy=cot flags={' '.join(FLAGS)}", flush=True)
    print(f"cells={len(srs_files) * len(LANGS)}  cell_timeout={CELL_TIMEOUT}s  max_rounds={MAX_ROUNDS}", flush=True)

    for rnd in range(1, MAX_ROUNDS + 1):
        incomplete = [(s, lang) for s in srs_files for lang, _ in LANGS if not is_complete(s.stem, lang)]
        if not incomplete:
            print(f"\n=== all cells complete before round {rnd} ===", flush=True)
            break
        print(f"\n########## ROUND {rnd}/{MAX_ROUNDS}  ({len(incomplete)} cell(s) to run) ##########", flush=True)
        for srs in srs_files:
            if all(is_complete(srs.stem, lang) for lang, _ in LANGS):
                continue
            print(f">>> [{time.strftime('%H:%M:%S')}] SRS: {srs.stem}", flush=True)
            run_group(srs)
            emit_per_srs_csv(srs)
            emit_results()   # refresh after every SRS so output is always current

    emit_results()
    # Final status
    final_incomplete = [(s.stem, lang) for s in srs_files for lang, _ in LANGS if not is_complete(s.stem, lang)]
    n_total = len(srs_files) * len(LANGS)
    print(f"\n=== CoT experiment DONE {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"complete: {n_total - len(final_incomplete)}/{n_total}", flush=True)
    if final_incomplete:
        print(f"STILL INCOMPLETE after {MAX_ROUNDS} rounds: {final_incomplete}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
