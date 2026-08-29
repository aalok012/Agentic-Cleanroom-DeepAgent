#!/usr/bin/env python3
"""Emit the CoT experiment results in BOTH CSV and JSON from whatever cells exist so far.

Standalone + idempotent: reads each outputs/cot/<srs>/<lang>/<stem>_full_ir.json, pulls the
headline metrics.summary + total tokens, and (re)writes:
  * cot_deepseek_results.csv   (requested columns)
  * cot_deepseek_results.json  (same data + run metadata)
Safe to call repeatedly (a watcher loop uses it) and after the run finishes.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cleanroom.utils.llm_client import DEFAULT_MODEL  # noqa: E402
SRS_DIR = ROOT / "data" / "srs"
BASE = ROOT / "outputs" / "cot"
# Model under test — follows .env (LLM_MODEL) rather than a hardcoded slug.
MODEL = os.getenv("EXPERIMENT_MODEL") or DEFAULT_MODEL
FLAGS = "--prove --certify --max-cert-loops 4 --prove-rounds 4"
LANGS = ["python", "java", "javascript"]
COMBINED_CSV = ROOT / "cot_deepseek_results.csv"
COMBINED_JSON = ROOT / "cot_deepseek_results.json"

COLUMNS = [
    "srs", "language", "verification_pass_ratio", "test_case_pass_ratio",
    "avg_input_token", "avg_output_token", "total_time",
    "avg_verification_iter", "avg_test_iter", "PassVer@1", "total_tokens",
]
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


def _is_complete(stem: str, lang: str) -> bool:
    for j in (BASE / stem / lang / "runs").glob("*.json"):
        try:
            if json.loads(j.read_text(encoding="utf-8")).get("status") == "complete":
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for srs in sorted(SRS_DIR.glob("*.xml")):
        for lang in LANGS:
            d = BASE / srs.stem / lang
            row = {c: "" for c in COLUMNS}
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
            row["status"] = "complete" if _is_complete(srs.stem, lang) else "incomplete"
            rows.append(row)
    return rows


def main() -> None:
    rows = build_rows()
    with COMBINED_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    payload = {
        "model": MODEL, "strategy": "cot", "flags": FLAGS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "columns": COLUMNS,
        "cells": rows,
    }
    COMBINED_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    done = sum(1 for r in rows if r.get("status") == "complete")
    print(f"emit_cot_results: {done}/{len(rows)} cells complete -> "
          f"{COMBINED_CSV.name} + {COMBINED_JSON.name}")


if __name__ == "__main__":
    main()
