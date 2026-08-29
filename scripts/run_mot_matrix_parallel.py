#!/usr/bin/env python3
"""Parallel MoT matrix runner — DeepSeek only, --prompt-strategy mot.

Runs every SRS in data/srs/*.xml across the 3 languages (python/fastapi, java/spring,
javascript/express) as a flat pool of N concurrent pipelines (default 2). Each cell has a
per-run wall-clock timeout and is retried (with --resume, so it continues from the last
checkpoint) on any failure or timeout. Self-contained and unattended: when the matrix
finishes it emits the per-SRS CSVs (collect_metrics.py) and the 3-way comparison
(compare_three_way.py), so results are on disk by morning even without supervision.

Tunable via env: MOT_CONCURRENCY (2), MOT_TIMEOUT_MIN (120), MOT_MAX_ATTEMPTS (3).
Idempotent: a cell whose runs/*.json already shows "status": "complete" is skipped, so the
whole script is safe to re-run.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from src.cleanroom.utils.llm_client import DEFAULT_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# Model under test. Defaults to whatever .env configures (LLM_MODEL), so this driver
# follows the served model instead of the retired OpenRouter slug it was written against.
MODEL = os.getenv("EXPERIMENT_MODEL") or DEFAULT_MODEL
FLAGS = ["--prove", "--certify", "--max-cert-loops", "4", "--prove-rounds", "4"]
BASE = ROOT / "outputs" / "mot"
LANGS = [("python", "fastapi"), ("java", "spring"), ("javascript", "express")]

CONCURRENCY = int(os.environ.get("MOT_CONCURRENCY", "2"))
# Per-run wall-clock BACKSTOP (minutes) — only catches a truly hung pipeline. This is NOT the
# API timeout: each LLM request already times out at CLEANROOM_LLM_TIMEOUT (120s) and retries
# (CLEANROOM_LLM_MAX_RETRIES) in the client, so hangs are handled there. Kept generous (180min >
# the ~120min longest healthy run) so slow-but-working cells are never clipped.
TIMEOUT_SEC = int(os.environ.get("MOT_TIMEOUT_MIN", "180")) * 60
MAX_ATTEMPTS = int(os.environ.get("MOT_MAX_ATTEMPTS", "3"))
# Per-LLM-request timeout (seconds) + retries, enforced inside the client. This is the "120s
# timeout, retry on API failure" the run is configured for.
LLM_TIMEOUT_SEC = os.environ.get("CLEANROOM_LLM_TIMEOUT", "120")
LLM_MAX_RETRIES = os.environ.get("CLEANROOM_LLM_MAX_RETRIES", "3")

_all_srs = sorted(glob.glob("data/srs/*.xml"))
# Comparison SRS (have baseline + CoT rows for the 3-way table) run first; foodsaver — the
# slowest SRS and the only one with no baseline/CoT counterpart — runs last, so a tight night
# leaves foodsaver incomplete rather than a comparison-relevant cell.
SRS_FILES = ([s for s in _all_srs if "foodsaver" not in s.lower()]
             + [s for s in _all_srs if "foodsaver" in s.lower()])

MASTER_LOG = BASE / "_matrix_parallel.log"
_print_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with _print_lock:
        print(line, flush=True)
        try:
            with MASTER_LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def cell_complete(out: Path) -> bool:
    for j in out.glob("runs/*.json"):
        try:
            if json.loads(j.read_text(encoding="utf-8")).get("status") == "complete":
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def run_cell(stem: str, srs: str, lang: str, stack: str) -> tuple[str, str]:
    out = BASE / stem / lang
    name = f"{stem}/{lang}"
    if cell_complete(out):
        log(f"skip    {name}  (already complete)")
        return ("skip", name)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "run.log"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        cmd = [
            "uv", "run", "python", "run_pipeline.py", srs,
            "--prompt-strategy", "mot", "--model", MODEL,
            "--language", lang, "--stack", stack,
            *FLAGS, "--output-dir", str(out),
        ]
        if attempt > 1 or (out / f"{stem}_ckpt.json").exists():
            cmd.append("--resume")  # resume from an existing checkpoint (skip completed stages)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PIPELINE_LEDGER_FILE=str(out / "ledger.md"),
                   CLEANROOM_LLM_TIMEOUT=LLM_TIMEOUT_SEC, CLEANROOM_LLM_MAX_RETRIES=LLM_MAX_RETRIES)
        log(f"start   {name}  (attempt {attempt}/{MAX_ATTEMPTS})")
        with logf.open("a", encoding="utf-8") as lf:
            lf.write(f"\n===== attempt {attempt} @ {datetime.now()} =====\n{' '.join(cmd)}\n\n")
            lf.flush()
            # start_new_session=True => own process group, so a timeout kills the whole tree.
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            try:
                rc = proc.wait(timeout=TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                lf.write(f"\n!! TIMEOUT after {TIMEOUT_SEC}s — killing process group\n")
                lf.flush()
                _kill_tree(proc)
                rc = -1

        if rc == 0 and cell_complete(out):
            log(f"ok      {name}  (attempt {attempt})")
            return ("ok", name)
        reason = "timeout" if rc == -1 else f"rc={rc}"
        more = "  — retrying" if attempt < MAX_ATTEMPTS else ""
        log(f"fail    {name}  (attempt {attempt}, {reason}){more}")
        if attempt < MAX_ATTEMPTS:
            time.sleep(20)

    log(f"FAIL    {name}  (gave up after {MAX_ATTEMPTS} attempts — see {logf})")
    return ("fail", name)


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def worker(jobq: "queue.Queue", results: list, rlock: threading.Lock) -> None:
    while True:
        try:
            item = jobq.get_nowait()
        except queue.Empty:
            return
        try:
            r = run_cell(*item)
            with rlock:
                results.append(r)
        finally:
            jobq.task_done()


def emit_outputs() -> None:
    log("matrix complete — emitting per-SRS CSVs (collect_metrics.py)")
    for srs in SRS_FILES:
        stem = Path(srs).stem
        csv_out = ROOT / f"{stem}_results_mot_deepseek.csv"
        try:
            subprocess.run(
                ["uv", "run", "python", "scripts/collect_metrics.py",
                 "--root", str(BASE / stem), "--out", str(csv_out)],
                check=False)
            log(f"  CSV  {csv_out.name}")
        except OSError as e:
            log(f"  CSV  FAILED for {stem}: {e}")

    log("consolidating all MoT cells into mot_deepseek_results.csv (emit_mot_results.py)")
    try:
        subprocess.run(["uv", "run", "python", "scripts/emit_mot_results.py"], check=False)
    except OSError as e:
        log(f"  emit_mot_results FAILED: {e}")

    log("running 3-way comparison (compare_three_way.py)")
    report = BASE / "_three_way_report.txt"
    try:
        res = subprocess.run(["uv", "run", "python", "scripts/compare_three_way.py"],
                             capture_output=True, text=True, check=False)
        report.write_text((res.stdout or "") + (res.stderr or ""), encoding="utf-8")
        with _print_lock:
            print(res.stdout, flush=True)
        log(f"3-way report written to {report}")
    except OSError as e:
        log(f"3-way comparison FAILED: {e}")


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(srs).stem, srs, lang, stack)
            for srs in SRS_FILES for (lang, stack) in LANGS]
    log("=" * 70)
    log(f"MoT parallel matrix START  model={MODEL}  strategy=mot")
    log(f"concurrency={CONCURRENCY}  api_timeout={LLM_TIMEOUT_SEC}s  api_retries={LLM_MAX_RETRIES}  "
        f"run_backstop={TIMEOUT_SEC // 60}min  cell_attempts={MAX_ATTEMPTS}")
    log(f"srs={len(SRS_FILES)}  langs={len(LANGS)}  cells={len(jobs)}")
    log(f"flags={' '.join(FLAGS)}")

    jobq: "queue.Queue" = queue.Queue()
    for j in jobs:
        jobq.put(j)
    results: list = []
    rlock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(jobq, results, rlock), daemon=False)
               for _ in range(CONCURRENCY)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = sum(1 for s, _ in results if s == "ok")
    fail = sum(1 for s, _ in results if s == "fail")
    skip = sum(1 for s, _ in results if s == "skip")
    mins = (time.time() - t0) / 60
    log(f"matrix DONE in {mins:.1f}min  ok={ok} fail={fail} skip={skip}")
    failed = [n for s, n in results if s == "fail"]
    if failed:
        log("FAILED cells: " + ", ".join(failed))

    emit_outputs()
    log("ALL DONE — deliverables: <srs>_results_mot_deepseek.csv, three_way_deepseek.csv, "
        "three_way_summary_deepseek.csv")
    log("=" * 70)


if __name__ == "__main__":
    main()
