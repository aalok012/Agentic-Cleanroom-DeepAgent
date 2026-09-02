# API Usage Log

Track OpenAI **token consumption** and **LLM call counts** for the shared professor API key.
Updated automatically at the end of each `run_pipeline.py` invocation (including failed runs).

## Cumulative totals

| metric | value |
|---|---:|
| runs logged | 1 |
| LLM calls | 0 |
| input tokens | 0 |
| output tokens | 0 |
| total tokens | 0 |
| estimated cost (USD) | $0.0000 |

## Run history

| date (UTC) | run_id | status | SRS | model | calls | input tok | output tok | total tok | cost (USD) | seconds | result |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-08-31 22:36:47 | `20260831-223647-human` | failed | Human.xml | qwen/qwen-2.5-coder-32b-instruct | 0 | 0 | 0 | 0 | $0.0000 | 0.0 | [detail](results/new_feature_2026-08-26/raw_results/human-deepagent-20260831/runs/20260831-223647-human.md) |

## Notes

- Costs use `src/cleanroom/utils/cost.py` list prices for the reported model.
- `status=failed` means the pipeline exited early; tokens from completed stages are still counted.
- **Per-run detail** (config, pass@k, stage breakdown): `outputs/runs/<run_id>.md` and the cumulative index [RUN_RESULTS.md](RUN_RESULTS.md).
- Latest per-SRS snapshot (overwritten): `outputs/<srs>_run_report.md`.
