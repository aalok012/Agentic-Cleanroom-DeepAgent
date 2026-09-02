# Pipeline Run Results

Paper-friendly index of every `run_pipeline.py` invocation. Each run has a dedicated markdown + JSON file under `outputs/runs/`.

| run_id | date (UTC) | status | SRS | stack | certify | pass@1 | case rate | tokens | cost (USD) | seconds | detail |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `20260831-223647-human` | 2026-08-31 22:36:47 | failed | Human.xml | auto | yes | - | - | 0 | $0.0000 | 0.0 | [20260831-223647-human.md](results/new_feature_2026-08-26/raw_results/human-deepagent-20260831/runs/20260831-223647-human.md) |

## Notes

- Newest runs appear at the top of the table.
- Token/cost totals across all runs: see [API_USAGE.md](API_USAGE.md).
- Latest per-SRS artifacts (overwritten each run): `outputs/<srs>_full_ir.json`, `outputs/<srs>_run_report.md`.
