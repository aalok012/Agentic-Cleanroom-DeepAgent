# Full pipeline — run metrics across different models

_Auto-appended after every `run_pipeline.py` run. One section per run._

---

## Run 20260831-223647-human — FAILED

- **When**: 2026-08-31 17:36:47
- **SRS**: Human.xml  ·  **language**: ?  ·  **stack**: auto
- **Command**: `python run_pipeline.py data/srs/Human.xml --gen-driver deepagent --repair-driver deepagent --no-prove --certify --samples 3 --output-dir results/new_feature_2026-08-26/raw_results/human-deepagent-20260831`

### Models used (per stage)

| stage | model |
|---|---|
| spec | `qwen/qwen-2.5-coder-32b-instruct` |
| dependency | `qwen/qwen-2.5-coder-32b-instruct` |
| planning | `qwen/qwen-2.5-coder-32b-instruct` |
| code | `qwen/qwen-2.5-coder-32b-instruct` |
| test | `qwen/qwen-2.5-coder-32b-instruct` |
| proof | `qwen/qwen-2.5-coder-32b-instruct` |
| cert | `qwen/qwen-2.5-coder-32b-instruct` |

### Pipeline flags

- agents: spec=on, dependency=on, planning=on, proof=off, code=on, test=on, certify=on, recovery=on, compile_repair=off
- certify=True · samples=3 · k=[1] · prove=False · max_cert_loops=2 · max_compile_repair_loops=2 · repair_driver=deepagent
- temperature=0.0 · cert_temperature=0.4 · prove_rounds=6 · baseline=False

- tokens: 0 (in 0 / out 0) · calls 0
- estimated cost: $0.0000

- proved feature ids: (none)

