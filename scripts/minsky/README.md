# Running the pipeline against Minsky

`minsky.cs.txstate.edu` is a Slurm + Apptainer GPU box in the CS department. There is
no permanent LLM endpoint: you submit a job that serves the model on the GPU node,
tunnel to it from your laptop, and run the pipeline locally.

```
laptop                  minsky login            GPU node (per job)
┌────────────────┐      ┌────────────┐          ┌──────────────────────┐
│ run_pipeline   │      │            │          │ apptainer --nv       │
│  LLM_BASE_URL= │─8000─┤ ssh -L     ├──────────┤ vllm serve           │
│  localhost:8000│      │            │  :80XX   │ /v1/chat/completions │
└────────────────┘      └────────────┘          └──────────────────────┘
```

**Off campus you must be on the Texas State VPN** (remoteaccess.txstate.edu) before
the tunnel will connect.

## The constraints that shape everything here

| Minsky policy | Consequence for this pipeline |
|---|---|
| **1 GPU per user** | No tensor parallelism. Only one model served at a time — the two models run in **sequence**, never side by side. |
| **48 GB VRAM** (RTX 6000 Ada / L40S) | A 32B model at bf16 is ~64 GB of weights. **Both models must be quantized.** |
| **8 h max walltime** | Hard ceiling on a pipeline run. A run that outlives the job dies mid-experiment. |
| Second job → `QOSMaxGRESPerUser` | Expected, not an error: it starts when the first finishes. |

Checkpoint/requeue (§06 of the Minsky guide) is for training, **not** for this server job:
a requeue lands on a different node and silently breaks your tunnel. Size the walltime instead.

## The models

Configured in [`models.conf`](models.conf), selected with `MODEL_KEY`:

| Key | Served model | Quantization | Weights | Tool calling |
|---|---|---|---|---|
| `qwen-coder` | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | 4-bit AWQ (official build) | ~19 GB | native (hermes parser) |
| `qwen3` | `Qwen/Qwen3-32B-AWQ` | 4-bit AWQ (official build) | ~19 GB | native (hermes parser) |
| `r1-distill` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | FP8 at load | ~32 GB | **none** |

AWQ is preferred — smaller, faster to start. The R1 distill has no official 4-bit build,
so it falls back to FP8, which is native on both Ada card types and quantized by vLLM at
load time. That means **it downloads the full ~64 GB bf16 checkpoint first** — check you
have the disk quota before the first R1 run. If you vet a community AWQ quant, set
`MODEL_AWQ` in `models.conf` and use `MODEL_VARIANT=awq`.

`--kv-cache-dtype fp8` roughly doubles usable context in the VRAM left after weights,
which matters most for the FP8 model (~13 GB spare vs ~27 GB for AWQ).

### Qwen3-32B is a hybrid reasoning model

It ships with thinking **on** by default, so it emits `<think>…</think>` before the answer —
the same parsing hazard as the R1 distill, handled the same way with
`--reasoning-parser qwen3`. Unlike the distill it **does** support tool calling, so structured
output still goes through the native hermes path instead of the text-recovery fallback; it is
the closest thing here to a like-for-like comparison against `qwen-coder`, differing in model
generation rather than in serving contract.

Qwen advise temperature 0.6 in thinking mode (it degenerates into loops at 0), so `tunnel.sh`
and `run_all.sh` write `CLEANROOM_LLM_TEMPERATURE=0.6` for this key too.

The one thing to check on the first run: `--reasoning-parser qwen3` needs a reasonably recent
vLLM. If `/opt/apptainer/images/vllm-openai-latest.sif` rejects the flag, drop it (Qwen3's
`<think>` block is then left in `content` and `_extract_json` may trip on it) or point
`VLLM_SIF` at a newer image.

### Two things specific to the R1 distill

1. **It emits `<think>…</think>` before answering.** `--reasoning-parser deepseek_r1`
   routes that into `reasoning_content`, so the `content` the pipeline parses is clean.
   Without it, [`_extract_json`](../../src/cleanroom/utils/llm_client.py#L200) can pick up
   braces from inside the reasoning block.
2. **It has no tool-calling support.** The pipeline's `function_calling` structured output
   gets nothing back and falls through to the text-recovery path in
   [`_coerce_structured`](../../src/cleanroom/utils/llm_client.py#L214). That path exists for
   exactly this case, but it's a looser contract than Qwen's native tool calls — expect more
   retries, and treat any schema-failure gap between the two models as **partly a serving
   artifact, not purely model quality**.

DeepSeek also advise temperature ~0.6 for the distills (they loop at 0). `tunnel.sh` writes
`CLEANROOM_LLM_TEMPERATURE=0.6` into `.env` for this model only; it raises the deterministic
stages while the pass@k stage keeps its own sampling temperature.

## Run it

**1 — serve** (on Minsky):

```bash
sbatch --export=ALL,MODEL_KEY=qwen-coder scripts/minsky/serve_vllm.sbatch
squeue -u $USER                      # wait for ST=R
tail -f cleanroom-vllm_*.out         # wait for "Uvicorn running on http://0.0.0.0:PORT"
```

First start is slow — it downloads weights into `$HF_HOME`, which persists between jobs.

**2 — tunnel + configure** (on your laptop, VPN up if off campus):

```bash
MINSKY_USER=your-netid ./scripts/minsky/tunnel.sh
```

Reads the job's registered node over SSH, writes `.env`, and holds the tunnel open.
Leave it running.

**3 — run** (second terminal):

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
./run_example.sh
```

Then `scancel` the job, submit the next `MODEL_KEY`, re-run `tunnel.sh`, and repeat.
Results go in `results/new_feature_2026-08-26/`.

## Still worth checking

- **Disk quota** for the R1 checkpoint (~64 GB into `~/.cache/huggingface`). Run
  [`probe_minsky.sh`](probe_minsky.sh) if you're unsure — it's read-only.
- **Whether the GPU node has outbound internet** to pull from HuggingFace. If not,
  pre-download from the login node before submitting.
