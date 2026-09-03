#!/usr/bin/env bash
#SBATCH --job-name=cleanroom-vllm
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# ─────────────────────────────────────────────────────────────────────────────
# serve_vllm.sbatch — serve one study model on Minsky's single 48 GB GPU behind
# an OpenAI-compatible API, using the stock vllm-openai image.
#
#   sbatch --export=ALL,MODEL_KEY=qwen-coder scripts/minsky/serve_vllm.sbatch
#   sbatch --export=ALL,MODEL_KEY=qwen3      scripts/minsky/serve_vllm.sbatch
#   sbatch --export=ALL,MODEL_KEY=r1-distill scripts/minsky/serve_vllm.sbatch
#
# Minsky policy: 1 GPU per user → only ONE of these can run at a time, and the
# 8 h walltime is the hard ceiling on a pipeline run. Do NOT requeue this job:
# a restart lands on a different node and silently breaks your SSH tunnel.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Slurm COPIES the batch script into its spool directory before running it, so
# `dirname $BASH_SOURCE` points at /var/spool/slurmd/job<N>/ — not at the repo. Resolve the
# script directory from the submit directory instead, and fall back to $BASH_SOURCE for the
# case where this file is executed directly rather than through sbatch.
find_conf() {
  local candidates=(
    "${CLEANROOM_MINSKY_DIR:-}"
    "${SLURM_SUBMIT_DIR:-}/scripts/minsky"
    "${SLURM_SUBMIT_DIR:-}"
    "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
  )
  for dir in "${candidates[@]}"; do
    [ -n "$dir" ] && [ -f "$dir/models.conf" ] && { echo "$dir"; return 0; }
  done
  return 1
}

REPO_DIR="${REPO_DIR:-$(find_conf)}" || {
  echo "ERROR: cannot locate models.conf."
  echo "  Submit from the repo root (so SLURM_SUBMIT_DIR points at it), or set"
  echo "  CLEANROOM_MINSKY_DIR=/path/to/scripts/minsky in --export."
  echo "  Looked in: \$CLEANROOM_MINSKY_DIR, \$SLURM_SUBMIT_DIR{,/scripts/minsky}"
  exit 1
}
. "$REPO_DIR/models.conf"

MODEL_KEY="${MODEL_KEY:?set MODEL_KEY=qwen-coder|qwen3|r1-distill}"
model_config "$MODEL_KEY"

VLLM_SIF="${VLLM_SIF:-/opt/apptainer/images/vllm-openai-latest.sif}"
VLLM_PORT="${VLLM_PORT:-$(( 8000 + (UID % 1000) ))}"
MAX_LEN="${MAX_LEN:-32768}"

STATE_DIR="$HOME/.cleanroom-vllm"; JOBS_DIR="$STATE_DIR/jobs"
mkdir -p "$JOBS_DIR"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "════════════════════════════════════════════════════════════"
echo " job     : $SLURM_JOB_ID on $(hostname -s)"
echo " key     : $MODEL_KEY   variant: ${MODEL_VARIANT:-$DEFAULT_VARIANT}"
echo " model   : $VLLM_MODEL"
echo " port    : $VLLM_PORT"
echo " extra   : ${EXTRA_ARGS[*]}"
echo " note    : $NOTE"
echo "════════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "-- free space in \$HF_HOME (weights land here):"
df -h "$HF_HOME" 2>/dev/null | tail -1 || true

# Register where we landed so tunnel.sh finds the node without hardcoding it.
cat > "$JOBS_DIR/$SLURM_JOB_ID.json" <<JSON
{
  "job_id": "$SLURM_JOB_ID",
  "key": "$MODEL_KEY",
  "node": "$(hostname -s)",
  "port": $VLLM_PORT,
  "model": "$VLLM_MODEL",
  "started": "$(date -Is)"
}
JSON
trap 'rm -f "$JOBS_DIR/$SLURM_JOB_ID.json"' EXIT

# --kv-cache-dtype fp8 roughly doubles usable context in the memory left over
# after the weights — worth it on a 48 GB card holding a 32B model.
apptainer exec --nv \
  --bind "$HOME:$HOME" \
  --env HF_HOME="$HF_HOME" \
  "$VLLM_SIF" \
  vllm serve "$VLLM_MODEL" \
    --served-model-name "$VLLM_MODEL" \
    --host 0.0.0.0 --port "$VLLM_PORT" \
    --tensor-parallel-size "$VLLM_TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --kv-cache-dtype fp8 \
    "${EXTRA_ARGS[@]}"
