#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_example.sh — end-to-end CLEANROOM-AGENT demo on a single SRS.
#
# Runs the full pipeline (contracts → dependency → planning → clean-room
# proof/code/test generation → proof-guided + pass@k certification) on ONE
# specification and prints where the per-requirement audit labels landed.
#
# Usage:
#   ./run_example.sh                       # default: Human.xml, python, Qwen2.5-Coder-32B
#   MODEL=gpt-5.1 ./run_example.sh         # pick a model (id as your endpoint lists it)
#   SRS="data/srs/dineout_srs.xml" LANG=java ./run_example.sh
#
# Requires: uv (https://docs.astral.sh/uv/) and an OpenAI-compatible endpoint —
# a self-hosted server via LLM_BASE_URL, or OPENAI_API_KEY for api.openai.com.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via environment) --------------------------------------
SRS="${SRS:-data/srs/Human.xml}"          # smallest subject (2 FRs) → fast demo
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"  # must match your endpoint's /v1/models
LANG="${LANG:-python}"                     # python | java | javascript
OUT="${OUT:-outputs/example}"

# ---- preflight --------------------------------------------------------------
if [ ! -f .env ] && [ -z "${LLM_BASE_URL:-}${OPENAI_BASE_URL:-}${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: no LLM endpoint configured. Create a .env with:"
  echo "    LLM_BASE_URL=http://your-server:8000/v1"
  echo "    LLM_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
  echo "  (or OPENAI_API_KEY=sk-... to use api.openai.com)."
  exit 1
fi
[ -f .env ] && set -a && . ./.env && set +a || true

echo "════════════════════════════════════════════════════════════════"
echo " CLEANROOM-AGENT — single-example run"
echo "   SRS        : $SRS"
echo "   model      : $MODEL"
echo "   language   : $LANG"
echo "   output dir : $OUT"
echo "════════════════════════════════════════════════════════════════"

# ---- run the full pipeline (proof-guided + pass@k certification) -------------
uv run python run_pipeline.py "$SRS" \
  --model "$MODEL" \
  --language "$LANG" \
  --prove --certify \
  --output-dir "$OUT"

echo ""
echo "Done. Per-requirement audit labels and metrics are in:"
echo "   $OUT/  (see the *_run_report.md and runs/*.json for this run)"
