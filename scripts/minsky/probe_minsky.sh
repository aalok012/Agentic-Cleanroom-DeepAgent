#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# probe_minsky.sh — READ-ONLY survey of Minsky. Submits no jobs, writes nothing
# outside /tmp. Run it on the LOGIN NODE and paste the output back.
#
#   scp scripts/minsky/probe_minsky.sh $MINSKY_USER@minsky.cs.txstate.edu:~/
#   ssh $MINSKY_USER@minsky.cs.txstate.edu 'bash ~/probe_minsky.sh'
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
h() { printf '\n══ %s %s\n' "$1" "$(printf '═%.0s' $(seq 1 $((60 - ${#1}))))"; }

h "host"
hostname -f; uname -sr

h "apptainer"
apptainer --version 2>/dev/null || echo "apptainer NOT on PATH"
echo "-- images in /opt/apptainer/images:"
ls -lh /opt/apptainer/images/ 2>/dev/null || echo "  (cannot list)"
echo "-- anything vllm/sglang-shaped anywhere obvious:"
ls /opt/apptainer/images/ 2>/dev/null | grep -iE 'vllm|sglang|tgi|triton|llm' || echo "  none"

h "DISK — the R1 bf16 checkpoint is ~64 GB; this is the main open question"
df -h "$HOME" 2>/dev/null | tail -1
quota -s 2>/dev/null || echo "  (no quota command)"
echo "-- already-downloaded weights:"
du -sh "$HOME/.cache/huggingface" 2>/dev/null || echo "  no HF cache yet"
ls "$HOME/.cache/huggingface/hub" 2>/dev/null || true

h "slurm partitions + GPUs"
sinfo -o '%20P %10D %15N %20G %10m %12l' 2>/dev/null || echo "  sinfo unavailable"
echo "-- GPU types actually configured:"
sinfo -o '%N %G' -h 2>/dev/null | sort -u
echo "-- what your account may submit to:"
sacctmgr -nP show assoc user="$USER" format=Account,Partition,QOS 2>/dev/null || echo "  (sacctmgr unavailable)"
scontrol show partition 2>/dev/null | grep -E 'PartitionName|MaxTime|TRES=' | head -30

h "limits"
echo "-- your running/pending jobs:"; squeue -u "$USER" 2>/dev/null
echo "-- QOS limits:"; sacctmgr -nP show qos format=Name,MaxWall,MaxTRESPU 2>/dev/null | head

h "storage (model weights are large — 100+ GB for a big model)"
df -h "$HOME" 2>/dev/null
for d in /scratch /work /data "$HOME/.cache/huggingface"; do
  [ -e "$d" ] && { echo "-- $d:"; df -h "$d" 2>/dev/null | tail -1; du -sh "$d" 2>/dev/null; }
done
echo "-- quota:"; quota -s 2>/dev/null || echo "  (no quota cmd)"
echo "-- existing HF cache (weights already downloaded?):"
ls "$HOME/.cache/huggingface/hub" 2>/dev/null || echo "  none"
ls /opt/models /shared/models /data/models 2>/dev/null || true

h "network — can a compute node be reached by hostname from here?"
echo "  (checked for real once a job is running; noting login hostname resolution)"
getent hosts "$(hostname -s)" 2>/dev/null
echo "-- outbound internet from login node (needed to pull weights from HF):"
curl -s -o /dev/null -w '  huggingface.co -> HTTP %{http_code} in %{time_total}s\n' \
  --max-time 10 https://huggingface.co || echo "  no outbound HTTPS"

h "python/tooling on login node"
python3 -V 2>/dev/null; which uv 2>/dev/null || echo "  uv not installed (only needed if you run the pipeline here)"

h "done"
echo "Paste everything above back."
