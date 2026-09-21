#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TRAIN_ROOT="$ROOT/outputs/ablation_negative_frozen"
ARM_ROOT="$ROOT/experiments/training_set_ablation/negative_frozen"
T2V_PY="$ROOT/../miniconda3/envs/t2v/bin/python"

echo "[$(date -Is)] waiting for negative-frozen training finalization"
while :; do
  if [[ -f "$TRAIN_ROOT/controller_summary.json" ]]; then
    # Require the controller to have written a successful finalization marker.
    if "$T2V_PY" - "$TRAIN_ROOT/controller_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get('finalization_complete') is True else 1)
PY
    then
      break
    fi
  fi
  if ! pgrep -f 'run_batch_skill_controller.py.*ablation_negative_frozen' >/dev/null; then
    echo "training controller exited before finalization" >&2
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] training complete; expanding aligned 431-prompt manifest"
mkdir -p "$ARM_ROOT"
"$T2V_PY" experiments/training_set_ablation/expand_trigger_prompts_full.py \
  --output-dir "$ARM_ROOT" \
  --cache-dir "$ARM_ROOT" \
  --skill-dir "$TRAIN_ROOT/working_skill" \
  --rules-file "$TRAIN_ROOT/working_skill/rules.json" \
  --activation-mode normal \
  --drop-conditional-negative \
  --workers 8

echo "[$(date -Is)] generating 1293 Wan2.2 videos on GPUs 0-7"
mkdir -p "$ARM_ROOT/logs"
pids=()
for shard in $(seq 0 7); do
  gpu="$shard"
  "$T2V_PY" experiments/benchmark_eval/scripts/generate_videos.py \
    --experiment "$ARM_ROOT/experiment.json" --gpu "$gpu" \
    --shard-index "$shard" --num-shards 8 --continue-on-error \
    >"$ARM_ROOT/logs/generate_shard_${shard}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
  sleep 2
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if ((failed)); then
  echo "one or more Wan2.2 generation shards failed" >&2
  exit 1
fi
"$T2V_PY" experiments/benchmark_eval/scripts/audit_generation.py \
  --experiment "$ARM_ROOT/experiment.json"

echo "[$(date -Is)] running 9-dimension VBench on GPUs 0-3"
experiments/training_set_ablation/run_vbench2_ablation.sh negative_frozen
echo "[$(date -Is)] negative-frozen pipeline complete"
