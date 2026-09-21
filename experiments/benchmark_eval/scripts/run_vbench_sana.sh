#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/vbench_eval_common.sh"

run_group "sana" "${SANA_GPU_IDS:-0,1}" \
  sana_raw  "$ROOT/experiments/cross_model_results/sana-2b/sana_raw/videos/main_vbench/sana_raw" \
  sana_ours "$ROOT/experiments/cross_model_results/sana-2b/sana_ours/videos/main_vbench/sana_ours"
