#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/vbench_eval_common.sh"

run_group "wan" "${WAN_GPU_IDS:-4,5}" \
  wan22_raw  "$ROOT/experiments/cross_model_results/wan2.2-5b/wan22_raw/videos/main_vbench/wan22_raw" \
  wan22_ours "$ROOT/experiments/cross_model_results/wan2.2-5b/wan22_ours/videos/main_vbench/wan22_ours"
