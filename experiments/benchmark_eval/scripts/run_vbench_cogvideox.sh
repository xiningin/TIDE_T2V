#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/vbench_eval_common.sh"

run_group "cogvideox" "${COGVIDEOX_GPU_IDS:-2,3}" \
  cogvideox15_raw  "$ROOT/experiments/cross_model_results/cogvideox1.5-5b/cogvideox15_raw/videos/main_vbench/cogvideox15_raw" \
  cogvideox15_ours "$ROOT/experiments/cross_model_results/cogvideox1.5-5b/cogvideox15_ours/videos/main_vbench/cogvideox15_ours" \
  cogvideox15_vpo  "$ROOT/experiments/cross_method_results/vpo/cogvideox15_vpo/videos/main_vbench/cogvideox15_vpo"
