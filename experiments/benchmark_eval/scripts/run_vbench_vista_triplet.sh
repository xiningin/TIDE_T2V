#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "$SCRIPT_DIR/vbench_eval_common.sh"

# This is a deliberately separate output root: it is a matched 80-prompt
# subset, not the 1,293-video main VBench arm.
export EXPECTED_VIDEOS=80
export OUTPUT_ROOT="${TRIPLET_OUTPUT_ROOT:-$ROOT/experiments/cross_method_results/vista/vista_vbench_eval/triplet_results}"
export LOG_ROOT="${TRIPLET_LOG_ROOT:-$ROOT/experiments/cross_method_results/vista/vista_vbench_eval/triplet_logs}"

run_group "vista_triplet" "${VISTA_TRIPLET_GPU_IDS:-6,7}" \
  vista_self_improve "$ROOT/experiments/cross_method_results/vista/vista_vbench_eval/matched_triplet_1sample/vista_self_improve" \
  sana_ours "$ROOT/experiments/cross_method_results/vista/vista_vbench_eval/matched_triplet_1sample/sana_ours" \
  sana_raw "$ROOT/experiments/cross_method_results/vista/vista_vbench_eval/matched_triplet_1sample/sana_raw"
