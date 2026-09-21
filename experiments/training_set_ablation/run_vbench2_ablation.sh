#!/usr/bin/env bash
set -Eeuo pipefail

ABLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${ABLA_DIR}/../.." && pwd)"
COMMON="${PROJECT_DIR}/experiments/benchmark_eval/scripts/vbench_eval_common.sh"
source "${COMMON}"

# The installed evaluation environment is the local `vbench` conda env,
# which contains the official VBench evaluator and its nine main dimensions.
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/../miniconda3/envs/vbench/bin/python}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-${PROJECT_DIR}/../miniconda3/envs/vbench/bin/torchrun}"
export VBENCH_ROOT="${VBENCH_ROOT:-${PROJECT_DIR}/../VBench}"
export CACHE_DIR="${VBENCH_CACHE_DIR:-${PROJECT_DIR}/../../model/vbench}"
export FULL_JSON_DIR="${FULL_JSON_DIR:-${PROJECT_DIR}/experiments/benchmark_eval/data/vbench/metadata/VBench_full_info.json}"
export OUTPUT_ROOT="${ABLA_DIR}/vbench_results"
export LOG_ROOT="${ABLA_DIR}/vbench_logs"
export EXPECTED_VIDEOS=1293

arm="${1:-}"
case "${arm}" in
  no_tigger)
    run_group "ablation_no_tigger" "0,1,2,3" \
      no_tigger "${ABLA_DIR}/no_tigger/videos/sana_ablation_vbench_core/no_tigger"
    ;;
  random_trigger)
    run_group "ablation_random_trigger" "4,5,6,7" \
      random_trigger "${ABLA_DIR}/random_trigger/videos/sana_ablation_vbench_core/random_trigger"
    ;;
  negative_frozen)
    run_group "ablation_negative_frozen" "0,1,2,3" \
      negative_frozen "${ABLA_DIR}/negative_frozen/videos/sana_ablation_vbench_core/negative_frozen"
    ;;
  *)
    echo "Usage: $0 {no_tigger|random_trigger|negative_frozen}" >&2
    exit 2
    ;;
esac
