#!/usr/bin/env bash
# Prepare, generate, and audit one benchmark arm. Evaluation is intentionally separate.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
ARM=${1:-}
ACTION=${2:-all}

if [[ -z "${ARM}" ]]; then
  echo "Usage: $0 ARM [validate|prepare|generate|audit|all]" >&2
  exit 2
fi

experiment_path() {
  PYTHONPATH="${SCRIPT_DIR}" "${PYTHON_BIN}" - "${ARM}" <<'PY'
import sys
from generation_common import DEFAULT_REGISTRY, experiment_dir

arm = sys.argv[1]
print(experiment_dir(arm, DEFAULT_REGISTRY) / arm / "experiment.json")
PY
}

run_validate() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_generation.py" \
    --arm "${ARM}" \
    --validate-only
}

run_prepare() {
  local args=(
    --arm "${ARM}"
    --workers "${REWRITE_WORKERS:-8}"
    --rewrite-gpu "${REWRITE_GPU:-0}"
  )
  if [[ -n "${PREPARE_MAX_TASKS:-}" ]]; then
    args+=(--max-tasks "${PREPARE_MAX_TASKS}")
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_generation.py" "${args[@]}"
}

run_generate() {
  local experiment
  experiment=$(experiment_path)
  if [[ ! -f "${experiment}" ]]; then
    echo "Missing ${experiment}; run the prepare action first." >&2
    return 2
  fi

  local gpu_text=${GPU_IDS:-0,1,2,3,4,5,6,7}
  local gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "${gpu_text}"
  if [[ ${#gpu_ids[@]} -eq 0 ]]; then
    echo "GPU_IDS did not contain any devices." >&2
    return 2
  fi

  local arm_root
  arm_root=$(dirname -- "${experiment}")
  mkdir -p "${arm_root}/logs"
  local num_shards=${#gpu_ids[@]}
  local pids=()
  local logs=()
  local shard_index gpu log_path
  for shard_index in "${!gpu_ids[@]}"; do
    gpu=${gpu_ids[${shard_index}]}
    log_path="${arm_root}/logs/generate_shard_${shard_index}_gpu_${gpu}.log"
    local args=(
      --experiment "${experiment}"
      --gpu "${gpu}"
      --shard-index "${shard_index}"
      --num-shards "${num_shards}"
    )
    if [[ "${CONTINUE_ON_ERROR:-1}" == "1" ]]; then
      args+=(--continue-on-error)
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      args+=(--dry-run)
    fi
    if [[ -n "${MAX_TASKS:-}" ]]; then
      args+=(--max-tasks "${MAX_TASKS}")
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_videos.py" "${args[@]}" \
      >"${log_path}" 2>&1 &
    pids+=("$!")
    logs+=("${log_path}")
    echo "Started shard ${shard_index}/${num_shards} on GPU ${gpu}; log: ${log_path}"
    if [[ ${shard_index} -lt $((num_shards - 1)) ]]; then
      sleep "${STARTUP_STAGGER_SECONDS:-2}"
    fi
  done

  local failed=0
  local index
  set +e
  for index in "${!pids[@]}"; do
    wait "${pids[${index}]}"
    if [[ $? -ne 0 ]]; then
      echo "Shard ${index} failed; inspect ${logs[${index}]}" >&2
      failed=1
    fi
  done
  set -e
  return "${failed}"
}

run_audit() {
  local experiment
  experiment=$(experiment_path)
  if [[ ! -f "${experiment}" ]]; then
    echo "Missing ${experiment}; run the prepare action first." >&2
    return 2
  fi
  "${PYTHON_BIN}" "${SCRIPT_DIR}/audit_generation.py" --experiment "${experiment}"
}

case "${ACTION}" in
  validate)
    run_validate
    ;;
  prepare)
    run_prepare
    ;;
  generate)
    run_generate
    ;;
  audit)
    run_audit
    ;;
  all)
    run_validate
    run_prepare
    if run_generate; then
      run_audit
    else
      run_audit || true
      exit 1
    fi
    ;;
  *)
    echo "Unknown action '${ACTION}'. Expected validate, prepare, generate, audit, or all." >&2
    exit 2
    ;;
esac
