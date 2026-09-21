#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash vbench2_eval/run_pipeline.sh [validate|prepare|generate|evaluate|all]

Environment overrides:
  SUITE, OUTPUT_ROOT, INITIAL_SKILL, TRAINED_SKILL, T2V_CONFIG
  INITIAL_ARM_NAME (default initial), TRAINED_ARM_NAME (default trained)
  GPU_IDS (generation, default 0,1,2,3,4,5), EVAL_GPU, DIMENSIONS
  SAMPLES_PER_PROMPT, DIVERSITY_SAMPLES_PER_PROMPT, BASE_SEED, MAX_PROMPTS
  RESUME (0/1), CONTINUE_ON_ERROR (0/1), VBENCH_DIR, EVAL_CONDA_ENV
EOF
}

STAGE="${1:-all}"
if (($# > 1)); then
  usage >&2
  exit 2
fi
case "${STAGE}" in
  validate|prepare|generate|evaluate|all) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

SUITE="${SUITE:-${SCRIPT_DIR}/data/vbench2_quick_eval_44.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/vbench2_skill_eval}"
INITIAL_SKILL="${INITIAL_SKILL:-${PROJECT_ROOT}/outputs/evalcrafter_rulepool4_qwen36_local/wave_0000/frozen_skill}"
TRAINED_SKILL="${TRAINED_SKILL:-${PROJECT_ROOT}/outputs/evalcrafter_rulepool4_qwen36_local/wave_0011/frozen_skill}"
INITIAL_ARM_NAME="${INITIAL_ARM_NAME:-initial}"
TRAINED_ARM_NAME="${TRAINED_ARM_NAME:-trained}"
T2V_CONFIG="${T2V_CONFIG:-${PROJECT_ROOT}/config/deepseek_qwen37plus_sana_posneg_train.yaml}"
T2V_PY="${T2V_PY:-python}"
CONDA_BIN="${CONDA_BIN:-conda}"
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-vbench2}"
VBENCH_DIR="${VBENCH_DIR:-/path/to/VBench-2.0}"
MODEL_ENV_FILE="${MODEL_ENV_FILE:-${SCRIPT_DIR}/model_env.sh}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
EVAL_GPU="${EVAL_GPU:-0}"
WORKERS="${WORKERS:-6}"
BASE_SEED="${BASE_SEED:-42}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
DIVERSITY_SAMPLES_PER_PROMPT="${DIVERSITY_SAMPLES_PER_PROMPT:-20}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
DIMENSIONS="${DIMENSIONS:-}"
RESUME="${RESUME:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

OUTPUT_ROOT="$(realpath -m -- "${OUTPUT_ROOT}")"
EXPERIMENT="${OUTPUT_ROOT}/experiment.json"
IFS=',' read -r -a GENERATION_GPUS <<< "${GPU_IDS}"
if ((${#GENERATION_GPUS[@]} == 0)); then
  printf 'GPU_IDS must contain at least one GPU id.\n' >&2
  exit 2
fi
SHARDS="${SHARDS:-${#GENERATION_GPUS[@]}}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

require_file() {
  [[ -f "$1" ]] || { log "ERROR: required file not found: $1"; exit 1; }
}

prepare() {
  local command=(
    "${T2V_PY}" "${SCRIPT_DIR}/prepare_skill_arms.py"
    --suite "${SUITE}"
    --output "${OUTPUT_ROOT}"
    --skill-arm "${INITIAL_ARM_NAME}=${INITIAL_SKILL}"
    --skill-arm "${TRAINED_ARM_NAME}=${TRAINED_SKILL}"
    --rewrite-config "${T2V_CONFIG}"
    --t2v-config "${T2V_CONFIG}"
    --workers "${WORKERS}"
    --shards "${SHARDS}"
    --base-seed "${BASE_SEED}"
    --samples-per-prompt "${SAMPLES_PER_PROMPT}"
    --diversity-samples-per-prompt "${DIVERSITY_SAMPLES_PER_PROMPT}"
    --max-prompts "${MAX_PROMPTS}"
  )
  if [[ -n "${DIMENSIONS}" ]]; then
    command+=(--dimensions "${DIMENSIONS}")
  fi
  if [[ "${STAGE}" == "validate" ]]; then
    command+=(--validate-only)
  fi
  log "[prepare] suite=${SUITE} output=${OUTPUT_ROOT} shards=${SHARDS}"
  "${command[@]}"
}

generate() {
  require_file "${EXPERIMENT}"
  mkdir -p "${OUTPUT_ROOT}/receipts" "${OUTPUT_ROOT}/logs"
  local shard_count
  shard_count="$("${T2V_PY}" - "${EXPERIMENT}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["shards"])
PY
)"
  local pids=()
  local shard gpu command
  for ((shard=0; shard<shard_count; shard++)); do
    gpu="${GENERATION_GPUS[$((shard % ${#GENERATION_GPUS[@]}))]}"
    require_file "${OUTPUT_ROOT}/jobs/shard_${shard}.jsonl"
    command=(
      "${T2V_PY}" "${SCRIPT_DIR}/generate_sana_shard.py"
      --jobs "${OUTPUT_ROOT}/jobs/shard_${shard}.jsonl"
      --results "${OUTPUT_ROOT}/receipts/shard_${shard}.jsonl"
      --experiment "${EXPERIMENT}"
      --gpu "${gpu}"
    )
    [[ "${RESUME}" == "1" ]] || command+=(--no-resume)
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && command+=(--continue-on-error)
    log "[generate] shard=${shard} gpu=${gpu}"
    ("${command[@]}") >"${OUTPUT_ROOT}/logs/generate-shard-${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" != "0" ]]; then
    log "ERROR: one or more generation shards failed; inspect ${OUTPUT_ROOT}/logs"
    exit 1
  fi
  "${T2V_PY}" "${SCRIPT_DIR}/audit_outputs.py" --experiment "${EXPERIMENT}"
}

evaluate() {
  require_file "${EXPERIMENT}"
  require_file "${MODEL_ENV_FILE}"
  require_file "${CONDA_BIN}"
  require_file "${VBENCH_DIR}/vbench2/VBench2_full_info.json"
  # shellcheck source=/dev/null
  source "${MODEL_ENV_FILE}"
  "${T2V_PY}" "${SCRIPT_DIR}/audit_outputs.py" --experiment "${EXPERIMENT}"
  mapfile -t arms < <("${T2V_PY}" - "${EXPERIMENT}" <<'PY'
import json
import sys
for arm in json.load(open(sys.argv[1], encoding="utf-8"))["arms"]:
    print(arm)
PY
  )
  mkdir -p "${OUTPUT_ROOT}/logs"
  local arm command
  for arm in "${arms[@]}"; do
    command=(
      "${CONDA_BIN}" run --no-capture-output -n "${EVAL_CONDA_ENV}"
      python "${SCRIPT_DIR}/evaluate_vbench2.py"
      --experiment "${EXPERIMENT}"
      --arm "${arm}"
      --vbench-dir "${VBENCH_DIR}"
    )
    [[ "${RESUME}" == "1" ]] || command+=(--no-resume)
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && command+=(--continue-on-error)
    [[ -z "${DIMENSIONS}" ]] || command+=(--dimensions "${DIMENSIONS}")
    log "[evaluate] arm=${arm} gpu=${EVAL_GPU}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${command[@]}" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/evaluate-${arm}.log"
  done
  "${T2V_PY}" "${SCRIPT_DIR}/compare_results.py" --experiment "${EXPERIMENT}"
}

require_file "${T2V_PY}"
case "${STAGE}" in
  validate) prepare ;;
  prepare) prepare ;;
  generate) generate ;;
  evaluate) evaluate ;;
  all) prepare; generate; evaluate ;;
esac

log "Done: ${OUTPUT_ROOT}"
