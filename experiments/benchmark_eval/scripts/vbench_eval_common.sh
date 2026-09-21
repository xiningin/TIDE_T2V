#!/usr/bin/env bash
# Shared implementation for the three GPU-pinned VBench entrypoints.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
VBENCH_ROOT="${VBENCH_ROOT:-/path/to/VBench}"
CACHE_DIR="${VBENCH_CACHE_DIR:-${HOME}/.cache/vbench}"
FULL_JSON_DIR="${FULL_JSON_DIR:-${ROOT}/experiments/benchmark_eval/data/vbench/metadata/VBench_full_info.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/benchmark_eval/outputs/vbench}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/experiments/benchmark_eval/logs}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-1293}"

DIMENSIONS=(
  subject_consistency
  motion_smoothness
  dynamic_degree
  aesthetic_quality
  imaging_quality
  overall_consistency
  multiple_objects
  human_action
  spatial_relationship
)

die() {
  echo "[vbench] ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -s "$path" ]] || die "required file is missing or empty: $path"
}

check_environment() {
  require_file "$PYTHON_BIN"
  require_file "$TORCHRUN_BIN"
  [[ -f "$VBENCH_ROOT/evaluate.py" ]] || die "VBench evaluate.py not found: $VBENCH_ROOT/evaluate.py"
  require_file "$FULL_JSON_DIR"

  # These are the local checkpoints needed by the nine dimensions below.  The
  # evaluator itself can download missing files, but failing before torchrun
  # gives a much clearer and safer error.
  require_file "$CACHE_DIR/dino_model/dino_vitbase16_pretrain.pth"
  require_file "$CACHE_DIR/amt_model/amt-s.pth"
  require_file "$CACHE_DIR/raft_model/models/raft-things.pth"
  require_file "$CACHE_DIR/clip_model/ViT-L-14.pt"
  require_file "$CACHE_DIR/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
  require_file "$CACHE_DIR/ViCLIP/ViClip-InternVid-10M-FLT.pth"
  # ViCLIP's tokenizer is not bundled with the model checkpoint.  If it is
  # missing, the evaluator falls back to wget and can silently save an HTML
  # error page (the cluster's raw.githubusercontent.com route has a bad
  # certificate), causing a late FileNotFoundError in every rank.
  require_file "$CACHE_DIR/ViCLIP/bpe_simple_vocab_16e6.txt.gz"
  require_file "$CACHE_DIR/grit_model/grit_b_densecap_objectdet.pth"
  require_file "$CACHE_DIR/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth"

  # A truncated Hugging Face download still looks non-empty and passes the
  # existence check, but torch.load then fails much later in the ViCLIP
  # dimensions with "failed finding central directory".  Check the archive
  # footer and the expected scale before starting any GPU workers.
  "$PYTHON_BIN" - "$CACHE_DIR/ViCLIP/ViClip-InternVid-10M-FLT.pth" <<'PY'
import os
import sys
import zipfile

path = sys.argv[1]
if os.path.getsize(path) < 1_000_000_000:
    raise RuntimeError(f"ViCLIP checkpoint is suspiciously small: {os.path.getsize(path)} bytes")
with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    if not any(name.endswith("data.pkl") for name in names):
        raise RuntimeError("ViCLIP checkpoint has no torch data.pkl entry")
PY

  ensure_aesthetic_weight
  "$PYTHON_BIN" - <<'PY'
import clip

if not hasattr(clip, "load"):
    location = getattr(clip, "__file__", "unknown")
    raise RuntimeError(
        f"wrong 'clip' package imported from {location}; "
        "VBench requires OpenAI CLIP (pip package: openai-clip)"
    )
PY
  # GRIT-based dimensions import these packages only after several earlier
  # dimensions have initialized.  Check them during preflight so a missing
  # dependency fails immediately before any GPU workers are launched.
  "$PYTHON_BIN" - <<'PY'
import boto3
import detectron2
PY
  "$PYTHON_BIN" - <<'PY'
from pyiqa.archs.musiq_arch import MUSIQ

assert MUSIQ is not None
PY
}

ensure_aesthetic_weight() {
  local path="$CACHE_DIR/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
  mkdir -p "$(dirname "$path")"

  # A previous failed download can leave a zero-byte file.  Validate the
  # checkpoint with torch.load, rather than checking only its existence.
  if [[ -s "$path" ]] && "$PYTHON_BIN" - "$path" <<'PY'
import sys
import torch

obj = torch.load(sys.argv[1], map_location="cpu")
if not isinstance(obj, dict) or "weight" not in obj or "bias" not in obj:
    raise RuntimeError("not a LAION aesthetic linear checkpoint")
PY
  then
    echo "[vbench] aesthetic checkpoint: $path"
    return
  fi

  local tmp="${path}.tmp.$$"
  rm -f "$tmp"
  echo "[vbench] downloading aesthetic checkpoint from GitHub API"
  # The raw.githubusercontent.com route is intercepted in some clusters and
  # may return an HTML page.  The GitHub API carries the same 4-KB checkpoint
  # as base64 JSON, so decode that response instead.
  "$PYTHON_BIN" - "$tmp" <<'PY'
import base64
import json
import ssl
import sys
import urllib.request

target = sys.argv[1]
url = "https://api.github.com/repos/LAION-AI/aesthetic-predictor/contents/sa_0_4_vit_l_14_linear.pth"
request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "TIDE-VBench"})
context = ssl._create_unverified_context()
with urllib.request.urlopen(request, context=context, timeout=90) as response:
    payload = json.load(response)
data = base64.b64decode(payload["content"])
if len(data) < 1000:
    raise RuntimeError(f"downloaded checkpoint is suspiciously small: {len(data)} bytes")
with open(target, "wb") as handle:
    handle.write(data)
PY
  if ! "$PYTHON_BIN" - "$tmp" <<'PY'
import sys
import torch

obj = torch.load(sys.argv[1], map_location="cpu")
if not isinstance(obj, dict) or "weight" not in obj or "bias" not in obj:
    raise RuntimeError("not a LAION aesthetic linear checkpoint")
PY
  then
    rm -f "$tmp"
    die "downloaded aesthetic checkpoint failed validation"
  fi
  mv -f "$tmp" "$path"
  echo "[vbench] aesthetic checkpoint installed: $path"
}

check_videos() {
  local videos_dir="$1"
  [[ -d "$videos_dir" ]] || die "video directory not found: $videos_dir"
  local count
  # Follow symlinks so evaluation bundles can reference audited source videos
  # without duplicating hundreds of gigabytes of MP4 data.
  count="$(find -L "$videos_dir" -maxdepth 1 -type f -iname '*.mp4' | wc -l)"
  [[ "$count" -eq "$EXPECTED_VIDEOS" ]] || die "$videos_dir has $count mp4 files; expected $EXPECTED_VIDEOS"
  echo "[vbench] videos: $count in $videos_dir"
}

has_result() {
  local output_dir="$1"
  find "$output_dir" -maxdepth 1 -type f -name '*_eval_results.json' -print -quit 2>/dev/null | grep -q .
}

run_one() {
  local arm="$1"
  local videos_dir="$2"
  local gpu_ids="$3"

  local output_dir="$OUTPUT_ROOT/$arm"
  local log_file="$LOG_ROOT/vbench_${arm}.log"
  local -a gpu_id_list
  IFS=',' read -r -a gpu_id_list <<< "$gpu_ids"
  local nproc="${#gpu_id_list[@]}"
  ((nproc >= 1)) || die "no GPU ids supplied for $arm"
  local gpu_id
  for gpu_id in "${gpu_id_list[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "invalid GPU id '$gpu_id' in '$gpu_ids'"
  done
  mkdir -p "$output_dir" "$LOG_ROOT"

  check_videos "$videos_dir"
  if has_result "$output_dir"; then
    echo "[vbench] $arm already has an eval result; skipping"
    return 0
  fi

  echo "[vbench] starting $arm on CUDA_VISIBLE_DEVICES=$gpu_ids"
  echo "[vbench] log: $log_file"
  {
    echo "===== $(date -Is) START $arm ====="
    echo "videos_dir=$videos_dir"
    echo "output_dir=$output_dir"
    echo "gpu_ids=$gpu_ids"
    echo "dimensions=${DIMENSIONS[*]}"
  } >"$log_file"

  # Do not rely on `set -e` here: errexit is suppressed when a compound
  # command is used as the left side of `||`, which previously allowed the
  # trailing log message to overwrite torchrun's non-zero exit status.
  set +e
  CUDA_VISIBLE_DEVICES="$gpu_ids" \
    VBENCH_CACHE_DIR="$CACHE_DIR" \
    PYTHONPATH="$VBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$TORCHRUN_BIN" --nproc_per_node="$nproc" --standalone \
      "$VBENCH_ROOT/evaluate.py" \
      --full_json_dir "$FULL_JSON_DIR" \
      --videos_path "$videos_dir" \
      --output_path "$output_dir" \
      --dimension "${DIMENSIONS[@]}" \
      --load_ckpt_from_local True >>"$log_file" 2>&1
  local eval_status=$?
  set -e

  # Intentionally omit --read_frame False.  VBench declares that argument
  # with type=bool, so the string "False" is parsed as True and makes it look
  # for a non-existent pre-extracted frame directory.
  if ((eval_status != 0)); then
    echo "[vbench] $arm FAILED with exit code $eval_status; last 40 log lines:"
    tail -n 40 "$log_file" >&2 || true
    return "$eval_status"
  fi
  echo "===== $(date -Is) END $arm =====" >>"$log_file"

  if ! has_result "$output_dir"; then
    echo "[vbench] $arm exited successfully but no *_eval_results.json was written" >&2
    return 1
  fi
  echo "[vbench] finished $arm"
}

run_group() {
  local group_name="$1"
  local gpu_ids="$2"
  shift 2
  # One lock per group prevents accidentally starting two copies on the same
  # GPU pair.  Different group scripts use disjoint GPU pairs and can run in
  # parallel.
  mkdir -p "$LOG_ROOT"
  local lock_file="$LOG_ROOT/.vbench_${group_name}.lock"
  exec 9>"$lock_file"
  flock -n 9 || die "$group_name is already running (lock: $lock_file)"

  check_environment
  if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    while (($#)); do
      check_videos "$2"
      shift 2
    done
    echo "[vbench] preflight passed for $group_name (PREFLIGHT_ONLY=1)"
    return 0
  fi
  while (($#)); do
    local arm="$1"
    local videos_dir="$2"
    shift 2
    run_one "$arm" "$videos_dir" "$gpu_ids"
  done
}
