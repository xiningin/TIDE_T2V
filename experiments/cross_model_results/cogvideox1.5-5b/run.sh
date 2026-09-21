#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARM=${1:-}
ACTION=${2:-all}
case "${ARM}" in
  cogvideox15_raw|cogvideox15_ours)
    exec "${HERE}/../../benchmark_eval/scripts/run_generation.sh" "${ARM}" "${ACTION}"
    ;;
  *)
    echo "Usage: $0 {cogvideox15_raw|cogvideox15_ours} [validate|prepare|generate|audit|all]" >&2
    exit 2
    ;;
esac
