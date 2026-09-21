#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARM=${1:-}
ACTION=${2:-all}
case "${ARM}" in
  sana_raw|sana_ours)
    exec "${HERE}/../../benchmark_eval/scripts/run_generation.sh" "${ARM}" "${ACTION}"
    ;;
  *)
    echo "Usage: $0 {sana_raw|sana_ours} [validate|prepare|generate|audit|all]" >&2
    exit 2
    ;;
esac
