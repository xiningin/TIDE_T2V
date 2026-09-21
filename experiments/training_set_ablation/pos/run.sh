#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${HERE}/../../benchmark_eval/scripts/run_generation.sh" sana_ours_wo_neg "${1:-all}"
