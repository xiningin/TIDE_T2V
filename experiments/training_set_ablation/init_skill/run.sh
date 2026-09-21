#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${HERE}/../../benchmark_eval/scripts/run_generation.sh" sana_init_skill "${1:-all}"
