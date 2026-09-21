#!/usr/bin/env bash

# Shared local model/cache locations used by the official VBench-2.0 code.
export VBENCH2_CACHE_DIR="${VBENCH2_CACHE_DIR:-${HOME}/.cache/vbench2}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/vbench2/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/vbench2/torch}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${HOME}/.cache/vbench2/modelscope}"
export NO_ALBUMENTATIONS_UPDATE=1
