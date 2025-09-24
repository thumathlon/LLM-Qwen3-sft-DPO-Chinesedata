#!/usr/bin/env bash
# Minimal environment setup for AutoDL/single-node training.
# Goal: be safe, side‑effect free by default, and avoid breaking shells.

set -e

# Defaults: no mirror, no directory creation. Use flags to opt‑in.
DRY_RUN=true
USE_HF_MIRROR=false
INIT_DIRS=false

for arg in "$@"; do
  case "$arg" in
    --dry-run=false|--no-dry-run|--dry-run=0)
      DRY_RUN=false ;;
    --use-hf-mirror)
      USE_HF_MIRROR=true ;;
    --init-dirs)
      INIT_DIRS=true ;;
  esac
done

# Be robust across shells: if BASH_SOURCE is unavailable, fall back to PWD.
if [ -n "${BASH_SOURCE:-}" ]; then
  PROJECT_ROOT_DEFAULT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd 2>/dev/null || pwd)
else
  PROJECT_ROOT_DEFAULT="${PWD}"
fi
export PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

# Minimal variables; do not override if user already set them.
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_raw}"
export PROC_ROOT="${PROC_ROOT:-${PROJECT_ROOT}/data_proc}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Hugging Face mirror is optional to avoid network issues on some hosts.
if [ "${USE_HF_MIRROR}" = "true" ] && [ -z "${HF_ENDPOINT:-}" ]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi

# Keep user caches unless explicitly provided from outside.
if [ -n "${HF_HOME:-}" ]; then
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
fi

if [ "${INIT_DIRS}" = "true" ] && [ "${DRY_RUN}" = "false" ]; then
  mkdir -p "${DATA_ROOT}" "${PROC_ROOT}"
  # Only create HF caches if HF_HOME is explicitly set to a project path.
  if [ -n "${HF_HOME:-}" ]; then
    mkdir -p "${HF_DATASETS_CACHE:-}" "${HUGGINGFACE_HUB_CACHE:-}" "${TRANSFORMERS_CACHE:-}"
  fi
else
  printf "[setup_env] dry-run or minimal mode: no directories created.\n" >&2
fi

printf "Environment configured:\n"
printf "  PROJECT_ROOT=%s\n" "${PROJECT_ROOT}"
printf "  DATA_ROOT=%s\n" "${DATA_ROOT}"
printf "  PROC_ROOT=%s\n" "${PROC_ROOT}"
if [ -n "${HF_ENDPOINT:-}" ]; then
  printf "  HF_ENDPOINT=%s\n" "${HF_ENDPOINT}"
fi
if [ -n "${HF_HOME:-}" ]; then
  printf "  HF_HOME=%s\n" "${HF_HOME}"
fi
printf "  TOKENIZERS_PARALLELISM=%s\n" "${TOKENIZERS_PARALLELISM}"

printf "Hints:\n"
printf "  - Use --use-hf-mirror if your network benefits from HF mirror.\n"
printf "  - Use --init-dirs to create %s and %s.\n" "${DATA_ROOT}" "${PROC_ROOT}"
