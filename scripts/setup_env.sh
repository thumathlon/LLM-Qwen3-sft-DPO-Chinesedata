#!/usr/bin/env bash
# Environment bootstrap for AutoDL / single-node experiments.
# 默认 dry-run，只打印计划；远程执行时需 `source scripts/setup_env.sh --dry-run false`。

set -euo pipefail

DRY_RUN=true
for arg in "$@"; do
  case "$arg" in
    --dry-run=false|--dry-run=False|--dry-run=0|--dry-run-off)
      DRY_RUN=false
      ;;
  esac
done

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data_raw}"
export PROC_ROOT="${PROC_ROOT:-${PROJECT_ROOT}/data_proc}"
export TOKENIZERS_PARALLELISM="false"

if [ "${DRY_RUN}" = "false" ]; then
  mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
           "${DATA_ROOT}" "${PROC_ROOT}"
else
  printf "[setup_env] dry-run 模式：未创建任何目录，仅打印配置。\n"
fi

cat <<EOF
Environment variables configured:
  HF_ENDPOINT=${HF_ENDPOINT}
  HF_HOME=${HF_HOME}
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
  DATA_ROOT=${DATA_ROOT}
  PROC_ROOT=${PROC_ROOT}

Remember: 在远程环境执行 `source scripts/setup_env.sh --dry-run false`
EOF
