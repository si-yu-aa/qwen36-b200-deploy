#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ENV=${RUNTIME_ENV:-/workspace/qwen36-deploy.env}
[[ -r ${RUNTIME_ENV} ]] || { echo "Missing runtime config: ${RUNTIME_ENV}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${RUNTIME_ENV}"

WORKSPACE=${WORKSPACE:-/workspace}
SGLANG_VENV=${SGLANG_VENV:-${WORKSPACE}/venvs/sglang}
API_KEY_FILE=${API_KEY_FILE:-${WORKSPACE}/.qwen36_api_key}
PROXY_KEEP_ALIVE_SECONDS=${PROXY_KEEP_ALIVE_SECONDS:-60}

export TIMING_PROXY_HOST=127.0.0.1
export TIMING_PROXY_PORT=30001
export TIMING_PROXY_UPSTREAM=http://127.0.0.1:30000
export TIMING_PROXY_API_KEY_FILE=${API_KEY_FILE}
export TIMING_PROXY_LOG=${WORKSPACE}/logs/qwen36-timing.jsonl
export TIMING_PROXY_KEEP_ALIVE=${PROXY_KEEP_ALIVE_SECONDS}

exec "${SGLANG_VENV}/bin/python" /opt/supervisor-scripts/qwen36-timing-proxy.py
