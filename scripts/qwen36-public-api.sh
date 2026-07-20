#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ENV=${RUNTIME_ENV:-/workspace/qwen36-deploy.env}
[[ -r ${RUNTIME_ENV} ]] || { echo "Missing runtime config: ${RUNTIME_ENV}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${RUNTIME_ENV}"

WORKSPACE=${WORKSPACE:-/workspace}
SGLANG_VENV=${SGLANG_VENV:-${WORKSPACE}/venvs/sglang}
API_KEY_FILE=${API_KEY_FILE:-${WORKSPACE}/.qwen36_api_key}
PUBLIC_API_HOST=${PUBLIC_API_HOST:-0.0.0.0}
PUBLIC_API_PORT=${PUBLIC_API_PORT:-8000}
PROXY_KEEP_ALIVE_SECONDS=${PROXY_KEEP_ALIVE_SECONDS:-60}

export TIMING_PROXY_HOST=${PUBLIC_API_HOST}
export TIMING_PROXY_PORT=${PUBLIC_API_PORT}
export TIMING_PROXY_UPSTREAM=http://127.0.0.1:30000
export TIMING_PROXY_API_KEY_FILE=${API_KEY_FILE}
export TIMING_PROXY_LOG=${WORKSPACE}/logs/qwen36-public-timing.jsonl
export TIMING_PROXY_KEEP_ALIVE=${PROXY_KEEP_ALIVE_SECONDS}

exec "${SGLANG_VENV}/bin/python" /opt/supervisor-scripts/qwen36-timing-proxy.py
