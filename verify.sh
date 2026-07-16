#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=versions.env
source "${REPO_ROOT}/versions.env"
WORKSPACE=${WORKSPACE:-/workspace}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKSPACE}/qwen36-deploy.env}
[[ -r ${RUNTIME_ENV} ]] && source "${RUNTIME_ENV}"
API_KEY_FILE=${API_KEY_FILE:-${WORKSPACE}/.qwen36_api_key}
SGLANG_VENV=${SGLANG_VENV:-${WORKSPACE}/venvs/sglang}
DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-8}

fail() { printf '[verify] ERROR: %s\n' "$*" >&2; exit 1; }
[[ -s ${API_KEY_FILE} ]] || fail "missing API key file"
api_key=$(<"${API_KEY_FILE}")

supervisorctl status qwen36-nvfp4 | grep -q RUNNING || fail "model service is not RUNNING"
supervisorctl status qwen36-timing-proxy | grep -q RUNNING || fail "timing proxy is not RUNNING"

pid=$(pgrep -f 'sglang.launch_server' | head -n 1)
[[ -n ${pid} ]] || fail "cannot find sglang.launch_server"
mapfile -d '' argv < "/proc/${pid}/cmdline"
has_arg_pair() {
  local key=$1 value=$2 index
  for ((index=0; index<${#argv[@]}-1; index++)); do
    [[ ${argv[index]} == "${key}" && ${argv[index+1]} == "${value}" ]] && return 0
  done
  return 1
}
has_arg_pair --speculative-algorithm DFLASH || fail "DFlash is not enabled"
has_arg_pair --speculative-dflash-block-size "${DFLASH_BLOCK_SIZE}" || fail "unexpected DFlash block size"
has_arg_pair --attention-backend trtllm_mha || fail "unexpected attention backend"
has_arg_pair --speculative-draft-attention-backend fa4 || fail "FA4 draft attention is not enabled"
has_arg_pair --fp4-gemm-backend flashinfer_trtllm || fail "unexpected FP4 GEMM backend"
has_arg_pair --moe-runner-backend flashinfer_trtllm || fail "unexpected MoE backend"

curl -fsS -H "Authorization: Bearer ${api_key}" http://127.0.0.1:30000/v1/models >/dev/null
curl -fsS -H "Authorization: Bearer ${api_key}" http://127.0.0.1:30001/_timing/health >/dev/null

response=$(curl -fsS --max-time 120 \
  -H "Authorization: Bearer ${api_key}" \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:30000/v1/chat/completions \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK only.\"}],\"temperature\":0,\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
RESPONSE_JSON=${response} "${SGLANG_VENV}/bin/python" - <<'PY'
import json
import os

payload = json.loads(os.environ["RESPONSE_JSON"])
choices = payload.get("choices") or []
if not choices:
    raise SystemExit(f"missing choices: {payload}")
print("[verify] smoke response:", choices[0]["message"].get("content", "").strip())
PY
printf '[verify] PASS: NVFP4 + DFlash B%s is serving on 127.0.0.1:30000\n' "${DFLASH_BLOCK_SIZE}"
