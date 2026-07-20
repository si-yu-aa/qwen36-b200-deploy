#!/usr/bin/env bash
set -Eeuo pipefail

RUNTIME_ENV=${RUNTIME_ENV:-/workspace/qwen36-deploy.env}
[[ -r ${RUNTIME_ENV} ]] || { echo "Missing runtime config: ${RUNTIME_ENV}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${RUNTIME_ENV}"

WORKSPACE=${WORKSPACE:-/workspace}
SGLANG_SRC=${SGLANG_SRC:-${WORKSPACE}/src/sglang-nvfp4-dflash}
SGLANG_VENV=${SGLANG_VENV:-${WORKSPACE}/venvs/sglang}
MAIN_MODEL_DIR=${MAIN_MODEL_DIR:-${WORKSPACE}/models/Qwen3.6-35B-A3B-NVFP4}
DRAFT_MODEL_DIR=${DRAFT_MODEL_DIR:-${WORKSPACE}/models/Qwen3.6-35B-A3B-DFlash}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B}
DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}

[[ -x ${SGLANG_VENV}/bin/python ]] || { echo "Missing Python environment" >&2; exit 1; }

export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export PYTHONPATH="${SGLANG_SRC}/python${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

args=(
  -m sglang.launch_server
  --model-path "${MAIN_MODEL_DIR}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --fp4-gemm-backend flashinfer_trtllm
  --moe-runner-backend flashinfer_trtllm
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "${DRAFT_MODEL_DIR}"
  --speculative-draft-model-quantization unquant
  --speculative-dflash-block-size "${DFLASH_BLOCK_SIZE}"
  --speculative-draft-attention-backend fa4
  --attention-backend trtllm_mha
  --linear-attn-prefill-backend flashinfer
  --linear-attn-decode-backend flashinfer
  --mamba-radix-cache-strategy extra_buffer
  --mamba-ssm-dtype bfloat16
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --tp-size 1
  --context-length 262144
  --max-running-requests 32
  --cuda-graph-max-bs-decode 32
  --cuda-graph-backend-prefill tc_piecewise
  --flashinfer-allreduce-fusion-backend auto
  --mem-fraction-static 0.60
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --enable-metrics
  --enable-request-time-stats-logging
  --host 127.0.0.1
  --port 30000
)

cd "${WORKSPACE}"
if [[ -r /opt/supervisor-scripts/utils/logging.sh ]]; then
  # Vast's pty helper keeps long-running server output line-buffered in Supervisor.
  # shellcheck disable=SC1091
  source /opt/supervisor-scripts/utils/logging.sh ""
  pty "${SGLANG_VENV}/bin/python" "${args[@]}" 2>&1
else
  exec "${SGLANG_VENV}/bin/python" "${args[@]}"
fi
