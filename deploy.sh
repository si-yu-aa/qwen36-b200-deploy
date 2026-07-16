#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=versions.env
source "${REPO_ROOT}/versions.env"

WORKSPACE=${WORKSPACE:-/workspace}
SRC_DIR=${SRC_DIR:-${WORKSPACE}/src/sglang-nvfp4-dflash}
VENV_DIR=${VENV_DIR:-${WORKSPACE}/venvs/sglang}
MAIN_MODEL_DIR=${MAIN_MODEL_DIR:-${WORKSPACE}/models/Qwen3.6-35B-A3B-NVFP4}
DRAFT_MODEL_DIR=${DRAFT_MODEL_DIR:-${WORKSPACE}/models/Qwen3.6-35B-A3B-DFlash}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKSPACE}/qwen36-deploy.env}
API_KEY_FILE=${API_KEY_FILE:-${WORKSPACE}/.qwen36_api_key}
DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-8}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-1800}
SKIP_MODEL_DOWNLOAD=${SKIP_MODEL_DOWNLOAD:-0}
SKIP_PYTHON_INSTALL=${SKIP_PYTHON_INSTALL:-0}
ALLOW_NON_B200=${ALLOW_NON_B200:-0}

log() { printf '[deploy] %s\n' "$*"; }
die() { printf '[deploy] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

[[ ${EUID} -eq 0 ]] || die "run this script as root inside the Vast instance"
[[ -r /etc/vast-agents-guide.md ]] || die "this does not look like a Vast.ai base-image instance"
[[ ${DFLASH_BLOCK_SIZE} =~ ^[0-9]+$ ]] || die "DFLASH_BLOCK_SIZE must be an integer"

for command in git uv nvidia-smi supervisorctl curl; do
  need "${command}"
done

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
log "GPU: ${gpu_name}; VRAM: ${gpu_memory} MiB"
if [[ ${ALLOW_NON_B200} != 1 && ${gpu_name} != *B200* ]]; then
  die "expected an NVIDIA B200; set ALLOW_NON_B200=1 only if this is intentional"
fi
(( gpu_memory >= 150000 )) || die "at least 150000 MiB VRAM is expected for this TP=1 profile"

mkdir -p "${WORKSPACE}/src" "${WORKSPACE}/venvs" "${WORKSPACE}/models" "${WORKSPACE}/logs"

if [[ -n ${QWEN36_API_KEY:-} ]]; then
  log "installing API key from QWEN36_API_KEY (value will not be printed)"
  umask 077
  printf '%s' "${QWEN36_API_KEY}" > "${API_KEY_FILE}"
elif [[ -s ${API_KEY_FILE} ]]; then
  log "reusing existing API key file ${API_KEY_FILE}"
else
  die "export QWEN36_API_KEY before the first deployment"
fi
chmod 600 "${API_KEY_FILE}"

if [[ ! -d ${SRC_DIR}/.git ]]; then
  log "cloning the pinned SGLang fork"
  git clone --filter=blob:none --branch "${SGLANG_BRANCH}" "${SGLANG_REPO}" "${SRC_DIR}"
else
  log "refreshing existing SGLang source"
  git -C "${SRC_DIR}" remote set-url origin "${SGLANG_REPO}"
  git -C "${SRC_DIR}" fetch origin "${SGLANG_BRANCH}"
fi
git -C "${SRC_DIR}" checkout --detach "${SGLANG_COMMIT}"
actual_commit=$(git -C "${SRC_DIR}" rev-parse HEAD)
[[ ${actual_commit} == "${SGLANG_COMMIT}" ]] || die "SGLang commit mismatch: ${actual_commit}"

if [[ ${SKIP_PYTHON_INSTALL} != 1 ]]; then
  if [[ ! -x ${VENV_DIR}/bin/python ]]; then
    log "creating Python 3.12 environment"
    uv venv --python 3.12 "${VENV_DIR}"
  fi
  log "installing the pinned SGLang wheel and its CUDA 13 dependencies"
  uv pip install --python "${VENV_DIR}/bin/python" "sglang==${SGLANG_PACKAGE_VERSION}"
else
  [[ -x ${VENV_DIR}/bin/python ]] || die "SKIP_PYTHON_INSTALL=1 but ${VENV_DIR}/bin/python is absent"
fi

check_version() {
  local distribution=$1 expected=$2 actual
  actual=$("${VENV_DIR}/bin/python" -c 'import importlib.metadata,sys; print(importlib.metadata.version(sys.argv[1]))' "${distribution}")
  [[ ${actual} == "${expected}" ]] || die "${distribution}=${actual}; expected ${expected}"
  log "verified ${distribution}=${actual}"
}
check_version sglang "${SGLANG_PACKAGE_VERSION}"
check_version torch "${TORCH_VERSION}"
check_version flashinfer-python "${FLASHINFER_VERSION}"
check_version flash-attn-4 "${FLASH_ATTN_4_VERSION}"
check_version sglang-kernel "${SGLANG_KERNEL_VERSION}"

download_model() {
  local repo=$1 revision=$2 destination=$3
  if [[ -s ${destination}/config.json ]]; then
    log "model already present: ${destination}"
    return
  fi
  log "downloading ${repo}@${revision} to ${destination}"
  "${VENV_DIR}/bin/hf" download "${repo}" --revision "${revision}" --local-dir "${destination}"
}

if [[ ${SKIP_MODEL_DOWNLOAD} != 1 ]]; then
  download_model "${MAIN_MODEL_REPO}" "${MAIN_MODEL_REVISION}" "${MAIN_MODEL_DIR}"
  download_model "${DRAFT_MODEL_REPO}" "${DRAFT_MODEL_REVISION}" "${DRAFT_MODEL_DIR}"
else
  [[ -s ${MAIN_MODEL_DIR}/config.json ]] || die "main model is absent while SKIP_MODEL_DOWNLOAD=1"
  [[ -s ${DRAFT_MODEL_DIR}/config.json ]] || die "draft model is absent while SKIP_MODEL_DOWNLOAD=1"
fi

log "writing non-secret runtime configuration"
install -m 0644 /dev/null "${RUNTIME_ENV}"
printf '%s\n' \
  "WORKSPACE=${WORKSPACE}" \
  "SGLANG_SRC=${SRC_DIR}" \
  "SGLANG_VENV=${VENV_DIR}" \
  "MAIN_MODEL_DIR=${MAIN_MODEL_DIR}" \
  "DRAFT_MODEL_DIR=${DRAFT_MODEL_DIR}" \
  "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
  "API_KEY_FILE=${API_KEY_FILE}" \
  "DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE}" \
  > "${RUNTIME_ENV}"

log "installing Supervisor programs"
install -m 0755 "${REPO_ROOT}/scripts/qwen36-nvfp4-sglang.sh" /opt/supervisor-scripts/qwen36-nvfp4-sglang.sh
install -m 0755 "${REPO_ROOT}/scripts/qwen36-timing-proxy.py" /opt/supervisor-scripts/qwen36-timing-proxy.py
install -m 0644 "${REPO_ROOT}/supervisor/qwen36-nvfp4.conf" /etc/supervisor/conf.d/qwen36-nvfp4.conf
install -m 0644 "${REPO_ROOT}/supervisor/qwen36-timing-proxy.conf" /etc/supervisor/conf.d/qwen36-timing-proxy.conf
supervisorctl reread
supervisorctl update
supervisorctl restart qwen36-nvfp4
supervisorctl restart qwen36-timing-proxy

log "waiting up to ${WAIT_TIMEOUT}s for the OpenAI-compatible endpoint"
deadline=$((SECONDS + WAIT_TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 2 -H "Authorization: Bearer $(<"${API_KEY_FILE}")" http://127.0.0.1:30000/v1/models >/dev/null; then
    log "service is ready"
    "${REPO_ROOT}/verify.sh"
    exit 0
  fi
  sleep 5
done
supervisorctl status qwen36-nvfp4 qwen36-timing-proxy || true
die "service did not become ready; inspect Supervisor output and GPU memory"
