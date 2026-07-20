#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=../versions.env
source "${REPO_ROOT}/versions.env"

fail() { printf '[check-pins] ERROR: %s\n' "$*" >&2; exit 1; }
for command in git curl sha256sum; do
  command -v "${command}" >/dev/null 2>&1 || fail "missing command: ${command}"
done

scratch=$(mktemp -d)
trap 'rm -rf -- "${scratch}"' EXIT
source_dir=${scratch}/sglang
patch_file=${scratch}/sglang-pr-30078.patch

printf '[check-pins] cloning %s branch %s\n' "${SGLANG_REPO}" "${SGLANG_BRANCH}"
git clone --quiet --filter=blob:none --branch "${SGLANG_BRANCH}" "${SGLANG_REPO}" "${source_dir}"
git -C "${source_dir}" checkout --quiet --detach "${SGLANG_COMMIT}"
actual_commit=$(git -C "${source_dir}" rev-parse HEAD)
[[ ${actual_commit} == "${SGLANG_COMMIT}" ]] || fail "unexpected SGLang commit ${actual_commit}"

curl -fsSL "${SGLANG_MODELOPT_PATCH_URL}" -o "${patch_file}"
printf '%s  %s\n' "${SGLANG_MODELOPT_PATCH_SHA256}" "${patch_file}" | sha256sum -c - >/dev/null
grep -q "^From ${SGLANG_MODELOPT_PATCH_COMMIT} " "${patch_file}" || fail \
  "patch does not contain the expected leading commit"
git -C "${source_dir}" apply --check --exclude='test/*' "${patch_file}"
git -C "${source_dir}" apply --exclude='test/*' "${patch_file}"
git -C "${source_dir}" diff --check

check_hf_revision() {
  local repo=$1 revision=$2
  curl -fsS -o /dev/null "https://huggingface.co/api/models/${repo}/revision/${revision}" || fail \
    "Hugging Face revision is unavailable: ${repo}@${revision}"
  printf '[check-pins] verified %s@%s\n' "${repo}" "${revision}"
}
check_hf_revision "${NVIDIA_MAIN_MODEL_REPO}" "${NVIDIA_MAIN_MODEL_REVISION}"
check_hf_revision "${MMANGKAD_MAIN_MODEL_REPO}" "${MMANGKAD_MAIN_MODEL_REVISION}"
check_hf_revision "${DRAFT_MODEL_REPO}" "${DRAFT_MODEL_REVISION}"

printf '[check-pins] PASS: SGLang base, ModelOpt patch, and model revisions are reproducible\n'
