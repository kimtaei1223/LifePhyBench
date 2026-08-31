#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 SNAPSHOT_ID" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
snapshot_id="$1"
snapshot_root="${repo_root}/evidence/snapshots/${snapshot_id}"
source_git_status="$(git -C "${repo_root}" status --short --branch)"
source_git_diff="$(git -C "${repo_root}" diff --binary)"

if [[ -e "${snapshot_root}" ]]; then
  echo "snapshot already exists: ${snapshot_root}" >&2
  exit 1
fi

source_paths=(
  "outputs/physics_residual_v12_1_recovery"
  "outputs/physics_residual_v12_refinement"
  "outputs/physics_residual_v12_2_scoped_confirmatory"
  "outputs/physics_residual_v12_3_factorial_ablation"
)

for source_path in "${source_paths[@]}"; do
  if [[ ! -d "${repo_root}/${source_path}" ]]; then
    echo "missing required evidence directory: ${source_path}" >&2
    exit 1
  fi
done

mkdir -p "${snapshot_root}/artifacts/outputs" "${snapshot_root}/manifests"

for source_path in "${source_paths[@]}"; do
  cp -a \
    "${repo_root}/${source_path}" \
    "${snapshot_root}/artifacts/outputs/"
done

# Raw TensorBoard event names embed the training host. They are not required
# for the sealed result tables and are excluded from publishable snapshots.
find "${snapshot_root}/artifacts" -type f \
  -name 'events.out.tfevents.*' -delete

# Make copied text artifacts portable and remove workstation paths. Stable-
# Baselines archives store the TensorBoard path inside their `data` member, so
# sanitize those archives as well.
while IFS= read -r -d '' artifact; do
  if grep -Iq . "${artifact}"; then
    sed -i "s#${repo_root}#\${PROJECT_ROOT}#g" "${artifact}"
  fi
done < <(find "${snapshot_root}/artifacts" -type f -print0)
while IFS= read -r -d '' archive; do
  "${repo_root}/.venv-mujoco/bin/python" \
    "${repo_root}/scripts/sanitize_sb3_archive.py" \
    --archive "${archive}" --project-root "${repo_root}"
done < <(find "${snapshot_root}/artifacts" -type f -name '*.zip' -print0)
while IFS= read -r -d '' protocol; do
  "${repo_root}/.venv-mujoco/bin/python" \
    "${repo_root}/scripts/seal_privacy_redaction.py" --protocol "${protocol}"
done < <(find "${snapshot_root}/artifacts" -type f \
  \( -name 'FROZEN_PROTOCOL.json' -o -name 'FROZEN_FRESH_PROTOCOL.json' \) \
  -print0)

{
  echo "snapshot_id=${snapshot_id}"
  echo "created_at=$(date -Iseconds)"
  echo "source_commit=$(git -C "${repo_root}" rev-parse HEAD)"
  echo "source_branch=$(git -C "${repo_root}" branch --show-current)"
  echo "python=$(${repo_root}/.venv-mujoco/bin/python --version 2>&1)"
  echo "kernel=$(uname -srmo)"
} > "${snapshot_root}/manifests/SNAPSHOT_METADATA.txt"

printf '%s\n' "${source_git_status}" \
  > "${snapshot_root}/manifests/GIT_STATUS.txt"
git -C "${repo_root}" log -1 --format='commit %H%nsubject %s' \
  > "${snapshot_root}/manifests/SOURCE_COMMIT.txt"
printf '%s' "${source_git_diff}" \
  > "${snapshot_root}/manifests/UNCOMMITTED.patch"

"${repo_root}/.venv-mujoco/bin/python" -m pip freeze \
  | sed -E 's#(git\+ssh://git@github.com/)[^/]+/#\1REPOSITORY_OWNER/#' \
  > "${snapshot_root}/manifests/PYTHON_PACKAGES.txt"
lscpu > "${snapshot_root}/manifests/CPU.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader \
    > "${snapshot_root}/manifests/GPU.txt" 2>&1; then
    echo "GPU query failed in the snapshot environment." \
      >> "${snapshot_root}/manifests/GPU.txt"
  fi
else
  echo "nvidia-smi unavailable" > "${snapshot_root}/manifests/GPU.txt"
fi

(
  cd "${snapshot_root}"
  find artifacts -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > manifests/ARTIFACTS.sha256
)

checkpoint="${snapshot_root}/artifacts/outputs/physics_residual_v12_refinement/residual_model.pt"
expected_checkpoint_sha="c38734683b1aeebebd709a543354b14bc7bc12c440a3ef6ee0622ab882ea07f2"
actual_checkpoint_sha="$(sha256sum "${checkpoint}" | cut -d ' ' -f 1)"

if [[ "${actual_checkpoint_sha}" != "${expected_checkpoint_sha}" ]]; then
  echo "checkpoint hash mismatch" >&2
  exit 1
fi

for protocol_dir in \
  "physics_residual_v12_2_scoped_confirmatory" \
  "physics_residual_v12_3_factorial_ablation"; do
  protocol_root="${snapshot_root}/artifacts/outputs/${protocol_dir}"
  expected_protocol_sha="$(tr -d '[:space:]' < "${protocol_root}/FROZEN_PROTOCOL.sha256")"
  actual_protocol_sha="$(sha256sum "${protocol_root}/FROZEN_PROTOCOL.json" | cut -d ' ' -f 1)"
  if [[ "${actual_protocol_sha}" != "${expected_protocol_sha}" ]]; then
    echo "protocol hash mismatch: ${protocol_dir}" >&2
    exit 1
  fi
done

(
  cd "${snapshot_root}"
  sha256sum manifests/ARTIFACTS.sha256 \
    manifests/CPU.txt \
    manifests/GIT_STATUS.txt \
    manifests/GPU.txt \
    manifests/PYTHON_PACKAGES.txt \
    manifests/SNAPSHOT_METADATA.txt \
    manifests/SOURCE_COMMIT.txt \
    manifests/UNCOMMITTED.patch \
    > SNAPSHOT_ROOT.sha256
)

echo "created evidence snapshot: ${snapshot_root}"
