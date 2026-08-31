#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 SNAPSHOT_ID" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
snapshot_id="$1"
snapshot_root="${repo_root}/evidence/snapshots/${snapshot_id}"

if [[ -e "${snapshot_root}" ]]; then
  echo "snapshot already exists: ${snapshot_root}" >&2
  exit 1
fi

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
  echo "refusing to snapshot a dirty worktree" >&2
  exit 1
fi

"${repo_root}/.venv-mujoco/bin/python" \
  "${repo_root}/scripts/render_reacher_replication_artifacts.py"

required_paths=(
  "outputs/physics_residual_v12_1_recovery"
  "outputs/physics_residual_v12_refinement"
  "outputs/physics_residual_v12_2_scoped_confirmatory"
  "outputs/physics_residual_v12_3_factorial_ablation"
  "outputs/reacher_replication/belief_development"
  "outputs/reacher_replication/confirmatory"
  "outputs/reacher_replication/margin_extension"
  "outputs/reacher_replication/low_level/SELECTION.json"
  "outputs/reacher_replication/low_level/manifest.json"
  "outputs/reacher_replication/low_level/reacher-static-task-seed5100-steps2000k"
  "outputs/reacher_replication/monolithic_baseline/SELECTION.json"
  "outputs/reacher_replication/monolithic_baseline/reacher-monolithic-lifetime-seed25100-decisions100k"
  "paper_artifacts/physics_residual_v12_3"
  "paper_artifacts/reacher_replication"
  "configs/reacher_cross_task_replication_v1.json"
  "configs/reacher_cross_task_replication_v1.sha256"
  "configs/reacher_cross_task_stage2_v1.json"
  "configs/reacher_cross_task_stage2_v1.sha256"
  "configs/reacher_margin_extension_v1.json"
  "configs/reacher_margin_extension_v1.sha256"
  "docs/PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md"
  "docs/REACHER_REPLICATION_FINAL_RESULTS.md"
  "docs/INTEGRATED_PUSHER_REACHER_AUDIT.md"
  "docs/TMLR_CHECKLIST.md"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${repo_root}/${path}" ]]; then
    echo "missing required evidence path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${snapshot_root}/artifacts" "${snapshot_root}/manifests"
for path in "${required_paths[@]}"; do
  destination="${snapshot_root}/artifacts/$(dirname "${path}")"
  mkdir -p "${destination}"
  cp -a "${repo_root}/${path}" "${destination}/"
done

# Publishable evidence must not expose workstation identifiers. Runtime logs
# and TensorBoard event files are redundant with the sealed result tables.
find "${snapshot_root}/artifacts" -type f \
  \( -name 'events.out.tfevents.*' -o -name '*.log' \) -delete

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

source_files=(
  "scripts/audit_repository_privacy.py"
  "scripts/reproduce_clean_checkout.py"
  "scripts/render_physics_residual_v12_3_artifacts.py"
  "scripts/render_reacher_replication_artifacts.py"
  "scripts/run_reacher_low_level_replication.py"
  "scripts/run_reacher_belief_development.py"
  "scripts/run_reacher_monolithic_baseline.py"
  "scripts/run_reacher_confirmatory.py"
  "scripts/run_reacher_margin_extension.py"
  "scripts/sanitize_sb3_archive.py"
  "scripts/seal_privacy_redaction.py"
)
mkdir -p "${snapshot_root}/artifacts/scripts"
for path in "${source_files[@]}"; do
  cp -a "${repo_root}/${path}" "${snapshot_root}/artifacts/scripts/"
done

{
  echo "snapshot_id=${snapshot_id}"
  echo "created_at=$(date -Iseconds)"
  echo "source_commit=$(git -C "${repo_root}" rev-parse HEAD)"
  echo "source_branch=$(git -C "${repo_root}" branch --show-current)"
  echo "python=$(${repo_root}/.venv-mujoco/bin/python --version 2>&1)"
  echo "kernel=$(uname -srm)"
} > "${snapshot_root}/manifests/SNAPSHOT_METADATA.txt"

git -C "${repo_root}" log -1 --format='commit %H%nsubject %s' \
  > "${snapshot_root}/manifests/SOURCE_COMMIT.txt"
"${repo_root}/.venv-mujoco/bin/python" -m pip freeze \
  | sed -E 's#(git\+ssh://git@github.com/)[^/]+/#\1REPOSITORY_OWNER/#' \
  > "${snapshot_root}/manifests/PYTHON_PACKAGES.txt"
lscpu | sed -n -E \
  '/^(Architecture|CPU\(s\)|Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)):/p' \
  > "${snapshot_root}/manifests/CPU.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader > "${snapshot_root}/manifests/GPU.txt" 2>&1; then
    echo "GPU query failed" > "${snapshot_root}/manifests/GPU.txt"
  fi
else
  echo "nvidia-smi unavailable" > "${snapshot_root}/manifests/GPU.txt"
fi

(
  cd "${snapshot_root}"
  find artifacts -type f -print0 | sort -z | xargs -0 sha256sum \
    > manifests/ARTIFACTS.sha256
)

for protocol_dir in \
  "outputs/physics_residual_v12_2_scoped_confirmatory" \
  "outputs/physics_residual_v12_3_factorial_ablation" \
  "outputs/reacher_replication/confirmatory"; do
  protocol_root="${snapshot_root}/artifacts/${protocol_dir}"
  expected="$(tr -d '[:space:]' < "${protocol_root}/FROZEN_PROTOCOL.sha256")"
  actual="$(sha256sum "${protocol_root}/FROZEN_PROTOCOL.json" | cut -d ' ' -f 1)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "protocol hash mismatch: ${protocol_dir}" >&2
    exit 1
  fi
done

extension_root="${snapshot_root}/artifacts/outputs/reacher_replication/margin_extension"
expected="$(tr -d '[:space:]' < "${extension_root}/FROZEN_FRESH_PROTOCOL.sha256")"
actual="$(sha256sum "${extension_root}/FROZEN_FRESH_PROTOCOL.json" | cut -d ' ' -f 1)"
if [[ "${actual}" != "${expected}" ]]; then
  echo "extension protocol hash mismatch" >&2
  exit 1
fi

(
  cd "${snapshot_root}"
  sha256sum manifests/ARTIFACTS.sha256 manifests/CPU.txt manifests/GPU.txt \
    manifests/PYTHON_PACKAGES.txt manifests/SNAPSHOT_METADATA.txt \
    manifests/SOURCE_COMMIT.txt > SNAPSHOT_ROOT.sha256
)

echo "created integrated evidence snapshot: ${snapshot_root}"
