#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
repro_python="${LIFEPHYBENCH_PYTHON:-python3.11}"
repro_venv="${1:-${repo_root}/.venv-reproduction}"
repro_mpl="/tmp/lifephybench-matplotlib-reproduction"

if [[ -e "${repro_venv}" ]]; then
  echo "refusing to overwrite existing environment: ${repro_venv}" >&2
  exit 1
fi

"${repro_python}" -m venv "${repro_venv}"
env -u PYTHONPATH "${repro_venv}/bin/python" -m pip install --upgrade \
  "pip==26.2.1" "setuptools==84.0.0" "wheel==0.48.0"
env -u PYTHONPATH "${repro_venv}/bin/python" -m pip install \
  -r "${repo_root}/requirements-reproduction.txt"
env -u PYTHONPATH "${repro_venv}/bin/python" -m pip install \
  --no-deps -e "${repo_root}"
env -u PYTHONPATH "${repro_venv}/bin/python" -m pip check
env -u PYTHONPATH MPLCONFIGDIR="${repro_mpl}" \
  "${repro_venv}/bin/python" \
  "${repo_root}/scripts/reproduce_clean_checkout.py" --run-tests \
  --report "${repo_root}/outputs/clean_checkout_reproduction.json"
