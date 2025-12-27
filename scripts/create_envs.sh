#!/usr/bin/env bash
set -euo pipefail

# Choose CPU by default for maximum compatibility.
# Set USE_CUDA=1 to use the CUDA env specs.
USE_CUDA="${USE_CUDA:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_DIR="${ROOT}/.conda/env"
DIAR_DIR="${ROOT}/.conda/diar"
ASR_DIR="${ROOT}/.conda/asr"

mkdir -p "${ROOT}/.conda"

echo "==> Creating main env at: ${ENV_DIR}"
conda env create -p "${ENV_DIR}" -f "${ROOT}/environments/env.yml" || \
  conda env update -p "${ENV_DIR}" -f "${ROOT}/environments/env.yml" --prune

if [[ "${USE_CUDA}" == "1" ]]; then
  DIAR_YML="${ROOT}/environments/diar.cuda.yml"
  ASR_YML="${ROOT}/environments/asr.cuda.yml"
else
  DIAR_YML="${ROOT}/environments/diar.cpu.yml"
  ASR_YML="${ROOT}/environments/asr.cpu.yml"
fi

echo "==> Creating diar env at: ${DIAR_DIR}"
conda env create -p "${DIAR_DIR}" -f "${DIAR_YML}" || \
  conda env update -p "${DIAR_DIR}" -f "${DIAR_YML}" --prune

echo "==> Creating asr env at: ${ASR_DIR}"
conda env create -p "${ASR_DIR}" -f "${ASR_YML}" || \
  conda env update -p "${ASR_DIR}" -f "${ASR_YML}" --prune

echo
echo "Done."
echo "Main: conda activate ${ENV_DIR}"
echo "Diar: ${DIAR_DIR}/bin/python diarize_cli.py --help"
echo "ASR : ${ASR_DIR}/bin/python transcribe_cli.py --help"
