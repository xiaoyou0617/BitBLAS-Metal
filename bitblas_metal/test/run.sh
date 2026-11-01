#!/usr/bin/env bash
# Usage: bash bitblas_metal/test/run.sh [smoke|stress|all] [--device cpu|mps|all]
# Run this with the project's virtual environment activated.
set -euo pipefail
BITBLAS_METAL_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BITBLAS_METAL_TEST_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
exec "${BITBLAS_METAL_TEST_PYTHON:-python}" -B -u -m bitblas_metal.test "$@"
