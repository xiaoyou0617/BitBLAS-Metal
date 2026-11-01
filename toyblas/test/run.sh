#!/usr/bin/env bash
# Usage: bash toyblas/test/run.sh [smoke|stress|all] [--device cpu|mps|all]
# Run this with the project's virtual environment activated.
set -euo pipefail
TOYBLAS_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TOYBLAS_TEST_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
exec "${TOYBLAS_TEST_PYTHON:-python}" -B -u -m toyblas.test "$@"
