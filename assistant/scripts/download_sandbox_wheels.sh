#!/usr/bin/env bash
# Backend-owned arguments only. Download wheels; never build or install on host.
set -euo pipefail
PYTHON_BIN="$1"
DESTINATION="$2"
shift 2
exec "$PYTHON_BIN" -I -m pip --isolated download \
  --disable-pip-version-check --no-cache-dir --timeout 30 --retries 1 \
  --index-url https://pypi.org/simple --only-binary=:all: \
  --dest "$DESTINATION" "$@"
