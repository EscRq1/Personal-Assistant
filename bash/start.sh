#!/usr/bin/env bash
# Launch backend (FastAPI) + frontend (Electron). Ctrl+C / kill stops both.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

API_PORT="${API_PORT:-5000}"
API_HOST="127.0.0.1"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
BACKEND_DIR="assistant/src"
FRONTEND_DIR="electron-gui"

log()  { printf '\033[1;36m[start]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[start]\033[0m %s\n' "$*" >&2; }

# Optional env passthrough
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES
[ -n "${WEB_FETCH_CA_BUNDLE:-}" ]  && export WEB_FETCH_CA_BUNDLE

# ---------------------------------------------------------------- preflight
fail() { err "$1"; err "Run ./setup.sh first."; exit 1; }

[ -f "$BACKEND_DIR/main.py" ] || fail "Missing $BACKEND_DIR/main.py"
[ -f "$FRONTEND_DIR/package.json" ] || fail "Missing $FRONTEND_DIR/package.json"
[ -f ".venv/bin/activate" ] || fail "Missing .venv (run ./setup.sh)"
[ -s "$NVM_DIR/nvm.sh" ] || fail "Missing nvm (run ./setup.sh)"

# ---------------------------------------------------------------- helpers
port_up() {
  if command -v nc >/dev/null 2>&1; then
    nc -z "$API_HOST" "$API_PORT" >/dev/null 2>&1
  else
    ( exec 3<>"/dev/tcp/$API_HOST/$API_PORT" ) >/dev/null 2>&1
  fi
}

BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  log "Shutting down..."
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
  [ -n "$BACKEND_PID" ]  && kill "$BACKEND_PID"  2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- backend
# shellcheck disable=SC1091
source .venv/bin/activate
log "Starting backend (FastAPI) on http://$API_HOST:$API_PORT"
python "$BACKEND_DIR/main.py" &
BACKEND_PID=$!

log "Waiting for backend to accept connections on port $API_PORT (max 180s)..."
for _ in $(seq 1 180); do
  if port_up; then log "Backend is up."; break; fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    err "Backend process exited early."
    wait "$BACKEND_PID"
    exit 1
  fi
  sleep 1
done
port_up || { err "Backend did not come up in 180s."; exit 1; }

# ---------------------------------------------------------------- frontend
# shellcheck disable=SC1091
export NVM_DIR
source "$NVM_DIR/nvm.sh"
nvm use --silent >/dev/null 2>&1 || true

log "Starting frontend (Electron) in $FRONTEND_DIR/"
( cd "$FRONTEND_DIR" && exec npm start ) &
FRONTEND_PID=$!

log "Running. Press Ctrl+C to stop both."
# Wait on the frontend; when it exits, cleanup tears down the backend too.
wait "$FRONTEND_PID"
