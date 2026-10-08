#!/usr/bin/env bash
# One-time setup: Python venv + Node/npm for the Electron frontend.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Personal-Assistant setup"

# ---------------------------------------------------------------- Python
PY_BIN="${PYTHON:-python3}"
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PY_BIN not found. Install Python 3.10+ first." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "==> Creating virtual environment (.venv)"
  "$PY_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> Upgrading pip"
python -m pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "==> Installing Python dependencies (requirements.txt)"
  python -m pip install -r requirements.txt
else
  echo "WARNING: requirements.txt not found at repo root; skipping pip install." >&2
fi

# ---------------------------------------------------------------- Node
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ ! -s "$NVM_DIR/nvm.sh" ]; then
  echo "==> nvm not found; installing to $NVM_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  else
    echo "ERROR: need curl or wget to install nvm." >&2
    exit 1
  fi
fi
# shellcheck disable=SC1091
export NVM_DIR
source "$NVM_DIR/nvm.sh"

# Pick Node version: .nvmrc > default LTS
if [ -f .nvmrc ]; then
  echo "==> Installing Node from .nvmrc: $(cat .nvmrc)"
  nvm install "$(cat .nvmrc)"
else
  echo "==> Installing Node LTS"
  nvm install --lts
fi
nvm use --silent

# Frontend deps: workspaces-root aware
if grep -Eq '"workspaces"[[:space:]]*:' package.json 2>/dev/null; then
  echo "==> Root package.json declares workspaces; running npm install at root"
  npm install
else
  echo "==> Installing frontend dependencies in electron-gui/"
  ( cd electron-gui && npm install )
fi

echo
echo "==> Setup complete. Start the app with:  ./start.sh"
