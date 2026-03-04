#!/bin/bash
# Start GANSU-UI backend server

set -e
cd "$(dirname "$0")"

# If .env does not exist, create template and exit
if [ ! -f .env ]; then
    cat > .env <<'TEMPLATE'
# GANSU-UI configuration
# Edit this file then run ./run.sh again.

GANSU_PATH=$HOME/GANSU
HF_MAIN_PATH=$HOME/GANSU/build/HF_main
PORT=8000
TEMPLATE
    echo ".env was not found. A template has been created."
    echo "Edit .env to set GANSU_PATH, HF_MAIN_PATH, etc., then run again."
    exit 1
fi

# Load .env
set -a
source .env
set +a

PORT="${1:-${PORT:-8000}}"

# Check HF_main exists
HF_MAIN_PATH="${HF_MAIN_PATH:-${GANSU_PATH:-$HOME/GANSU}/build/HF_main}"
if [ ! -f "$HF_MAIN_PATH" ]; then
    echo "WARNING: HF_main not found at $HF_MAIN_PATH"
    echo "Set GANSU_PATH or HF_MAIN_PATH in .env"
fi

# Setup venv
VENV_DIR="backend/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r backend/requirements.txt
fi

echo "Starting GANSU-UI backend on 0.0.0.0:$PORT"
python3 -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port "$PORT"
