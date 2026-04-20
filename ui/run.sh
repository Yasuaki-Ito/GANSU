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
PORT=8000
TEMPLATE
    echo ".env was not found. A template has been created."
    echo "Edit .env to set GANSU_PATH, etc., then run again."
    exit 1
fi

# Load .env
set -a
source .env
set +a

PORT="${1:-${PORT:-8000}}"

# Auto-detect libgansu.so
GANSU_PATH="${GANSU_PATH:-$HOME/GANSU}"
if [ -z "$GANSU_LIB" ]; then
    for lib in "$GANSU_PATH/build/libgansu.so" "$GANSU_PATH/build/libgansu.dylib"; do
        if [ -f "$lib" ]; then
            export GANSU_LIB="$lib"
            break
        fi
    done
fi

if [ -z "$GANSU_LIB" ] || [ ! -f "$GANSU_LIB" ]; then
    echo "WARNING: libgansu.so not found."
    echo "Build it: cd $GANSU_PATH/build && make gansu_shared"
    echo "Or set GANSU_LIB in .env"
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

# Add gansu Python module to path
export PYTHONPATH="$GANSU_PATH/python:$PYTHONPATH"

echo "Starting GANSU-UI on 0.0.0.0:$PORT"
echo "  GANSU_PATH: $GANSU_PATH"
echo "  GANSU_LIB:  $GANSU_LIB"
python3 -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port "$PORT"
