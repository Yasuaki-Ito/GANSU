#!/bin/bash
# Start GANSU-UI backend server

set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    cat > .env <<'TEMPLATE'
# GANSU-UI configuration
GANSU_PATH=$HOME/GANSU
PORT=8000
TEMPLATE
    echo ".env created. Edit if needed, then run again."
    exit 1
fi

set -a; source .env; set +a
PORT="${1:-${PORT:-8000}}"
GANSU_PATH="${GANSU_PATH:-$HOME/GANSU}"

# Find gansu binary
GANSU_BIN="${GANSU_BIN:-$GANSU_PATH/build/gansu}"
if [ ! -f "$GANSU_BIN" ]; then
    GANSU_BIN="$GANSU_PATH/build/HF_main"
fi
export GANSU_BIN GANSU_PATH

if [ ! -f "$GANSU_BIN" ]; then
    echo "WARNING: gansu binary not found at $GANSU_BIN"
    echo "Build it: cd $GANSU_PATH/build && make"
fi

# Setup venv
VENV_DIR="backend/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -q -r backend/requirements.txt
fi

echo "Starting GANSU-UI on 0.0.0.0:$PORT"
echo "  GANSU_BIN: $GANSU_BIN"
python3 -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port "$PORT"
