# GANSU top-level Makefile (for remote GPU server)
#
# Usage:
#   make              Build C++ CLI (gansu) and shared library (libgansu.so)
#   make serve        Start GANSU-UI server (pre-built frontend included)
#   make clean        Clean C++ build
#
# Frontend rebuild (local machine with npm):
#   cd ui/frontend && npm install && npm run build
#   Then commit ui/frontend/dist/
#
# Prerequisites:
#   C++ build:  cmake, CUDA toolkit (or CPU-only mode)
#   UI serve:   python3 (fastapi, uvicorn auto-installed by run.sh)

.PHONY: all gansu serve clean

all: gansu

# --- C++ build ---
gansu: build/Makefile
	cd build && $(MAKE) -j gansu_cli gansu_shared HF_main

build/Makefile:
	mkdir -p build && cd build && cmake ..

# --- Start UI server ---
serve:
	cd ui && bash run.sh

# --- Clean ---
clean:
	cd build && $(MAKE) clean 2>/dev/null || true
