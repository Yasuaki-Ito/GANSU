# GANSU-UI Makefile (runs on local machine with npm/node)
#
# Usage:
#   make          Build frontend (npm install + npm run build)
#   make dev      Start Vite dev server (hot reload)
#   make clean    Remove dist/
#
# After `make`, commit frontend/dist/ and deploy to server.

.PHONY: all dev clean

all:
	cd frontend && npm install && npm run build

dev:
	cd frontend && npm run dev

clean:
	rm -rf frontend/dist
