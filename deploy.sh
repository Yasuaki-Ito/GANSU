#!/bin/bash
# Deploy GANSU to remote server
# Usage: ./deploy.sh [user@host]

REMOTE="${1:-yasuaki@10.30.82.152}"

rsync -avz \
  --exclude='build' \
  --exclude='benchmark_results' \
  --exclude='ui/.env' \
  --exclude='ui/backend/.venv' \
  --exclude='ui/backend/__pycache__' \
  --exclude='ui/frontend/node_modules' \
  --exclude='.git' \
  --exclude='.claude' \
  --delete \
  ./GANSU/ "${REMOTE}:~/GANSU/"

echo "Deployed to ${REMOTE}:~/GANSU/"
echo "Remote: cd ~/GANSU/build && make -j && cd ../ui && bash run.sh"
