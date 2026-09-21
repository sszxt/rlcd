#!/usr/bin/env bash
# Sync this repo to the sameer GPU box. Run from a WSL Ubuntu shell (the SSH
# alias `sameer` and its key only exist there, not in Windows/Git Bash).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${1:-sameer}"

rsync -av --delete \
  --exclude '.venv/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude 'outputs/' \
  --exclude '.uv-cache/' \
  "$REPO_DIR/" "$HOST:~/rlcd/"
