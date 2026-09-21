#!/usr/bin/env bash
# Pull outputs/ (adapters, plots, eval json) back from sameer to this repo.
# Run from a WSL Ubuntu shell.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${1:-sameer}"

mkdir -p "$REPO_DIR/outputs"
rsync -av "$HOST:~/rlcd/outputs/" "$REPO_DIR/outputs/"
