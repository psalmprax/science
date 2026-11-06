#!/usr/bin/env bash
# APEDP-X Operational Storage Footprint & Lifecycle Pruning Runner
#
# Usage:
#   bash scripts/lifecycle_prune.sh          # Print status audit report
#   bash scripts/lifecycle_prune.sh --guard  # Check headroom and auto-prune if breached
#   bash scripts/lifecycle_prune.sh --prune  # Clean bloated build/package caches
#   bash scripts/lifecycle_prune.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

exec "$PYTHON" -m apedp_dyn.operational.lifecycle_pruner "$@"
