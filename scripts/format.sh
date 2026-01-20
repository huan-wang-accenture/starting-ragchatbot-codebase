#!/bin/bash
# Format all Python files with black

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Running black formatter..."
uv run black backend/ main.py

echo "Formatting complete!"
