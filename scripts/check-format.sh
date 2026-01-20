#!/bin/bash
# Check formatting without making changes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Checking code formatting with black..."
uv run black --check backend/ main.py

echo "All files are properly formatted!"
