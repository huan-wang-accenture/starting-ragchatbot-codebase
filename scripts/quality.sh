#!/bin/bash
# Run all code quality checks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Running Code Quality Checks ==="
echo

echo "1. Checking code formatting with black..."
uv run black --check backend/ main.py
echo "   ✓ Formatting check passed"
echo

echo "2. Running tests with pytest..."
uv run pytest backend/tests/ -v
echo "   ✓ Tests passed"
echo

echo "=== All quality checks passed! ==="
