#!/usr/bin/env bash
# BrokerIQ pre-commit hook.
#
# Runs the fast CI gates locally before every commit so nothing broken lands
# in history (GitHub Actions is currently blocked by a billing error on the
# account — this is the local equivalent).
#
# Gates (fast ones only; full evals + promptfoo run in CI / on demand):
#   1. ruff lint (src, tests, evals)
#   2. pytest unit tests
#
# Install (run once):
#   ln -sf ../../scripts/pre-commit.sh .git/hooks/pre-commit
#   chmod +x scripts/pre-commit.sh
#
# Skip with:  git commit --no-verify

set -euo pipefail

# Resolve the real script path even when invoked via .git/hooks/pre-commit symlink.
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
ROOT="$(cd -P "$(dirname "$SOURCE")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT"

echo "==> pre-commit: ruff"
uv run ruff check src tests evals

echo "==> pre-commit: pytest"
uv run pytest -q

echo "==> pre-commit: OK"

