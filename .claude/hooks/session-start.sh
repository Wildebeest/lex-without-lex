#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Fly CLI
if ! command -v fly &> /dev/null; then
  curl -L https://fly.io/install.sh | sh
  echo 'export PATH="$HOME/.fly/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
  export PATH="$HOME/.fly/bin:$PATH"
fi

# Install Python dev dependencies using uv (faster, avoids setuptools compat issues)
uv pip install -e ".[dev]" --system --quiet
