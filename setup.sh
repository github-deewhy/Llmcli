#!/usr/bin/env bash
set -e

echo "Setting up LLM CLI..."

# Ensure we are in the project root
cd "$(dirname "$0")"

# Check if pipx is available, otherwise fallback to venv
if command -v pipx &> /dev/null; then
    echo "Using pipx for isolated global installation..."
    pipx install --editable .
    echo "Installation complete. The 'llmcli' command is now available globally."
else
    echo "pipx not found. Falling back to local virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .
    echo "Installation complete. Activate the environment with: source .venv/bin/activate"
    echo "Then run: python -m llmcli"
fi

# Create the config directory and the single configuration file if missing
mkdir -p ~/.llmcli
if [ ! -f ~/.llmcli/config.toml ]; then
    cp config.toml.example ~/.llmcli/config.toml
    chmod 600 ~/.llmcli/config.toml
    echo "Created ~/.llmcli/config.toml (uses a local Ollama by default)"
fi

if [ ! -f ~/.llmcli/AGENTS.md ]; then
    cp AGENTS.md.example ~/.llmcli/AGENTS.md
    echo "Created default ~/.llmcli/AGENTS.md"
fi

echo "Setup complete!"
