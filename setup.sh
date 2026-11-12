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

# Create default config directory if it doesn't exist
mkdir -p ~/.llmcli
if [ ! -f ~/.llmcli/.env ]; then
    cp .env.example ~/.llmcli/.env
    echo "Created default ~/.llmcli/.env"
fi

if [ ! -f ~/.llmcli/AGENTS.md ]; then
    cp AGENTS.md.example ~/.llmcli/AGENTS.md
    echo "Created default ~/.llmcli/AGENTS.md"
fi

if [ ! -f ~/.llmcli/profiles.toml ]; then
    cp profiles.toml.example ~/.llmcli/profiles.toml
    echo "Created default ~/.llmcli/profiles.toml (no default profile is active until you set one)"
fi

echo "Setup complete!"
