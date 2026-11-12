# LLMCLI

A focused coding assistant for local and OpenAI-compatible language models.

## Features

- Interactive REPL with history and Enter-to-send input
- Streaming responses with Rich rendering
- Local Ollama support and OpenAI-compatible endpoints (Gemini, NVIDIA NIM, OpenRouter, LM Studio, ...)
- Named provider profiles: configure several models/services once, switch with `/profile`
- Context file loading from project or home directory
- `@file` attachment resolution for including file contents in prompts
- Coding-focused system prompt, with optional project context
- Interactive agent tools for inspecting files/logs and working across project files
- Diff approval before edits and explicit approval before running tests
- Persistent global and project Markdown memory stored in local files
- Four terminal color presets: `ocean`, `forest`, `sunset`, and `mono`
- Slash commands for help, history, settings, and model selection

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
llmcli
```

Or run directly:

```bash
python -m llmcli
```

## Environment configuration

Copy the example file and edit it as needed:

```bash
cp .env.example ~/.llmcli/.env
```

Supported variables (or use [profiles](#profiles) for several setups):

- `LLMCLI_PROVIDER`: `ollama` or `openai_compat`
- `LLMCLI_MODEL`: default model name
- `LLMCLI_ENDPOINT`: API base URL
- `LLMCLI_API_KEY`: optional API key
- `LLMCLI_CONTEXT_SIZE`: model context size
- `LLMCLI_MAX_TOKENS`: max generation length
- `LLMCLI_CONTEXT_FILE`: project context file name
- `LLMCLI_TEMPERATURE`, `LLMCLI_TOP_P`: optional sampling overrides
- `LLMCLI_EXTRA_BODY`: optional JSON object merged into OpenAI-compatible requests
- `LLMCLI_PROFILE`: name of the profile to activate (see below)
- `LLMCLI_THEME`: CLI color preset (`ocean`, `forest`, `sunset`, or `mono`)

Interactive agent workflows require a model/provider that supports structured
tool calls. One-shot `-p` prompts remain text-only.

## Profiles

Define each provider setup once in `~/.llmcli/profiles.toml` (`setup.sh` installs a
starter from `profiles.toml.example`), then choose between them:

```bash
llmcli --list-profiles          # see what is configured
llmcli --profile nvidia         # start with a profile
```

Inside the CLI use `/profiles` and `/profile gemini` (add `--keep` to carry the
conversation text over). API keys are never stored in `profiles.toml`; each profile
names an environment variable with `api_key_env`. Details, precedence rules and
troubleshooting are in [user-guide.md](./user-guide.md#profiles).

## Example

```bash
llmcli -p "Explain @src/main.py"
```

In the interactive console, press Enter to send and Esc+Enter to add a line.
Use `/help` to see commands, `/clear` to reset the conversation, `/model <name>`
to switch models, `/theme` to choose a color preset, `/memory` to manage local
memory, and `/quit` to exit. See [user-guide.md](./user-guide.md) for all options.
