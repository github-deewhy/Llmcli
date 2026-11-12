# LLMCLI

A focused coding assistant for local and OpenAI-compatible language models.

## Features

- Interactive REPL with history and Enter-to-send input
- Streaming responses with Rich rendering, also while the agent uses tools
- Local first: Ollama (the tested server), plus OpenAI-compatible local servers (llama.cpp `llama-server`, LM Studio, vLLM; not tested by us); a remote OpenAI-compatible URL is optional
- Thinking text from reasoning models is kept out of the answer and out of history (`/thinking on` shows it)
- Works with models that write tool calls as text instead of the native format
- `/status` shows what the server and model can do and how fast the last answer was
- One configuration file with named profiles; switch servers with `/profile`
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

## Configuration

With a local Ollama on its default port (`http://localhost:11434`) llmcli works with
no configuration at all. For anything else, edit one file, `~/.llmcli/config.toml`
(`setup.sh` creates it from `config.toml.example`):

```toml
[defaults]
profile = "llamacpp"

[profiles.llamacpp]
provider = "openai_compat"
endpoint = "http://localhost:9931/v1"
model = "local-model"
```

That file is the only place settings are read from. Command-line flags (`--profile`,
`--model`, `--theme`) override it for one run. No `.env` file and no environment
variable is read. A local server needs no API key; a remote URL may have one
`api_key` in its profile, and llmcli warns if the file is readable by other users.

```bash
llmcli --list-profiles          # see what is configured
llmcli --profile vllm           # start with a profile
```

Inside the CLI use `/profiles` and `/profile <name>` (add `--keep` to carry the
conversation text over). Every setting, the precedence rules and troubleshooting
are in [user-guide.md](./user-guide.md#configuration).

Interactive agent workflows work best with a model that supports native tool calls;
models that write tool calls as text are handled too (see `tool_mode` in the user
guide). One-shot `-p` prompts remain text-only.

## Tested servers

Ollama is the only server the llmcli project tests. llama.cpp `llama-server`, LM Studio
and vLLM use the same OpenAI-compatible code path and should work, but are not tested.

Live tests run against your own server and are skipped by default:

```bash
pytest -m live --live-profile ollama -v   # profile name from ~/.llmcli/config.toml
```

## Example

```bash
llmcli -p "Explain @src/main.py"
```

In the interactive console, press Enter to send and Esc+Enter to add a line.
Use `/help` to see commands, `/clear` to reset the conversation, `/model <name>`
to switch models, `/theme` to choose a color preset, `/memory` to manage local
memory, and `/quit` to exit. See [user-guide.md](./user-guide.md) for all options.

## Supported platforms

Linux and macOS (Python 3.10 or newer). Windows is not supported yet.
