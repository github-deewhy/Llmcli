# LLMCLI user guide

LLMCLI is a coding-focused assistant for Ollama and OpenAI-compatible APIs. In
interactive mode it can inspect project files, search source/logs, propose
changes, and run tests using model-requested tools.

## Starting the CLI

```text
llmcli [--profile NAME] [--model MODEL] [--theme THEME]
llmcli -p PROMPT
llmcli -p -
llmcli --list-profiles
```

- `-p, --prompt PROMPT`: send one prompt and exit. Use `-p -` to read the prompt
  from standard input.
- `--model MODEL`: override the configured model for this run (wins over a profile's model).
- `--profile NAME`: start with a named profile from `~/.llmcli/profiles.toml`.
- `--list-profiles`: show configured profiles and whether their API keys are set, then exit.
- `--theme THEME`: choose `ocean`, `forest`, `sunset`, or `mono` for this run.
- `-h, --help`: show command-line options.

In the interactive prompt, Enter sends the message and Esc+Enter inserts a
newline. Slash commands must start with `/`; other input, such as `clear`, is
sent as a prompt. Ctrl-C cancels an active generation or tool run; at the prompt
it exits the CLI. Type `@path/to/file` in a prompt to include a file.

## Interactive coding agent

The interactive CLI exposes these tools to the model:

- `list_files`, `read_file`, and `search_files` inspect the current working
  directory (the project root). `read_file` can select line ranges and read the
  tail of large logs. `.env` files, common credential files, and generated
  directories are excluded. Use `{"path":"server.log","tail":true}` for a large
  log's final lines.
- `replace_in_file` and `create_file` show a diff and require you to type `yes`
  before changing anything. Replacements must match one unique text block.
- `run_tests` runs a recognized test command only after you type `yes`.

Test command families include pytest, npm/yarn/pnpm test, Cargo, Go, Swift,
dotnet, and `make test`. Commands run without a shell, from the project root,
with a timeout and capped output. The tool runner does not offer general shell
access, file deletion, package installation, Git push, or MCP integrations.
Paths are confined to the project root; write access is limited to the explicit
create/replace tools. Model-requested edits and test execution are never
auto-approved.

The model/provider must support structured tool calls. If it returns ordinary
text without requesting a tool, the CLI reports that no files were changed and
no tests were run. Ollama support depends on the selected model's tool-calling
capability. One-shot `-p` mode remains text-only.

## Interactive commands

| Command | Details |
| --- | --- |
| `/help` or `/?` | List commands and their usage. |
| `/clear` or `/reset` | Clear the current conversation, keeping its instructions and memory. |
| `/status` or `/config` | Show active profile, provider, model, endpoint, and token settings. |
| `/system` or `/context` | Preview the system instructions and loaded context. |
| `/model <name>` | Switch the active model for this session. |
| `/models` | Ask the configured provider for its available models. |
| `/theme` | Show the current preset and available palettes. |
| `/theme <name>` | Change the current session's colors to `ocean`, `forest`, `sunset`, or `mono`. |
| `/memory` | Show memory command syntax and the local Markdown file paths. |
| `/memory show [global\|project\|all]` | Display one or both memory files. Defaults to `all`. |
| `/memory add <Markdown text>` | Append a note to this project's memory. |
| `/memory add --global <Markdown text>` | Append a note to memory shared across projects. |
| `/memory clear <global\|project\|all>` | Clear the selected memory file(s). Type `clear` at the confirmation prompt to proceed. |
| `/memory refresh` | Reload memory files from disk into the current session. |
| `/quit` or `/exit` | Exit the interactive CLI. |

Themes chosen with `/theme` apply for the current process. Set `LLMCLI_THEME` in
the environment or `.env` file to select a default for future runs; `--theme`
overrides that setting.

## Local persistent memory

Memory consists of ordinary Markdown files stored under `~/.llmcli/`:

- Global: `~/.llmcli/MEMORY.md` (shared across projects)
- Project: `~/.llmcli/projects/<project-name>-<path-hash>/MEMORY.md`

LLMCLI loads both files at startup and adds their contents to the coding context.
The project path hash keeps projects with the same directory name separate.
Files are created only when you use `/memory add`; memory is stored locally and
is not managed by a separate cloud database. When you chat with a remote
provider, loaded memory is included in the prompt sent to that provider, just
like the current conversation and project context. Edit the Markdown files
directly if you prefer, then run `/memory refresh` to load edits into an
already-running session. `/memory add` updates the active session immediately
as well as saving the note.

## Profiles

A profile is one complete provider setup: provider, endpoint, model, API key source, and
options. Define as many as you like and switch between them without editing `.env`.

### Setting up

Create `~/.llmcli/profiles.toml` (`setup.sh` copies `profiles.toml.example` there if the
file does not exist):

```toml
# default = "local"        # optional; must be above the [profiles.*] sections

[profiles.local]
provider = "ollama"
endpoint = "http://localhost:11434"
model = "qwen2.5-coder"

[profiles.nvidia]
provider = "openai_compat"
endpoint = "https://integrate.api.nvidia.com/v1"
model = "nvidia/nemotron-3-ultra-550b-a55b"
api_key_env = "NVIDIA_API_KEY"
max_tokens = 16384
temperature = 1
top_p = 0.95
extra_body = { chat_template_kwargs = { enable_thinking = true } }
```

Then put the real key in your environment or in `~/.llmcli/.env`:

```
NVIDIA_API_KEY=nvapi-...
```

Profile names may use letters, digits, `.`, `_` and `-`. Allowed settings are `provider`,
`model`, `endpoint`, `api_key_env`, `context_size`, `max_tokens`, `temperature`, `top_p`
and `extra_body`. Anything else (including a literal `api_key`) is rejected so typos and
accidentally stored secrets are caught immediately.

### Choosing a profile

In order of priority: `--profile NAME`, then `LLMCLI_PROFILE`, then `default` in
`profiles.toml`. If none applies, no profile is active and the plain `LLMCLI_*` settings
work exactly as before.

Inside the CLI:

- `/profiles` lists every profile. `●` marks the active one; the last column shows
  whether the profile's key variable is set.
- `/profile gemini` switches providers mid-session. It first checks that the new
  endpoint responds (Ollama also confirms the model is installed); if anything fails (unknown profile,
  missing key, endpoint down) you see the error and your current setup is left
  untouched.
- By default a switch starts a fresh conversation, because tool-call messages differ
  between providers. `/profile gemini --keep` carries over the plain user/assistant text
  and drops tool calls and tool results. Memory and project context always stay.
- `/model <name>` only changes the model within the current profile.

### How API keys work

A profile's key comes only from the variable named in `api_key_env` (looked up in the
environment, then `./.env`, then `~/.llmcli/.env`). A profile never falls back to
`LLMCLI_API_KEY`, so one service's key cannot be sent to another service's endpoint. A
profile without `api_key_env` sends no key (right for local Ollama).

### Precedence

From highest to lowest: CLI flags such as `--model`, the active profile, real environment
variables, `./.env`, `~/.llmcli/.env`, built-in defaults. Settings a profile does not
mention fall through to the lower layers. Note that an active profile therefore beats a
conflicting `LLMCLI_MODEL` in your environment; `/status` always shows which profile is in
effect.

### Troubleshooting

| Message | Meaning and fix |
| --- | --- |
| `Unknown profile 'x'. Available: ...` | Name is not in `profiles.toml`; check `/profiles`. |
| `... reads its API key from NVIDIA_API_KEY, which is not set` | Export the variable or add it to `~/.llmcli/.env`. |
| `... unknown setting(s): modle` | Typo in a profile key; the message lists the allowed names. |
| `unknown top-level key(s)` | A setting sits outside any `[profiles.<name>]` section. |
| `default profile 'x' is not defined` | Fix or remove the `default = ...` line. |

## Configuration

Without profiles, configuration is set with environment variables or `.env` files
(`./.env` overrides `~/.llmcli/.env`; real environment variables override both). Run
`llmcli` from any folder; keep API keys in `~/.llmcli/.env` rather than a project `.env`.

| Variable | Purpose |
| --- | --- |
| `LLMCLI_PROFILE` | Name of the profile to activate (see [Profiles](#profiles)). |
| `LLMCLI_PROVIDER` | `ollama` or `openai_compat`. |
| `LLMCLI_MODEL` | Default model name. |
| `LLMCLI_ENDPOINT` | Provider API base URL, as the service documents it. For `openai_compat` a bare host such as `http://localhost:1234` gets `/v1` appended; URLs with a path (`.../v1`, `.../v1beta/openai/`) are used as given. |
| `LLMCLI_API_KEY` | Optional API key for compatible endpoints (not used by profiles). |
| `LLMCLI_CONTEXT_SIZE` | Context window size. |
| `LLMCLI_MAX_TOKENS` | Maximum generated tokens. |
| `LLMCLI_TEMPERATURE` | Optional sampling temperature. |
| `LLMCLI_TOP_P` | Optional nucleus-sampling value. |
| `LLMCLI_EXTRA_BODY` | Optional JSON object merged into OpenAI-compatible requests, e.g. `{"chat_template_kwargs":{"enable_thinking":true}}`. |
| `LLMCLI_CONTEXT_FILE` | Project context filename; defaults to `CONTEXT.md`. |
| `LLMCLI_THEME` | Default color preset. |

### `@file` attachments

`@path/to/file.ext` in a prompt inlines that file. Text that merely looks like a
reference (e-mail addresses, files that do not exist) is left alone. `.env` files, key and
credential files, credential directories (`.ssh`, `.aws`, ...), generated directories
inside the project, and files over 512 KB are not attached; a short notice is shown instead.
