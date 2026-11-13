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
- `--profile NAME`: start with a named profile from `~/.llmcli/config.toml`.
- `--list-profiles`: show the profiles in `~/.llmcli/config.toml` and whether each has an API key, then exit.
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

Answers stream into the terminal while the model writes them, including when it
uses tools. Ctrl-C stops the answer at once; the unfinished answer is not kept in
the conversation.

Tool calls work in two ways, chosen by `tool_mode` (see Configuration):

- **Native**: the server returns structured tool calls. This is the default.
- **Text**: the model writes the call as a JSON block,
  `{"name": "read_file", "arguments": {"path": "app.py"}}`, and llmcli runs it.
  Small models often do this even when native calls are offered. With
  `tool_mode = "auto"` llmcli accepts such blocks and, after two in a row, switches
  the session to text mode. A call with a wrong tool name or wrong arguments gets
  one short error back so the model can correct it, once per question.

If the model answers with ordinary text and no tool call, the CLI reports that no
files were changed and no tests were run. One-shot `-p` mode remains text-only.

Thinking models (for example `qwen3`) write reasoning before the answer. llmcli
shows only a short "thinking…" line while that happens, never stores the reasoning
in the conversation and never sends it back to the model. `/thinking on` shows the
full text for the session. To switch thinking off, use the server's option in
`extra_body`: `{ think = false }` for Ollama.

## Interactive commands

| Command | Details |
| --- | --- |
| `/help` or `/?` | List commands and their usage. |
| `/clear` or `/reset` | Clear the current conversation, keeping its instructions and memory. |
| `/status` or `/config` | Show active profile, provider, model, endpoint, token and timeout settings, tool mode, what the server and model support (with where each value came from), and the speed of the last answer. |
| `/system` or `/context` | Preview the system instructions and loaded context. |
| `/model <name>` | Switch the active model for this session. |
| `/models` | Ask the configured provider for its available models. |
| `/probe` | Send one tiny request with a test tool and record whether the model makes a native tool call. Takes a few seconds to minutes on CPU. |
| `/thinking [on\|off]` | Show or hide a thinking model's reasoning for this session (hidden by default). |
| `/theme` | Show the current preset and available palettes. |
| `/theme <name>` | Change the current session's colors to `ocean`, `forest`, `sunset`, or `mono`. |
| `/memory` | Show memory command syntax and the local Markdown file paths. |
| `/memory show [global\|project\|all]` | Display one or both memory files. Defaults to `all`. |
| `/memory add <Markdown text>` | Append a note to this project's memory. |
| `/memory add --global <Markdown text>` | Append a note to memory shared across projects. |
| `/memory clear <global\|project\|all>` | Clear the selected memory file(s). Type `clear` at the confirmation prompt to proceed. |
| `/memory refresh` | Reload memory files from disk into the current session. |
| `/quit` or `/exit` | Exit the interactive CLI. |

Themes chosen with `/theme` apply for the current process. Set `theme` in
`[defaults]` of `~/.llmcli/config.toml` to select a default for future runs;
`--theme` overrides that setting.

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

## Configuration

All settings live in one file, `~/.llmcli/config.toml`. llmcli reads it and never
writes it. No `.env` file (in your home folder or in a project) and no environment
variable is read for settings.

With no file, llmcli uses a local Ollama: provider `ollama`, endpoint
`http://localhost:11434`, model `qwen2.5-coder`. `setup.sh` copies
`config.toml.example` to `~/.llmcli/config.toml` (mode 600) if the file does not exist.

### The file

```toml
[defaults]
profile = "ollama"          # profile used when --profile is not given
theme = "ocean"
context_size = 8192

[profiles.ollama]
provider = "ollama"
endpoint = "http://localhost:11434"
model = "qwen2.5-coder"

[profiles.vllm]
provider = "openai_compat"
endpoint = "http://localhost:8000/v1"
model = "Qwen/Qwen2.5-Coder-7B-Instruct"
context_size = 32768        # overrides [defaults] for this profile only
```

Only two kinds of table exist: `[defaults]` and `[profiles.<name>]`. Profile names may
use letters, digits, `.`, `_` and `-`.

### Where each setting goes

| Setting | `[defaults]` | Profile | Built-in default | Meaning |
| --- | --- | --- | --- | --- |
| `profile` | yes | no | none | Profile used when `--profile` is not given |
| `theme` | yes | no | `ocean` | `ocean`, `forest`, `sunset` or `mono` |
| `context_file` | yes | no | `AGENTS.md` | Project instruction file; a legacy `CONTEXT.md` is read when no `AGENTS.md` exists |
| `provider` | no | required | `ollama` | `ollama` or `openai_compat` |
| `endpoint` | no | required | `http://localhost:11434` | Server base URL (`http://` or `https://`). For `openai_compat` a bare host such as `http://localhost:1234` gets `/v1` appended; a URL with a path is used as given |
| `model` | no | required | `qwen2.5-coder` | Model name or id |
| `api_key` | no | optional | none | Only for a remote URL; never needed for a local server |
| `context_size` | yes | yes | 8192 | Context window in tokens. When the conversation no longer fits, the oldest turns are dropped and llmcli says so |
| `max_tokens` | yes | yes | 4096 | Maximum generated tokens. This much of `context_size` (at most a quarter) is kept free for the reply |
| `temperature` | yes | yes | server default | Sampling temperature |
| `top_p` | yes | yes | server default | Nucleus sampling |
| `extra_body` | yes | yes | none | Table merged into every request, e.g. `{ think = false }` for Ollama or `{ chat_template_kwargs = { enable_thinking = false } }` for llama.cpp or vLLM. A profile's value replaces the default one; they are not merged |
| `request_timeout` | yes | yes | 300 | Seconds of silence allowed from the server before llmcli gives up, including model load and prompt processing. Raise it on slow CPU machines |
| `tool_mode` | yes | yes | `auto` | `auto` (native, accepts text calls and switches after two), `native` (never reads text calls) or `text` (always uses text calls) |
| `edit_matching` | yes | yes | `exact` | `exact`, or `tolerant`: if `old_text` is not found exactly, try once with trailing spaces ignored and escaped quotes (`\"`) undone; applied only if exactly one place matches, and the result says so. Experimental; the default may change after measurement |
| `edit_tools` | yes | yes | `replace` | `replace`, or `replace+lines` to also offer `edit_lines` (replace a range of line numbers as `read_file` shows them). Experimental |
| `stop_guard` | yes | yes | `true` | `true` tells the model to answer in text when the task is done, and does not run a tool call that repeats the previous one exactly. On by default since 0.7.1 (measured: fewer round-limit hits) |

Server, model and key settings are profile-only, so a key can never be sent to another
profile's endpoint. A setting in the wrong table, an unknown setting or a wrong type is
an error that names the file and the setting.

### Precedence

From highest to lowest: command-line flags (`--model`, `--theme`), the active profile,
`[defaults]`, built-in defaults. `/status` shows which file and profile are in effect.

### Choosing a profile

1. `--profile NAME` on the command line, or `/profile NAME` in a session.
2. Otherwise `profile` in `[defaults]`.
3. Otherwise, if the file defines exactly one profile, that one.
4. If it defines several and none is chosen, llmcli stops and asks you to pick one.
5. If it defines none, the built-in local Ollama is used.

Inside the CLI:

- `/profiles` lists every profile. `●` marks the active one; the last column shows
  whether the profile has an API key (`set` or `none`). The key itself is never shown.
- `/profile NAME` switches servers mid-session. It first checks that the new endpoint
  responds (Ollama also confirms the model is installed); if anything fails your
  current setup is left untouched.
- By default a switch starts a fresh conversation, because tool-call messages differ
  between providers. `/profile NAME --keep` carries over the plain user/assistant text
  and drops tool calls and tool results. Memory and project context always stay.
- `/model <name>` and `/theme <name>` change the running session only; the file is
  not changed.

### API keys

Local servers need no key. For a remote OpenAI-compatible URL, put `api_key` in that
profile. Keep the file private (`chmod 600 ~/.llmcli/config.toml`); llmcli prints a
warning at start-up when the file contains a key and other users can read it.

### Upgrading from 0.2

Version 0.3 ignores `~/.llmcli/profiles.toml`, `~/.llmcli/.env`, any project `.env`,
and all `LLMCLI_*` environment variables. If it finds the old files or variables it
prints one line saying they are ignored; nothing is converted automatically. Move your
settings into `~/.llmcli/config.toml`: `api_key_env = "X"` becomes `api_key = "<value>"`
in the profile, and `default = "name"` becomes `profile = "name"` inside `[defaults]`.

### Troubleshooting

| Message | Meaning and fix |
| --- | --- |
| `Unknown profile 'x'. Available: ...` | The name is not in the file; check `/profiles`. |
| `Several profiles are defined ...` | Pass `--profile NAME` or set `profile` in `[defaults]`. |
| `... unknown setting 'modle'` | Typo; the message lists the allowed names. |
| `... is not allowed here; put it in ...` | The setting belongs in the other table (see the table above). |
| `... missing required setting(s): endpoint` | Every profile needs `provider`, `endpoint` and `model`. |
| `'default' at the top level is not used` | Write `profile = "name"` inside `[defaults]`. |
| `not valid TOML: ... (at line N, column M)` | Syntax error at that position. |
| `Cannot reach http://...` | The server is not running or the endpoint is wrong. |
| `No data from ... for 300 s` | The server went silent; a large model may still be loading on CPU. Raise `request_timeout`. |
| `... returned HTTP 404` | Usually the model is not installed (`ollama pull NAME`) or the URL is wrong. |
| `... sent a response llmcli could not read` | The server sent something unexpected; check its version and log. |

Network connections ignore proxy environment variables (`HTTP_PROXY`, `HTTPS_PROXY`)
and `.netrc`, because no settings are read from the environment.

### `@file` attachments

`@path/to/file.ext` in a prompt inlines that file. Text that merely looks like a
reference (e-mail addresses, files that do not exist) is left alone. `.env` files, key and
credential files, credential directories (`.ssh`, `.aws`, ...), generated directories
inside the project, and files over 512 KB are not attached; a short notice is shown instead.
