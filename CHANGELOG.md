# Changelog

Format: newest first. Update this file and the docs (README.md, user-guide.md,
config.toml.example) in the same change as the code.

## 0.4.0

M1: streaming provider layer. Only Ollama is tested against a live server.

### Added
- The agent streams answers while it works with tools; Ctrl-C stops at once and the
  unfinished answer is not kept.
- Thinking text (Ollama `thinking`, OpenAI-style `reasoning_content`, inline
  `<think>` tags) is kept apart from the answer, never stored or re-sent, and shown
  only with `/thinking on`.
- Tool calls written as text (a JSON block or `<tool_call>` tags) are accepted;
  `tool_mode` (`auto`, `native`, `text`) controls this. Tool arguments are checked
  against the tool's schema and a short error is returned to the model.
- `/probe` and a capabilities block in `/status` (context length, native tools,
  reasoning, usage, with the source of each value) plus last-answer speed.
- `request_timeout` setting (default 300 s).
- Opt-in live tests: `pytest -m live --live-profile NAME`.

### Changed
- Every request now has a finite timeout; errors are one line (cannot reach, timed
  out, HTTP status with a hint, unreadable response) and never include the server's
  reply body or the API key.
- `extra_body` is now also sent to Ollama, e.g. `{ think = false }`.
- HTTP connections ignore proxy environment variables and `.netrc`.
- Internal: providers expose one `stream()` method yielding typed events; `chat()`
  and `complete()` are removed.

## 0.3.0

Breaking: configuration moves to one file. See "Upgrading from 0.2" in user-guide.md.

### Changed
- All settings now come from `~/.llmcli/config.toml` (`[defaults]` plus
  `[profiles.<name>]`). Precedence: command-line flags > active profile > `[defaults]` >
  built-in defaults. With no file, a local Ollama is used.
- The default profile is `profile = "<name>"` inside `[defaults]` (was top-level
  `default`). With exactly one profile and no default, that profile is used; with
  several and no default, llmcli stops and asks for `--profile`.
- `provider`, `endpoint` and `model` are required in every profile and allowed only
  there; `theme` and `context_file` are allowed only in `[defaults]`.
- `/profiles` and `--list-profiles` show whether a profile has a key (`set`/`none`)
  instead of an environment-variable name. `/status` shows the configuration source.

### Removed
- `./.env`, `~/.llmcli/.env`, every `LLMCLI_*` environment variable and `api_key_env`
  are no longer read. A remote URL's key is an optional `api_key` in its profile.
- `~/.llmcli/profiles.toml`, `.env.example`, `profiles.toml.example`, the
  `python-dotenv` dependency, and cloud-vendor examples in code, tests and docs.

### Added
- `config.toml.example` with Ollama active and commented profiles for llama.cpp
  `llama-server`, LM Studio and vLLM.
- Start-up warning when `config.toml` contains an API key and other users can read it.
- Start-up notice naming old configuration that is now ignored (`~/.llmcli/profiles.toml`,
  `~/.llmcli/.env`, `LLMCLI_*` variable names). A project's own `.env` is not checked.
- Tests for precedence, profile selection, placement rules, validation, key redaction,
  permissions, the legacy notice, and a repository check for cloud-vendor names.

## 0.2.1

### Changed
- Project instruction file is now `AGENTS.md`; `CONTEXT.md` is still read as a legacy
  alias when no `AGENTS.md` exists. `CONTEXT.md.example` renamed to `AGENTS.md.example`.
- `run_oneshot` (`-p`) returns an exit code and prints short messages for connection,
  HTTP and value errors instead of a traceback (exit 1; 130 on Ctrl-C).

### Fixed
- `LLMCLI_CONTEXT_SIZE` / `LLMCLI_MAX_TOKENS` with a bad value (text, zero, negative) now
  give a clear configuration error instead of a traceback.
- Oversized tool results are shrunk (head and tail of the largest field, or fewer list
  items) to fit the budget and always stay valid JSON; previously only 1/8 of the budget
  was kept and the JSON could be cut mid-string.
- Memory and instruction files are capped at 32,000 bytes when read; `/memory add`
  refuses to grow a file past the cap.

### Added
- GitHub Actions CI (Python 3.10 to 3.13 on Linux, 3.12 on macOS), ruff, mypy. Windows is
  not supported yet and is not tested in CI.
- `LICENSE` placeholder (all rights reserved until a licence is chosen), `SECURITY.md`.
- 22 regression tests (107 total).

## 0.2.0

### Added
- **Named profiles** (`~/.llmcli/profiles.toml`): `--profile`, `--list-profiles`,
  `/profiles`, `/profile <name> [--keep]`, `LLMCLI_PROFILE`, and a `default` entry.
  Keys come from `api_key_env`, never from the file. See user-guide.md#profiles.
- `LLMCLI_TEMPERATURE`, `LLMCLI_TOP_P`, `LLMCLI_EXTRA_BODY` (JSON merged into
  OpenAI-compatible requests) for services such as NVIDIA NIM and Gemini.
- `profiles.toml.example`; `setup.sh` installs it if missing.
- `llmcli/security.py`: one shared blocklist for sensitive files.
- `tomli` dependency on Python 3.10 (`tomllib` is built in from 3.11).

### Changed
- **Config precedence fixed**: real environment variables now beat `./.env`
  (previously the file won). `.env` files are read without modifying `os.environ`.
  An active profile sits between CLI flags and the environment.
- OpenAI-compatible endpoints are used as given; only a bare host gets `/v1`
  appended (fixes Gemini `.../v1beta/openai/` and `.../v1` URLs).
- `@file` attachments ignore e-mail addresses and missing files, refuse
  sensitive/ignored paths, and cap size at 512 KB.
- `/status` shows the active profile; `/model` is documented as per-profile.

### Fixed
- Ollama health check accepted only exact names; `qwen2.5-coder` now matches
  `qwen2.5-coder:latest`.
- Invalid numeric/JSON settings give a clear "Configuration error" instead of a traceback.

### Removed
- Stray `deewhy-cli_qwen.md`, committed caches and `.env` from the distribution.

## 0.1.0
- Initial version.

## Known gaps / next
- Agent mode does not stream; reasoning text from thinking models is not shown.
- Token budgeting ignores system prompt and tool schemas; tool-output truncation is lossy.
- Test subprocesses inherit the full environment; `./.env` and `CONTEXT.md` from the
  working directory are trusted.
