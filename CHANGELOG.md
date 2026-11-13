# Changelog

Format: newest first. Update this file and the docs (README.md, user-guide.md,
config.toml.example) in the same change as the code.

## 0.8.1

### Fixed
- In text tool mode, tool results and repair notes are stored as user messages, and
  trimming treated them as the start of a new turn, so it could drop the task prompt
  in the middle of a long task. They are now marked as part of the current turn.

## 0.8.0

### Changed
- History budgeting now counts everything sent: system prompt, agent instructions and
  tool schemas, not only the messages (M3 part C, spec 16).
- Part of the window is kept free for the reply: `max_tokens`, at most a quarter of
  `context_size` (with the defaults 8192/4096: 2048 tokens).
- The 4-characters-per-token estimate is calibrated from the prompt token count the
  server reports (latest request, limited to 2–6 characters per token, kept in memory
  only, reset on `/profile`).
- History is trimmed right before each request instead of when a message is added.

### Added
- One line when history is trimmed (`History trimmed: dropped N oldest turns …`).
- A warning (at most once per question) when the prompt is still over budget and no
  older turn is left to drop.
- Benchmark runs pass `--max-tokens` to the budget, like the CLI.


### Changed
- `stop_guard` is now on by default. Measured on 2026-11-13 (NVIDIA API, 6 tasks × 3
  repeats): Lightning 30B-A3B passed 18/18 with 1 round-limit hit (5 without it).
  `edit_matching` stays `exact` (no measured gain) and `edit_tools` stays `replace`
  (helps the small Bolt model, costs Lightning 2 of 18); both remain available.

## 0.7.0

M3 part B: edit reliability and stopping (spec 16). The new behaviours are off by
default until the benchmark shows which help; only the clearer error is always on.

### Added
- When `replace_in_file` cannot find `old_text`, the error now shows the most similar
  lines of the file with their line numbers (up to 15), so the model can copy the
  exact text.
- Setting `edit_matching = "tolerant"`: if `old_text` is not found exactly, try once
  with trailing spaces ignored and escaped quotes (`\"`, `\'`) undone, in both
  `old_text` and `new_text`. The edit applies only if exactly one place matches and
  the result says what was ignored. Escaped newlines are not touched.
- Setting `edit_tools = "replace+lines"`: adds the `edit_lines` tool (replace a range
  of line numbers as `read_file` shows them), with the same diff and approval.
- Setting `stop_guard = true`: one sentence asks the model to answer in text when the
  task is done, and a tool call identical to the previous one is not run again; the
  model gets a short note instead.
- Benchmark: `--set KEY=VALUE` overrides these settings for a run; reports show the
  settings per profile and a "Repeats skipped" column. `report.json` schema version 3.

### Changed
- Benchmark reports are labelled `m3b-v0.7`.

## 0.6.0

M3 part A: measurement fixes (spec 16). Agent behaviour is unchanged.

### Added
- Benchmark: `--rpm N` paces model requests for servers with a rate limit; waiting
  time is excluded from wall time and from the task deadline.
- Benchmark: a tool-errors count per run and per task (tool calls that ran and
  failed, separate from invalid calls), and the most common provider error and tool
  error per task in `report.md`. `report.json` schema version 2.

### Changed
- OpenAI-compatible servers are asked for token counts
  (`stream_options.include_usage`). If a server answers HTTP 400, the request is
  sent once more without it, and it is not sent again to that server for the
  session. A `stream_options` value in `extra_body` is used as given.
- Tokens/s measured by llmcli (OpenAI-compatible servers) is shown only when at
  least 8 tokens arrived over at least 0.5 s; otherwise it is unknown. Before, a
  reply that arrived in one burst showed absurd speeds.
- Benchmark reports are labelled `baseline-v0.6`.

## 0.5.1

Fixes from the first benchmark run on a real model (Ollama, `qwen3:4b`).

### Fixed
- Some thinking models (Ollama's `qwen3:4b` is Qwen3 4B Thinking 2507) write their
  reasoning into the answer and end it with `</think>` without ever opening it, even
  with `think = false`. That text was shown as the answer, stored in history and sent
  back every round. It is now treated as reasoning: hidden unless `/thinking on`,
  never stored or re-sent. Limitation: an answer that mentions `</think>` before any
  other tag loses the text before it.
- Benchmark reports record the commit even when `git` is not installed (read from
  `.git`).

## 0.5.0

M2 part 1: benchmark harness, tested with scripted replies only.

### Added
- `python -m llmcli.bench` (`list`, `check`, `run`): a development tool, not a
  `llmcli` command. Six tasks in `bench/tasks/` (explain, fix a bug, add a function,
  refactor, write tests, multi-file change), each with hidden checks and a scripted
  solution; `check` and the test suite confirm every solution passes and doing
  nothing fails.
- Reports in `bench-results/`: `report.json`, `report.md` and one transcript per run,
  with success, wall time, tokens, tokens/s, rounds, tool-call validity, loop-guard
  hits, peak context use and history trims, plus the machine and profile settings
  (never the API key).
- Per-task deadline and `--max-tokens` cap for real-model runs; real-model runs need
  `--i-understand-this-runs-model-code` because edits and tests are approved
  automatically.

### Changed
- Internal: `CodingAgent.run()` returns how the run ended (`RunOutcome`);
  `Conversation` counts history trims. The interactive CLI behaves as before.
- `llmcli.__version__` now matches the package version (it still said 0.3.0).

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
