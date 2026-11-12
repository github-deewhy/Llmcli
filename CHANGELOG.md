# Changelog

Format: newest first. Update this file and the docs (README.md, user-guide.md,
.env.example, profiles.toml.example) in the same change as the code.

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
- GitHub Actions CI (Python 3.10 to 3.13 on Linux, 3.12 on macOS and Windows), ruff, mypy.
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
