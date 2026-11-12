# Security policy

llmcli runs commands and edits files on your machine on behalf of a language model, so security reports are taken seriously.

## Reporting a vulnerability

Please do **not** open a public issue for security problems.
Use GitHub's private reporting: **Security → Report a vulnerability** on this repository.
Include the version (`llmcli` v…), your OS, and steps to reproduce.

## Scope

In scope: path confinement bypasses, secret leakage (keys in logs, prompts or errors), unapproved command execution, prompt-injection paths that change permissions.
Out of scope: model output quality, and attacks that require the user to approve a malicious action after reading it.

## Status

This is pre-release software. Known gaps are tracked in the project documentation (security and trust model). Do not run it on untrusted repositories until folder trust is implemented (milestone M6).
