"""Shared rules for files that must never be exposed to a model."""

from pathlib import Path

SENSITIVE_FILENAMES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "secret",
    "secrets",
}
SENSITIVE_SUFFIXES = (".secret", ".secrets", ".token")
SENSITIVE_EXTENSIONS = {".pem", ".key", ".p12", ".pfx"}

IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".aws",
    ".ssh",
    ".kube",
}
# Directories that are sensitive wherever they appear on disk.
CREDENTIAL_DIRECTORIES = {".aws", ".ssh", ".kube", ".gnupg"}


def filename_block_reason(path: Path) -> str | None:
    """Return why a file's *name* is blocked, or None if it is allowed."""
    filename = path.name.lower()
    if filename == ".env" or filename.startswith(".env."):
        return "Access to environment files is blocked."
    if (
        filename in SENSITIVE_FILENAMES
        or filename.endswith(SENSITIVE_SUFFIXES)
        or path.suffix.lower() in SENSITIVE_EXTENSIONS
    ):
        return "Access to credential or private-key files is blocked."
    return None
