"""Tests for local persistent Markdown memory."""

from llmcli.memory import MemoryStore


def test_memory_is_local_and_separated_by_project(tmp_path):
    root = tmp_path / "home" / ".llmcli"
    project_a = tmp_path / "work" / "service"
    project_b = tmp_path / "personal" / "service"
    project_a.mkdir(parents=True)
    project_b.mkdir(parents=True)

    store_a = MemoryStore(root=root, project_root=project_a)
    store_b = MemoryStore(root=root, project_root=project_b)

    assert store_a.paths["global"] == store_b.paths["global"]
    assert store_a.paths["project"] != store_b.paths["project"]
    assert root in store_a.paths["project"].parents
    assert not store_a.paths["project"].exists()


def test_memory_add_appends_markdown_and_composes_system_prompt(tmp_path):
    store = MemoryStore(
        base_prompt="You are a coding assistant.",
        root=tmp_path / ".llmcli",
        project_root=tmp_path / "project",
    )
    store.add("global", "Use concise answers.")
    store.add("project", "Run tests with pytest.")

    prompt = store.compose_system_prompt()

    assert prompt.startswith("You are a coding assistant.")
    assert "Persistent global memory" in prompt
    assert "Use concise answers." in prompt
    assert "Persistent project memory" in prompt
    assert "Run tests with pytest." in prompt

    next_session = MemoryStore(
        base_prompt="You are a coding assistant.",
        root=tmp_path / ".llmcli",
        project_root=tmp_path / "project",
    )
    next_prompt = next_session.compose_system_prompt()
    assert "Use concise answers." in next_prompt
    assert "Run tests with pytest." in next_prompt


def test_memory_rejects_empty_notes(tmp_path):
    store = MemoryStore(root=tmp_path / ".llmcli", project_root=tmp_path / "project")

    try:
        store.add("project", "  ")
    except ValueError as error:
        assert "cannot be empty" in str(error)
    else:
        raise AssertionError("Expected empty memory note to be rejected")


def test_memory_path_that_is_a_directory_is_reported(tmp_path):
    store = MemoryStore(root=tmp_path / ".llmcli", project_root=tmp_path / "project")
    store.paths["global"].mkdir(parents=True)

    try:
        store.read()
    except OSError as error:
        assert "not a file" in str(error)
    else:
        raise AssertionError("Expected invalid memory path to be reported")
