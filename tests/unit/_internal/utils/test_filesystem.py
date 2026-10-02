from __future__ import annotations

import os
from pathlib import Path

import pytest

from bentoml._internal.utils.filesystem import resolve_user_filepath


@pytest.fixture()
def build_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    ctx = tmp_path / "build"
    ctx.mkdir()
    (ctx / "Dockerfile.template").write_text("FROM python:3.11-slim")
    return ctx


def test_resolves_template_in_context_outside_cwd(build_context: Path) -> None:
    assert resolve_user_filepath("Dockerfile.template", str(build_context)) == str(
        build_context / "Dockerfile.template"
    )


def test_uses_cwd_when_context_is_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    template = tmp_path / "Dockerfile.template"
    template.write_text("FROM python:3.11-slim")
    assert resolve_user_filepath(template.name, None) == str(template)


def test_allows_context_with_hidden_ancestor(tmp_path: Path, monkeypatch) -> None:
    ctx = tmp_path / ".cache" / "build"
    ctx.mkdir(parents=True)
    template = ctx / "Dockerfile.template"
    template.touch()
    monkeypatch.chdir(tmp_path)
    assert resolve_user_filepath(template.name, str(ctx)) == str(template)


def test_rejects_existing_file_outside_context(build_context: Path) -> None:
    outside = build_context.parent / "outside.txt"
    outside.touch()
    with pytest.raises(ValueError, match="outside"):
        resolve_user_filepath("../outside.txt", str(build_context))


@pytest.mark.parametrize("name", [".secret", ".hidden/template"])
def test_rejects_hidden_paths(build_context: Path, name: str) -> None:
    target = build_context / name
    target.parent.mkdir(exist_ok=True)
    target.touch()
    with pytest.raises(ValueError, match="hidden"):
        resolve_user_filepath(name, str(build_context))


def test_rejects_absolute_path(build_context: Path) -> None:
    with pytest.raises(ValueError, match="Absolute path"):
        resolve_user_filepath(
            str(build_context / "Dockerfile.template"), str(build_context)
        )


def test_missing_file(build_context: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_user_filepath("missing.txt", str(build_context))


def test_insecure_mode_allows_outside_file(build_context: Path) -> None:
    outside = build_context.parent / "outside.txt"
    outside.touch()
    assert resolve_user_filepath(
        "../outside.txt", str(build_context), secure=False
    ) == str(outside)


def test_rejects_symlink_escape(build_context: Path) -> None:
    outside = build_context.parent / "outside.txt"
    outside.touch()
    link = build_context / "template"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on Windows")
    with pytest.raises(ValueError, match="outside"):
        resolve_user_filepath(link.name, str(build_context))


@pytest.mark.skipif(os.name == "nt", reason="POSIX system directories")
def test_rejects_system_file_even_with_system_context() -> None:
    with pytest.raises(ValueError, match="system"):
        resolve_user_filepath("passwd", "/etc")
