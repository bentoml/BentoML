from __future__ import annotations

import os

import pytest

from bentoml._internal.utils.filesystem import resolve_user_filepath


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory that lives under a hidden ancestor, as a bento built with
    BENTOML_HOME=~/.bentoml does."""
    root = tmp_path / ".hidden-home" / "proj"
    (root / "sub").mkdir(parents=True)
    (root / "requirements.txt").write_text("bentoml\n")
    (root / ".secret").write_text("token\n")
    (root / "sub" / ".env").write_text("KEY=value\n")
    (tmp_path / "outside.txt").write_text("outside\n")
    monkeypatch.chdir(root)
    return root


def test_relative_path_under_hidden_ancestor(project):
    assert resolve_user_filepath("requirements.txt", None) == str(
        project / "requirements.txt"
    )
    assert resolve_user_filepath("./requirements.txt", None) == str(
        project / "requirements.txt"
    )


def test_relative_to_ctx(project):
    (project / "sub" / "extra.txt").write_text("x\n")
    assert resolve_user_filepath("extra.txt", os.path.join(str(project), "sub")) == str(
        project / "sub" / "extra.txt"
    )


@pytest.mark.parametrize("path", [".secret", "sub/.env", "./.secret"])
def test_hidden_files_below_cwd_are_rejected(project, path):
    with pytest.raises(ValueError, match="hidden files"):
        resolve_user_filepath(path, None)


def test_escaping_cwd_is_rejected(project):
    with pytest.raises(ValueError, match="outside of current working directory"):
        resolve_user_filepath("../../outside.txt", None)


def test_absolute_path_is_rejected(project):
    with pytest.raises(ValueError, match="Absolute path"):
        resolve_user_filepath(str(project / "requirements.txt"), None)


def test_absolute_path_allowed_when_insecure(project):
    target = str(project / "requirements.txt")
    assert resolve_user_filepath(target, None, secure=False) == target


def test_missing_file(project):
    with pytest.raises(FileNotFoundError):
        resolve_user_filepath("nope.txt", None)
