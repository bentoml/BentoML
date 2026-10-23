from __future__ import annotations

import importlib.metadata
import subprocess
import sys

import pytest
from packaging.version import Version

from bentoml._internal import configuration
from bentoml._internal.configuration import UV_REQUIREMENT
from bentoml._internal.configuration import get_uv_command


@pytest.fixture(autouse=True)
def clear_cache():
    get_uv_command.cache_clear()
    yield
    get_uv_command.cache_clear()


def _no_uv_package(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


def test_uses_uv_on_path_when_new_enough(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(configuration, "_uv_version", lambda _: Version("0.13.4"))

    assert get_uv_command() == ["/usr/bin/uv"]


def test_installs_pinned_uv_when_path_uv_is_old(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        with open(cmd[cmd.index("-r") + 1]) as f:
            calls.append((cmd, f.read()))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(configuration, "_uv_version", lambda _: Version("0.9.26"))
    monkeypatch.setattr(importlib.metadata, "version", _no_uv_package)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_uv_command() == [sys.executable, "-m", "uv"]
    [(cmd, requirements)] = calls
    assert "--require-hashes" in cmd
    assert requirements.strip() == UV_REQUIREMENT


def test_keeps_new_enough_uv_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.14.0")
    monkeypatch.setattr(subprocess, "run", pytest.fail)

    assert get_uv_command() == [sys.executable, "-m", "uv"]
