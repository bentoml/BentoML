from __future__ import annotations

import os
from pathlib import Path

import pytest

from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import BentoPathSpec


def _included(ctx: Path) -> set[str]:
    cfg = BentoBuildConfig(service="service:svc").with_defaults()
    spec = BentoPathSpec(cfg.include, cfg.exclude, str(ctx))
    found: set[str] = set()
    for root, _, files in os.walk(ctx):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), ctx).replace(os.sep, "/")
            if spec.includes(rel):
                found.add(rel)
    return found


def _make_venv(path: Path) -> None:
    site = path / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    (site / "dep.py").write_text("x = 1")
    (path / "pyvenv.cfg").write_text("home = /usr/bin\n")


# #4410: only ``.venv/`` and ``venv/`` were skipped by name, so a virtualenv under
# any other name was copied into the bento (and from there into the Docker context).
@pytest.mark.parametrize(
    "venv_dir", ["myenv", "env", ".env", "py311", "tools/build-env"]
)
def test_virtualenv_with_any_name_is_excluded(tmp_path: Path, venv_dir: str):
    (tmp_path / "service.py").write_text("# svc")
    _make_venv(tmp_path / venv_dir)

    assert _included(tmp_path) == {"service.py"}


def test_dir_without_pyvenv_cfg_is_kept(tmp_path: Path):
    # the marker is what matters: a dir that merely *looks* like an env is user code
    (tmp_path / "service.py").write_text("# svc")
    (tmp_path / "myenv" / "lib").mkdir(parents=True)
    (tmp_path / "myenv" / "lib" / "helpers.py").write_text("# user code")

    assert _included(tmp_path) == {"service.py", "myenv/lib/helpers.py"}
