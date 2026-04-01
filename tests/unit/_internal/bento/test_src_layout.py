from __future__ import annotations

import os
import shutil
from pathlib import Path

from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import BentoPathSpec


def test_src_layout_autodiscovery(tmp_path: Path):
    pyproject_toml = """
[project]
name = "my_service"
version = "0.1.0"
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_toml)
    (tmp_path / "src").mkdir()

    with open(pyproject_file, "rb") as f:
        build_config = BentoBuildConfig.from_pyproject(f, base_dir=str(tmp_path))

    assert build_config.python.is_src_layout is True


def test_src_layout_not_detected_without_src_dir(tmp_path: Path):
    pyproject_toml = """
[project]
name = "my_service"
version = "0.1.0"
"""
    pyproject_file = tmp_path / "pyproject.toml"
    pyproject_file.write_text(pyproject_toml)

    with open(pyproject_file, "rb") as f:
        build_config = BentoBuildConfig.from_pyproject(f, base_dir=str(tmp_path))

    assert build_config.python.is_src_layout is None


def test_src_layout_strips_src_prefix_during_copy(tmp_path: Path):
    """Verify that files under src/ are copied with the src/ prefix stripped
    when is_src_layout is True, so imports work correctly in the bento."""
    build_ctx = tmp_path / "project"
    build_ctx.mkdir()
    (build_ctx / "src" / "mypackage").mkdir(parents=True)
    (build_ctx / "src" / "mypackage" / "__init__.py").write_text("# init")
    (build_ctx / "src" / "mypackage" / "model.py").write_text("# model code")
    (build_ctx / "service.py").write_text("# service code")
    (build_ctx / "bentofile.yaml").write_text("service: service:svc")

    build_config = BentoBuildConfig(
        service="service:svc",
        python={"is_src_layout": True},
    ).with_defaults()

    ctx_path = build_ctx.resolve()
    specs = BentoPathSpec(build_config.include, build_config.exclude, str(build_ctx))

    target_fs = tmp_path / "bento" / "src"
    target_fs.mkdir(parents=True)

    is_src_layout = build_config.python.is_src_layout
    for root, _, files in os.walk(ctx_path):
        for f in files:
            dir_path = os.path.relpath(root, ctx_path)
            path = os.path.join(dir_path, f).replace(os.sep, "/")
            if specs.includes(path):
                dest_path = path
                if is_src_layout and path.startswith("src/"):
                    dest_path = path[4:]
                dest_dir = os.path.dirname(dest_path)
                target_fs.joinpath(dest_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy(ctx_path / path, target_fs / dest_path)

    assert (target_fs / "mypackage" / "__init__.py").exists()
    assert (target_fs / "mypackage" / "model.py").exists()
    assert (target_fs / "service.py").exists()
    assert not (target_fs / "src" / "mypackage").exists(), (
        "src/ prefix should be stripped, not nested"
    )


def test_no_src_layout_preserves_src_directory(tmp_path: Path):
    """Without is_src_layout, files under src/ keep their path."""
    build_ctx = tmp_path / "project"
    build_ctx.mkdir()
    (build_ctx / "src" / "mypackage").mkdir(parents=True)
    (build_ctx / "src" / "mypackage" / "__init__.py").write_text("# init")
    (build_ctx / "service.py").write_text("# service code")

    build_config = BentoBuildConfig(
        service="service:svc",
        python={"is_src_layout": False},
    ).with_defaults()

    ctx_path = build_ctx.resolve()
    specs = BentoPathSpec(build_config.include, build_config.exclude, str(build_ctx))

    target_fs = tmp_path / "bento" / "src"
    target_fs.mkdir(parents=True)

    is_src_layout = build_config.python.is_src_layout
    for root, _, files in os.walk(ctx_path):
        for f in files:
            dir_path = os.path.relpath(root, ctx_path)
            path = os.path.join(dir_path, f).replace(os.sep, "/")
            if specs.includes(path):
                dest_path = path
                if is_src_layout and path.startswith("src/"):
                    dest_path = path[4:]
                dest_dir = os.path.dirname(dest_path)
                target_fs.joinpath(dest_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy(ctx_path / path, target_fs / dest_path)

    assert (target_fs / "src" / "mypackage" / "__init__.py").exists(), (
        "Without is_src_layout, src/ should be preserved"
    )


def test_src_layout_in_requirements(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "bentoml._internal.configuration.is_editable_bentoml", lambda: False
    )
    monkeypatch.setattr(
        "bentoml._internal.configuration.get_bentoml_requirement",
        lambda: "bentoml==1.3.0",
    )
    monkeypatch.setattr(
        "bentoml._internal.bento.bentoml_builder.is_editable_bentoml", lambda: False
    )
    monkeypatch.setattr(
        "bentoml._internal.bento.build_config.get_bentoml_requirement",
        lambda: "bentoml==1.3.0",
    )
    monkeypatch.setattr(
        "bentoml._internal.bento.build_config.clean_bentoml_version", lambda: "1.3.0"
    )

    bento_fs = tmp_path / "bento"
    bento_fs.mkdir()

    build_ctx = tmp_path / "project"
    build_ctx.mkdir()
    (build_ctx / "src").mkdir()

    build_config = BentoBuildConfig(
        service="my_service.service:svc",
        python={"is_src_layout": True, "lock_packages": False},
    ).with_defaults()

    build_config.python.write_to_bento(bento_fs, str(build_ctx))

    requirements_file = bento_fs / "env" / "python" / "requirements.txt"
    assert requirements_file.exists()

    content = requirements_file.read_text()
    assert "bentoml==1.3.0" in content
    assert "src" not in content, "src should not appear in requirements"
