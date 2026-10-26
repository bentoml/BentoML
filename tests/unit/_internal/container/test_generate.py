from __future__ import annotations

from io import StringIO

from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import CondaOptions
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.container.generate import build_environment
from bentoml._internal.container.generate import generate_containerfile
from bentoml._internal.container.generate import normalize_line


def test_normalize_line_collapses_whitespace() -> None:
    assert normalize_line("  python:3.11-slim\n\tAS injected  ") == (
        "python:3.11-slim AS injected"
    )


def test_build_environment_registers_normalize_line_filter() -> None:
    environment = build_environment()

    assert environment.filters["normalize_line"]("  python:3.11-slim\n  ") == (
        "python:3.11-slim"
    )


def test_generate_containerfile_quotes_system_packages(tmp_path) -> None:
    dockerfile = generate_containerfile(
        DockerOptions(
            distro="debian",
            python_version="3.11",
            system_packages=["libpq-dev", "package name", "foo$(touch /tmp/pwned)"],
        ),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )

    assert "libpq-dev 'package name' 'foo$(touch /tmp/pwned)'" in dockerfile


def test_generate_containerfile_cache_mounts_use_sharing_locked(tmp_path) -> None:
    import re

    dockerfile = generate_containerfile(
        DockerOptions(
            distro="debian",
            python_version="3.11",
        ),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )
    cache_mounts = re.findall(r"--mount=type=cache[^ ]+", dockerfile)
    assert len(cache_mounts) > 0, "expected at least one cache mount"
    for mount in cache_mounts:
        assert "sharing=locked" in mount, f"cache mount missing sharing=locked: {mount}"


def test_generate_containerfile_normalizes_custom_base_image(tmp_path) -> None:
    dockerfile = generate_containerfile(
        DockerOptions(base_image="  python:3.11-slim\nRUN touch /tmp/pwned  "),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )

    assert "FROM python:3.11-slim RUN touch /tmp/pwned as base-container" in dockerfile
    assert "\nRUN touch /tmp/pwned" not in dockerfile


def test_generate_containerfile_merges_bento_and_docker_envs(tmp_path) -> None:
    config = BentoBuildConfig.from_yaml(
        StringIO(
            """\
service: service:TestService
envs:
  - name: TEST_ENV
    value: test
docker:
  distro: debian
  python_version: "3.10"
  env:
    TEST_ENV_2: test
"""
        )
    ).with_defaults()

    dockerfile = generate_containerfile(
        config.docker,
        str(tmp_path),
        conda=config.conda,
        bento_fs=tmp_path,
    )

    assert "ARG TEST_ENV=test\nENV TEST_ENV=$TEST_ENV\n" in dockerfile
    assert "ARG TEST_ENV_2=test\nENV TEST_ENV_2=$TEST_ENV_2\n" in dockerfile


def test_generate_containerfile_installs_hash_pinned_uv(tmp_path) -> None:
    dockerfile = generate_containerfile(
        DockerOptions(distro="debian", python_version="3.11"),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )

    assert "pip install uv\n" not in dockerfile
    assert "uv==0.13.4" in dockerfile
    assert "pip install --no-cache-dir --require-hashes" in dockerfile


def test_generate_containerfile_keeps_build_secrets_out_of_env(tmp_path) -> None:
    config = BentoBuildConfig.load(
        {
            "service": "service:TestService",
            "envs": [
                {"name": "BUILD_TOKEN", "value": "synthetic-secret", "stage": "build"},
                {"name": "PUBLIC", "value": "public"},
            ],
            "docker": {"env": {"DOCKER_ONLY": "docker"}},
        }
    ).with_defaults()

    dockerfile = generate_containerfile(
        config.docker, str(tmp_path), conda=config.conda, bento_fs=tmp_path
    )

    assert "BUILD_TOKEN" not in dockerfile
    assert "synthetic-secret" not in dockerfile
    assert "ARG PUBLIC=public" in dockerfile
    assert "ARG DOCKER_ONLY=docker" in dockerfile


def test_image_freeze_merges_docker_envs_and_preserves_stages(
    tmp_path, monkeypatch
) -> None:
    from _bentoml_sdk.images import Image
    from _bentoml_sdk.images import populate_image_from_build_config

    config = BentoBuildConfig.load(
        {
            "service": "service:TestService",
            "envs": [
                {"name": "BUILD_TOKEN", "value": "synthetic-secret", "stage": "build"},
                {"name": "PUBLIC", "value": "public"},
                {"name": "SHARED", "value": "default"},
                {"name": "RUNTIME_ONLY", "value": "runtime", "stage": "runtime"},
            ],
            "docker": {"env": {"DOCKER_ONLY": "docker", "SHARED": ""}},
        }
    )
    image = populate_image_from_build_config(None, config, str(tmp_path))
    assert image is not None
    monkeypatch.setattr(Image, "_freeze_python_requirements", lambda *args: "")

    image.freeze(tmp_path, config.with_defaults().envs)
    dockerfile = (tmp_path / "env/docker/Dockerfile").read_text()

    assert "ARG PUBLIC=public" in dockerfile
    assert "ARG DOCKER_ONLY=docker" in dockerfile
    assert "ARG SHARED\n" in dockerfile
    assert "ARG SHARED=default" not in dockerfile
    assert "ARG RUNTIME_ONLY" not in dockerfile
    assert "--mount=type=secret,id=BUILD_TOKEN" in dockerfile
    assert "ARG BUILD_TOKEN" not in dockerfile
    assert "ENV BUILD_TOKEN" not in dockerfile
    assert "synthetic-secret" not in dockerfile
