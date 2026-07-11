from __future__ import annotations

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


def test_generate_containerfile_sanitizes_env_dict_values(tmp_path) -> None:
    dockerfile = generate_containerfile(
        DockerOptions(
            distro="debian",
            python_version="3.11",
            env={"X": "a\nRUN echo PWNED_VIA_ENV_INJECTION", "GREETING": "hello world"},
        ),
        str(tmp_path),
        conda=CondaOptions(),
        bento_fs=tmp_path,
    )

    # A newline in an env value must not break out into a new Dockerfile instruction.
    assert "\nRUN echo PWNED_VIA_ENV_INJECTION" not in dockerfile
    # The payload is neutralized inside a single-quoted ARG value.
    assert "ARG X='a RUN echo PWNED_VIA_ENV_INJECTION'" in dockerfile
    # Legitimate multi-word values are preserved and quoted.
    assert "ARG GREETING='hello world'" in dockerfile


def test_generate_dockerfile_sanitizes_envs_values(tmp_path) -> None:
    from _bentoml_impl.docker import generate_dockerfile
    from bentoml._internal.bento.bento import ImageInfo
    from bentoml._internal.bento.build_config import BentoEnvSchema

    image = ImageInfo(python_version="3.11", base_image="python:3.11-slim")
    dockerfile = generate_dockerfile(
        image,
        tmp_path,
        envs=[BentoEnvSchema(name="X", value="a\nRUN echo PWNED_V2", stage="all")],
    )

    # The newline must not break out into a new Dockerfile instruction.
    assert "\nRUN echo PWNED_V2" not in dockerfile
    # The payload is neutralized inside a single-quoted ARG value.
    assert "ARG X='a RUN echo PWNED_V2'" in dockerfile
