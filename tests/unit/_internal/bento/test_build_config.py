from __future__ import annotations

from pathlib import Path

import pytest

from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import BentoEnvSchema
from bentoml._internal.bento.build_config import DockerOptions


@pytest.mark.parametrize("env_format", ["dict", "list", "dotenv"])
def test_merge_bento_and_docker_envs(env_format: str, tmp_path: Path) -> None:
    docker_env = {"DOCKER_ONLY": "docker", "SHARED": "docker"}
    env: dict[str, str] | list[str] | str = docker_env
    if env_format == "list":
        env = [f"{name}={value}" for name, value in docker_env.items()]
    elif env_format == "dotenv":
        env_file = tmp_path / ".env"
        env_file.write_text("DOCKER_ONLY=docker\nSHARED=docker\n")
        env = str(env_file)

    config = BentoBuildConfig.load(
        {
            "service": "service:TestService",
            "envs": [
                {"name": "BENTO_ONLY", "value": "bento"},
                {"name": "SHARED", "value": "bento"},
                {"name": "NAME_ONLY"},
            ],
            "docker": {"env": env},
        }
    )

    filled = config.with_defaults()

    assert filled.docker.env == {
        "BENTO_ONLY": "bento",
        "SHARED": "docker",
        "NAME_ONLY": "",
        "DOCKER_ONLY": "docker",
    }
    assert config.docker.env == docker_env
    assert config.envs == [
        BentoEnvSchema(name="BENTO_ONLY", value="bento"),
        BentoEnvSchema(name="SHARED", value="bento"),
        BentoEnvSchema(name="NAME_ONLY"),
    ]
    assert filled.docker.with_defaults(config.envs) == filled.docker


def test_empty_docker_env_value_overrides_bento_default() -> None:
    options = DockerOptions(env={"SHARED": ""})

    filled = options.with_defaults([BentoEnvSchema(name="SHARED", value="bento")])

    assert filled.env == {"SHARED": ""}


@pytest.mark.parametrize("default_envs", [None, []])
def test_docker_env_without_bento_envs(default_envs) -> None:
    options = DockerOptions(env={"DOCKER_ONLY": "docker"})

    assert options.with_defaults(default_envs).env == {"DOCKER_ONLY": "docker"}


@pytest.mark.parametrize("docker_env", [None, {}, []])
def test_bento_envs_without_docker_env(docker_env) -> None:
    options = DockerOptions(env=docker_env)

    filled = options.with_defaults([BentoEnvSchema(name="BENTO_ONLY", value="bento")])

    assert filled.env == {"BENTO_ONLY": "bento"}
    assert options.env is None


def test_docker_env_merge_preserves_stage_and_build_secret_value() -> None:
    config = BentoBuildConfig.load(
        {
            "envs": [
                {"name": "SHARED", "value": "secret", "stage": "build"},
                {"name": "RUNTIME", "value": "default", "stage": "runtime"},
            ],
            "docker": {"env": {"SHARED": "public", "RUNTIME": "override"}},
        }
    )

    filled = config.with_defaults()

    assert filled.envs == [
        BentoEnvSchema(name="SHARED", value="secret", stage="build"),
        BentoEnvSchema(name="RUNTIME", value="override", stage="runtime"),
        BentoEnvSchema(name="SHARED", value="public"),
    ]
    assert filled.docker.env == {"SHARED": "public", "RUNTIME": "override"}
    assert filled.with_defaults().envs == filled.envs
    assert config.envs[0].value == "secret"
    assert config.envs[1].value == "default"
