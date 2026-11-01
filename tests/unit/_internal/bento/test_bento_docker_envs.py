from __future__ import annotations

from unittest.mock import Mock

import pytest

from _bentoml_sdk.images import Image
from bentoml._internal.bento import Bento
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import PythonOptions
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.container import construct_containerfile
from bentoml.exceptions import BentoMLException


@pytest.mark.parametrize("spec", [1, 2])
def test_empty_docker_defaults_preserve_deployment_requirements(
    tmp_path, monkeypatch, spec
) -> None:
    module = f"docker_env_service_v{spec}"
    (tmp_path / f"{module}.py").write_text(
        "import bentoml\n"
        "@bentoml.service\n"
        "class EnvService:\n"
        "    @bentoml.api\n"
        "    def echo(self, text: str) -> str:\n"
        "        return text\n"
    )
    config = BentoBuildConfig.load(
        {
            "service": f"{module}:EnvService",
            "envs": [
                {"name": "REQUIRED"},
                {"name": "SHARED", "value": "default"},
                {"name": "BUILD_TOKEN", "value": "secret", "stage": "build"},
            ],
            "docker": {"env": {"HTTP_PROXY": "", "SHARED": "", "EXTRA": "extra"}},
        }
    )
    # Keep packaging and Dockerfile generation real while avoiding package downloads.
    monkeypatch.setattr(Image, "_freeze_python_requirements", lambda *args: "")
    monkeypatch.setattr(PythonOptions, "write_to_bento", lambda *args, **kwargs: None)
    bento = Bento.create(
        config,
        build_ctx=str(tmp_path),
        enabled_features=["no_image"] if spec == 1 else [],
    )
    restored = Bento.from_path(bento.path)
    assert restored.info.spec == spec
    monkeypatch.setattr(
        "bentoml._internal.cloud.deployment.ensure_bento", lambda **kwargs: restored
    )
    client = Mock(spec=RestApiClient)

    for cli in (False, True):
        params = DeploymentConfigParameters(
            bento=str(bento.tag),
            envs=[{"name": "REQUIRED", "value": "provided"}],
            cli=cli,
        )
        params.verify(_client=client)
        if cli:
            assert params.envs == [
                {"name": "REQUIRED", "value": "provided"},
                {"name": "SHARED", "value": "default"},
            ]
    with pytest.raises(BentoMLException, match=r"missing: \['REQUIRED'\]"):
        DeploymentConfigParameters(bento=str(bento.tag)).verify(_client=client)

    manifest = restored.get_manifest()
    assert [env.name for env in manifest.envs] == [
        "REQUIRED",
        "SHARED",
        "BUILD_TOKEN",
    ]
    dockerfile = bento.path_of("env/docker/Dockerfile")
    with open(dockerfile) as stream:
        content = stream.read()
    empty_default = "=" if spec == 1 else ""
    assert f"ARG HTTP_PROXY{empty_default}\nENV HTTP_PROXY=$HTTP_PROXY\n" in content
    assert f"ARG SHARED{empty_default}\nENV SHARED=$SHARED\n" in content
    assert "ARG EXTRA=extra\n" in content
    assert "ARG BUILD_TOKEN" not in content
    assert "ENV BUILD_TOKEN" not in content
    if spec == 2:
        assert "--mount=type=secret,id=BUILD_TOKEN" in content

    with construct_containerfile(restored) as (_, dockerfile):
        with open(dockerfile) as stream:
            rebuilt = stream.read()
        assert f"ARG HTTP_PROXY{empty_default}\nENV HTTP_PROXY=$HTTP_PROXY\n" in rebuilt
        assert f"ARG SHARED{empty_default}\nENV SHARED=$SHARED\n" in rebuilt
        assert "ARG EXTRA=extra\n" in rebuilt
        assert "ARG BUILD_TOKEN" not in rebuilt
        assert "ENV BUILD_TOKEN" not in rebuilt

    # Re-reading the packaged config preserves both required declarations and defaults.
    rebuilt_config = BentoBuildConfig.from_file(bento.path_of("src/bentofile.yaml"))
    assert rebuilt_config.with_defaults().envs == manifest.envs
    assert sorted(
        rebuilt_config.with_defaults().get_image_envs(), key=lambda env: env.name
    ) == sorted(config.with_defaults().get_image_envs(), key=lambda env: env.name)
