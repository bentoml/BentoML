from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from _bentoml_sdk.images import Image
from bentoml._internal.bento import Bento
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import BentoEnvSchema
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
                {
                    "name": "RUNTIME_TOKEN",
                    "value": "runtime-default",
                    "stage": "runtime",
                },
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
                {"name": "RUNTIME_TOKEN", "value": "runtime-default"},
            ]
    with pytest.raises(BentoMLException, match=r"missing: \['REQUIRED'\]"):
        DeploymentConfigParameters(bento=str(bento.tag)).verify(_client=client)

    manifest = restored.get_manifest()
    assert [env.name for env in manifest.envs] == [
        "REQUIRED",
        "SHARED",
        "BUILD_TOKEN",
        "RUNTIME_TOKEN",
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
    assert "RUNTIME_TOKEN" not in content
    assert "runtime-default" not in content
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
        assert "RUNTIME_TOKEN" not in rebuilt
        assert "runtime-default" not in rebuilt

    # Re-reading the packaged config preserves both required declarations and defaults.
    rebuilt_config = BentoBuildConfig.from_file(bento.path_of("src/bentofile.yaml"))
    assert rebuilt_config.with_defaults().envs == manifest.envs
    assert sorted(
        rebuilt_config.with_defaults().get_image_envs(), key=lambda env: env.name
    ) == sorted(config.with_defaults().get_image_envs(), key=lambda env: env.name)


@pytest.mark.parametrize("docker_value", [None, "explicit-build", ""])
@pytest.mark.parametrize("spec", [1, 2])
def test_service_runtime_override_stays_out_of_image(
    tmp_path, monkeypatch, docker_value, spec
):
    module = f"staged_service_{spec}_{docker_value is None}_{bool(docker_value)}"
    (tmp_path / f"{module}.py").write_text(
        "import bentoml\n"
        '@bentoml.service(envs=[{"name": "SHARED", "value": "runtime-only", "stage": "runtime"}])\n'
        "class StagedService:\n"
        "    @bentoml.api\n"
        "    def echo(self, text: str) -> str:\n"
        "        return text\n"
    )
    config = BentoBuildConfig.load(
        {
            "service": f"{module}:StagedService",
            "envs": [{"name": "SHARED", "value": "build-default", "stage": "all"}],
            "docker": {"env": {} if docker_value is None else {"SHARED": docker_value}},
        }
    )
    monkeypatch.setattr(Image, "_freeze_python_requirements", lambda *args: "")
    monkeypatch.setattr(PythonOptions, "write_to_bento", lambda *args, **kwargs: None)
    bento = Bento.create(
        config,
        build_ctx=str(tmp_path),
        enabled_features=["no_image"] if spec == 1 else [],
    )
    restored = Bento.from_path(bento.path)
    assert restored.info.spec == spec
    build_value = "build-default" if docker_value is None else docker_value
    assert restored.info.envs == [
        BentoEnvSchema(name="SHARED", value="build-default"),
        BentoEnvSchema(name="SHARED", value="runtime-only", stage="runtime"),
    ]
    if spec == 2:
        assert restored.info.image_envs == [
            BentoEnvSchema(name="SHARED", value=build_value),
            BentoEnvSchema(
                name="SHARED",
                value="runtime-only" if docker_value is None else docker_value,
                stage="runtime",
            ),
        ]
    packaged_config = BentoBuildConfig.from_file(bento.path_of("src/bentofile.yaml"))
    assert packaged_config.envs == restored.info.envs
    if spec == 2:
        assert packaged_config.get_image_envs() == restored.info.image_envs
    expected_arg = (
        f"ARG SHARED{'=' + build_value if build_value or spec == 1 else ''}\n"
    )
    with open(bento.path_of("env/docker/Dockerfile")) as stream:
        generated = stream.read()
    assert expected_arg in generated
    assert "runtime-only" not in generated
    with construct_containerfile(restored) as (_, dockerfile):
        with open(dockerfile) as stream:
            rebuilt = stream.read()
        assert expected_arg in rebuilt
        assert "runtime-only" not in rebuilt


@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("external_value", [None, "external", ""])
@pytest.mark.parametrize("docker_env", [False, True])
def test_legacy_entrypoint_restores_runtime_defaults(
    tmp_path, monkeypatch, legacy, external_value, docker_env
):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to execute the container entrypoint")
    module = f"runtime_defaults_{legacy}_{external_value is None}_{bool(external_value)}_{docker_env}"
    service = (
        'svc = bentoml.legacy.Service("runtime_defaults")\n'
        if legacy
        else "@bentoml.service\nclass RuntimeDefaults:\n    pass\n"
    )
    (tmp_path / f"{module}.py").write_text("import bentoml\n" + service)
    value = "production 'quoted' $HOME $(touch injected) `touch injected`\n第二行"
    config = BentoBuildConfig.load(
        {
            "service": f"{module}:{'svc' if legacy else 'RuntimeDefaults'}",
            "envs": [
                {"name": "BENTO_TEST_RUNTIME", "value": value, "stage": "runtime"},
                {"name": "BENTO_TEST_DOCKER", "value": "default", "stage": "runtime"},
                {"name": "BENTO_TEST_EMPTY", "value": "default", "stage": "runtime"},
                {"name": "BENTO_TEST_BUILD", "value": "secret", "stage": "build"},
                {"name": "BENTO_TEST_REQUIRED", "stage": "runtime"},
            ],
            "docker": {
                "env": {"BENTO_TEST_DOCKER": "docker", "BENTO_TEST_EMPTY": ""}
                if docker_env
                else {}
            },
        }
    )
    monkeypatch.setattr(PythonOptions, "write_to_bento", lambda *args, **kwargs: None)
    bento = Bento.create(config, build_ctx=str(tmp_path), enabled_features=["no_image"])
    restored = Bento.from_path(bento.path)
    assert restored.info.spec == 1
    keys = [env.name for env in config.envs]
    environment = {key: val for key, val in os.environ.items() if key not in keys}
    # Docker supplies its ENV values before starting the packaged entrypoint.
    environment.update(restored.info.docker.env or {})
    if external_value is not None:
        environment["BENTO_TEST_RUNTIME"] = external_value

    def check_entrypoint(context):
        environment["BENTO_PATH"] = Path(context).as_posix()
        output = subprocess.check_output(
            [
                bash,
                Path(context, "env/docker/entrypoint.sh").as_posix(),
                Path(sys.executable).as_posix(),
                "-c",
                f"import json, os; print(json.dumps({{key: os.environ.get(key) for key in {keys!r}}}))",
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
        )
        assert json.loads(output) == {
            "BENTO_TEST_RUNTIME": value if external_value is None else external_value,
            "BENTO_TEST_DOCKER": "docker" if docker_env else "default",
            "BENTO_TEST_EMPTY": "" if docker_env else "default",
            "BENTO_TEST_BUILD": None,
            "BENTO_TEST_REQUIRED": None,
        }
        assert not (tmp_path / "injected").exists()

    check_entrypoint(restored.path)
    with construct_containerfile(restored) as (context, dockerfile):
        content = Path(dockerfile).read_text()
        assert "BENTO_TEST_RUNTIME" not in content
        assert "BENTO_TEST_BUILD" not in content
        assert value not in content
        check_entrypoint(context)
