from __future__ import annotations

import os
import typing as t
import logging
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import patch
from unittest.mock import PropertyMock

import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml._internal.bento.gen import generate_dockerfile
from bentoml._internal.bento.gen import clean_bentoml_version
from bentoml._internal.bento.gen import get_templates_variables
from bentoml._internal.bento.docker import DistroSpec
from bentoml._internal.bento.docker import get_supported_spec
from bentoml._internal.bento.build_config import DockerOptions

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.logging import LogCaptureFixture


def test_invalid_spec():
    with pytest.raises(InvalidArgument):
        get_supported_spec("invalid_spec")  # type: ignore


@pytest.mark.parametrize("distro", ["debian", "alpine"])
def test_distro_spec(distro: str):
    assert DistroSpec.from_distro(distro)
    with pytest.raises(BentoMLException):
        DistroSpec.from_distro("rocky")
        DistroSpec.from_distro("invalid_distro")


@pytest.mark.parametrize(
    "version, expected",
    [
        ("1.0.0rc0.post234", "1.0.0rc0"),
        ("1.2.3a8+dev123", "1.2.3a8"),
        ("2.23.0.dev+post123", "2.23.0"),
    ],
)
def test_clean_bentoml_version(version: str, expected: str):
    assert clean_bentoml_version(version) == expected
    with pytest.raises(BentoMLException) as excinfo:
        clean_bentoml_version("1233")
    assert "Errors while parsing BentoML version" in str(excinfo.value)


if TYPE_CHECKING:

    class version_info:
        major: int
        minor: int
        patch: int

    @contextlib.contextmanager
    def setup_mock_version_info(
        version: tuple[int, int, int]
    ) -> t.Generator[version_info, None, None]:
        ...

else:

    @contextlib.contextmanager
    def setup_mock_version_info(
        version: tuple[int, int, int],
    ) -> t.Generator[MagicMock, None, None]:
        patcher = patch("bentoml._internal.bento.build_config.version_info")
        mock_version_info = patcher.start()
        try:
            type(mock_version_info).major = PropertyMock(return_value=version[0])
            type(mock_version_info).minor = PropertyMock(return_value=version[1])
            type(mock_version_info).micro = PropertyMock(return_value=version[2])
            yield mock_version_info
        finally:
            patcher.stop()


@pytest.mark.usefixtures("propagate_logs")
def test_get_templates_variables_with_default_image(caplog: LogCaptureFixture):

    # with cuda
    distro = "ubi8"
    docker = DockerOptions(distro=distro, cuda_version="default").with_defaults()
    opts = get_templates_variables(docker, use_conda=False)
    assert opts["__options__distro"] == distro
    assert opts["__options__cuda_version"] == "11.6.2"
    assert not opts["__options__base_image"]
    assert opts["__base_image__"] == "nvidia/cuda:11.6.2-cudnn8-runtime-ubi8"

    with setup_mock_version_info((3, 8, 13)) as version_info:
        distro = "ubi8"
        docker = DockerOptions(distro=distro).with_defaults()
        opts = get_templates_variables(docker, use_conda=False)
        assert (
            opts["__base_image__"]
            == f"registry.access.redhat.com/ubi8/python-{version_info.major}{version_info.minor}:1"
        )

    docker = DockerOptions(
        base_image="tensorflow/tensorflow:latest-devel-gpu"
    ).with_defaults()
    with caplog.at_level(logging.INFO):
        opts = get_templates_variables(docker, use_conda=False)
    assert (
        "BentoML will not install Python to custom base images; ensure the base image"
        in caplog.text
    )


def test_generate_dockerfile_options_call():
    with pytest.raises(BentoMLException) as excinfo:
        generate_dockerfile(DockerOptions(), build_ctx=".", use_conda=False)
    assert "Distro name is required, got None instead." in str(excinfo.value)


@pytest.mark.usefixtures("change_test_dir")
def test_generate_dockerfile_with_base_image():
    docker = DockerOptions(distro="ubi8", cuda_version="default").with_defaults()
    res = generate_dockerfile(docker, build_ctx=".", use_conda=False)
    assert "FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubi8" in res


@pytest.mark.usefixtures("change_test_dir")
def test_generate_dockerfile():
    res = generate_dockerfile(
        DockerOptions(
            dockerfile_template=os.path.join(
                "testdata", "configuration", "Dockerfile.template"
            )
        ).with_defaults(),
        build_ctx=".",
        use_conda=False,
    )
    assert "# syntax = docker/dockerfile:1.4-labs\n#\n" in res
    assert (
        "FROM --platform=$BUILDPLATFORM python:3.7-slim as buildstage\n\nRUN mkdir /tmp/mypackage"
        in res
    )
