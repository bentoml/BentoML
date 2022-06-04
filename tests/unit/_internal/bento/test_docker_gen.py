from __future__ import annotations

import typing as t
import logging
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import patch
from unittest.mock import PropertyMock

import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml._internal.bento.gen import get_template_env
from bentoml._internal.bento.gen import generate_dockerfile
from bentoml._internal.bento.gen import clean_bentoml_version
from bentoml._internal.bento.gen import validate_setup_blocks
from bentoml._internal.bento.docker import DistroSpec
from bentoml._internal.bento.docker import get_supported_spec
from bentoml._internal.bento.build_config import DockerOptions

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.logging import LogCaptureFixture


def get_jinja_environment():
    ...


def test_invalid_spec():
    with pytest.raises(InvalidArgument):
        get_supported_spec("invalid_spec")


@pytest.mark.parametrize("distro", ["debian", "alpine"])
def test_distro_spec(distro: str):
    assert DistroSpec.from_distro(distro)
    assert not DistroSpec.from_distro(None)
    with pytest.raises(BentoMLException):
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
def test_get_template_env_with_default_image(caplog: LogCaptureFixture):
    with pytest.raises(BentoMLException) as excinfo:
        get_template_env(DockerOptions(base_image="bentoml/model-server:latest"), None)
    assert "Distro spec is required" in str(excinfo.value)

    # with cuda
    distro = "ubi8"
    docker = DockerOptions(distro=distro, cuda_version="default").with_defaults()
    spec = DistroSpec.from_distro(distro, cuda=True)
    opts = get_template_env(docker, spec)
    assert opts["__options__distro"] == distro
    assert opts["__options__cuda_version"] == "11.6.2"
    assert not opts["__options__base_image"]
    assert opts["__base_image__"] == "nvidia/cuda:11.6.2-cudnn8-runtime-ubi8"

    with setup_mock_version_info((3, 8, 13)) as version_info:
        distro = "ubi8"
        docker = DockerOptions(distro=distro).with_defaults()
        spec = DistroSpec.from_distro(distro)
        opts = get_template_env(docker, spec)
        assert (
            opts["__base_image__"]
            == f"registry.access.redhat.com/ubi8/python-{version_info.major}{version_info.minor}:1"
        )

    docker = DockerOptions(
        base_image="tensorflow/tensorflow:latest-devel-gpu"
    ).with_defaults()
    spec = DistroSpec.from_distro("debian")
    with caplog.at_level(logging.INFO):
        opts = get_template_env(docker, spec)
    assert (
        "BentoML will not install Python to custom base images; ensure the base image"
        in caplog.text
    )
