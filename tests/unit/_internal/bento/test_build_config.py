from __future__ import annotations

import os
import re
import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from unittest.mock import patch

import fs
import yaml
import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml._internal.utils import bentoml_cattr
from bentoml._internal.bento.build_config import CondaOptions
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.bento.build_config import PythonOptions
from bentoml._internal.bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    ListFactory = list[str]
    DictFactory = dict[str, t.Any]
else:
    ListFactory = list
    DictFactory = dict

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.parametrize("python_version", [3.7, "3.7.9"])
def test_convert_python_version(python_version: str):
    from bentoml._internal.bento.build_config import (
        _convert_python_version,  # type: ignore
    )

    assert _convert_python_version(python_version) == "3.7"
    assert _convert_python_version(None) == None

    with pytest.raises(InvalidArgument):
        _convert_python_version("3.6")
        _convert_python_version("3")
        _convert_python_version("")


@pytest.mark.parametrize("cuda_version", [11, "default", "11.6.2"])
def test_convert_cuda_version(cuda_version: str):
    # no need to check for 11.6, 10.2 cases here since validation from attrs
    from bentoml._internal.bento.build_config import (
        _convert_cuda_version,  # type: ignore
    )

    assert _convert_cuda_version(cuda_version) == "11.6.2"
    assert _convert_cuda_version(None) == None
    assert _convert_cuda_version("") == None


def make_envars(factory: type, data: dict[str, str]):
    if factory == list:
        return [f"{k}={v}" for k, v in data.items()]
    elif factory == dict:
        return data
    else:
        raise ValueError(f"Unknown factory {factory}")


@pytest.mark.parametrize("factory", [ListFactory, DictFactory])
def test_convert_envars(factory: type):
    from bentoml._internal.bento.build_config import (
        _convert_user_envars,  # type: ignore
    )

    data = {"TEST": "hello", "FOO": "bar"}
    envars = make_envars(factory, data)

    assert _convert_user_envars(None) == None
    assert _convert_user_envars(envars) == data


def create_parametrized_opts(
    options_cls: t.Type[DockerOptions | CondaOptions | PythonOptions],
    test_type: t.Literal["valid", "raisewarning", "invalid"],
    *,
    attribute: str | None = None,
) -> t.Iterable[dict[str, t.Any]]:
    if test_type not in ["valid", "invalid", "raisewarning"]:
        raise InvalidArgument(f"Unknown test type {test_type}")

    if attribute is not None:
        assert hasattr(options_cls.__attrs_attrs__, attribute)
        # attribute has type attr.Attribute[t.Any]
        attr_name: str = getattr(options_cls.__attrs_attrs__, attribute).name
        rgx = re.compile(f"^{options_cls.__name__.lower()}_{test_type}_{attr_name}*")
    else:
        rgx = re.compile(rf"^{options_cls.__name__.lower()}_{test_type}\d+")

    testdata_dir = Path(__file__).parent / "testdata"
    testdata: list[dict[str, t.Any]] = []

    for data in [f for f in testdata_dir.iterdir() if rgx.match(f.name)]:
        with open(data, "r") as f:
            data = yaml.safe_load(f)
            print(data)
            testdata.append(data)

    return testdata


@pytest.mark.usefixtures("propagate_logs")
@pytest.mark.incremental
class TestDockerOptions:
    @pytest.mark.parametrize("opts", create_parametrized_opts(DockerOptions, "valid"))
    def test_valid_docker_options(self, opts: dict[str, t.Any]):
        assert DockerOptions(**opts)

    @pytest.mark.parametrize(
        "opts", create_parametrized_opts(DockerOptions, "raisewarning")
    )
    def test_raises_warning_docker_options(
        self, opts: dict[str, t.Any], caplog: LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert DockerOptions(**opts)
        assert "option is ignored" in caplog.text

    def test_invalid_cuda_supports_distro(self):
        with pytest.raises(BentoMLException):
            DockerOptions(distro="alpine", cuda_version="11.6.2")

    @pytest.mark.parametrize(
        "opts",
        create_parametrized_opts(
            DockerOptions,
            "invalid",
            attribute="env",
        ),
    )
    def test_invalid_docker_envars(self, opts: dict[str, t.Any]):
        with pytest.raises(InvalidArgument):
            DockerOptions(**opts)

    @pytest.mark.parametrize(
        "opts",
        create_parametrized_opts(
            DockerOptions,
            "raisewarning",
            attribute="env",
        ),
    )
    def test_raises_warning_envar(
        self, opts: dict[str, t.Any], caplog: LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert DockerOptions(**opts)
        assert "dict contains None value" in caplog.text

    @pytest.mark.parametrize("opts", create_parametrized_opts(DockerOptions, "invalid"))
    def test_invalid_options(self, opts: dict[str, t.Any]):
        with pytest.raises(ValueError):
            DockerOptions(**opts)

    @pytest.mark.parametrize(
        "opts",
        create_parametrized_opts(DockerOptions, "invalid", attribute="cuda_version"),
    )
    def test_invalid_cuda_version(self, opts: dict[str, t.Any]):
        with pytest.raises(BentoMLException):
            DockerOptions(**opts)

    @pytest.mark.parametrize(
        "opts", create_parametrized_opts(DockerOptions, "valid", attribute="distro")
    )
    def test_with_defaults_distro(self, opts: dict[str, t.Any]):
        from bentoml._internal.bento.build_config import DEFAULT_DOCKER_DISTRO

        filled = DockerOptions(**opts).with_defaults()
        assert filled.distro == DEFAULT_DOCKER_DISTRO
        assert filled.dockerfile_template == None

    @pytest.mark.usefixtures("change_test_dir")
    @patch("bentoml._internal.configuration.is_pypi_installed_bentoml")
    @pytest.mark.parametrize(
        "conda_option", [CondaOptions(), CondaOptions(channels=["default"])]
    )
    def test_write_to_bento(
        self,
        mock_is_pypi_installed_bentoml: MagicMock,
        conda_option: CondaOptions,
        tmpdir: Path,
        monkeypatch: MonkeyPatch,
    ):
        from bentoml._internal.bento.build_dev_bentoml_whl import BENTOML_DEV_BUILD

        monkeypatch.setenv(BENTOML_DEV_BUILD, str(False))
        mock_is_pypi_installed_bentoml.return_value = True

        test_fs = fs.open_fs(tmpdir.__fspath__())
        DockerOptions().with_defaults().write_to_bento(
            test_fs, os.getcwd(), conda_option.with_defaults()
        )

        docker_dir = tmpdir / "env" / "docker"
        assert os.path.isdir(docker_dir)
        assert os.path.isfile(docker_dir / "entrypoint.sh")
        assert not os.path.exists(docker_dir / "whl")

        DockerOptions(
            setup_script="./testdata/scripts/setup_script"
        ).with_defaults().write_to_bento(
            test_fs, os.getcwd(), conda_option.with_defaults()
        )

        assert os.path.isfile(docker_dir / "setup_script")

        with pytest.raises(InvalidArgument):
            DockerOptions(
                setup_script="./path/does/not/exist"
            ).with_defaults().write_to_bento(
                test_fs, os.getcwd(), conda_option.with_defaults()
            )

    @pytest.mark.usefixtures("change_test_dir")
    def test_structure_docker_options(self, tmpdir: Path):
        with_env_options = {"distro": "debian", "env": "./testdata/scripts/dot_env"}
        assert bentoml_cattr.structure(
            with_env_options, DockerOptions
        ) == DockerOptions(**with_env_options)

        with pytest.raises(BentoMLException) as excinfo:
            bentoml_cattr.structure(
                {"distro": "alpine", "env": "./testdata/script/doesnotexist"},
                DockerOptions,
            )
            assert "Invalid env file path" in str(excinfo.value)
