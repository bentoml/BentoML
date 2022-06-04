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

    from fs.base import FS
    from _pytest.logging import LogCaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch
    from _pytest.mark.structures import MarkDecorator


@pytest.fixture(scope="function", name="test_fs")
def fixture_test_fs(
    tmpdir: Path, request: FixtureRequest
) -> t.Generator[FS, None, None]:
    os.chdir(request.fspath.dirname)  # type: ignore (bad pytest stubs)
    yield fs.open_fs(tmpdir.__fspath__())
    os.chdir(request.config.invocation_dir)  # type: ignore (bad pytest stubs)


@pytest.mark.parametrize("python_version", [3.7, "3.7.9"])
def test_convert_python_version(python_version: str):
    from bentoml._internal.bento.build_config import (
        _convert_python_version,  # type: ignore
    )

    assert _convert_python_version(python_version) == "3.7"
    assert not _convert_python_version(None)

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
    assert not _convert_cuda_version(None)
    assert not _convert_cuda_version("")


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

    assert not _convert_user_envars(None)
    assert _convert_user_envars(envars) == data


def parametrize_options(
    options_cls: t.Type[DockerOptions | CondaOptions | PythonOptions],
    test_type: t.Literal["valid", "raisewarning", "invalid", "structure"],
    *,
    attribute: str | None = None,
    _argvalue: str | None = None,
) -> MarkDecorator:
    if test_type not in ["valid", "invalid", "raisewarning", "structure"]:
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
            testdata.append(yaml.safe_load(f))

    if not _argvalue:
        _argvalue = "options"

    return pytest.mark.parametrize(str(_argvalue), testdata)


@pytest.mark.incremental
class TestDockerOptions:
    @parametrize_options(DockerOptions, "valid")
    def test_valid_docker_options(self, options: dict[str, t.Any]):
        assert DockerOptions(**options)

    @parametrize_options(DockerOptions, "raisewarning")
    def test_raises_warning_docker_options(
        self, options: dict[str, t.Any], caplog: LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert DockerOptions(**options)
        assert "option is ignored" in caplog.text

    def test_invalid_cuda_supports_distro(self):
        with pytest.raises(BentoMLException):
            DockerOptions(distro="alpine", cuda_version="11.6.2")

    @parametrize_options(DockerOptions, "invalid", attribute="env")
    def test_invalid_docker_envars(self, options: dict[str, t.Any]):
        with pytest.raises(InvalidArgument):
            DockerOptions(**options)

    @parametrize_options(DockerOptions, "raisewarning", attribute="env")
    def test_raises_warning_envar(
        self, options: dict[str, t.Any], caplog: LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert DockerOptions(**options)
        assert "dict contains None value" in caplog.text

    @parametrize_options(DockerOptions, "invalid")
    def test_invalid_options(self, options: dict[str, t.Any]):
        with pytest.raises(ValueError):
            DockerOptions(**options)

    @parametrize_options(DockerOptions, "invalid", attribute="cuda_version")
    def test_invalid_cuda_version(self, options: dict[str, t.Any]):
        with pytest.raises(BentoMLException):
            DockerOptions(**options)

    @parametrize_options(DockerOptions, "valid", attribute="distro")
    def test_with_defaults_distro(self, options: dict[str, t.Any]):
        from bentoml._internal.bento.build_config import DEFAULT_DOCKER_DISTRO

        filled = DockerOptions(**options).with_defaults()
        assert filled.distro == DEFAULT_DOCKER_DISTRO
        assert not filled.dockerfile_template

    @pytest.mark.usefixtures("test_fs")
    @patch("bentoml._internal.configuration.is_pypi_installed_bentoml")
    @pytest.mark.parametrize(
        "conda_option", [CondaOptions(), CondaOptions(channels=["default"])]
    )
    def test_write_to_bento(
        self,
        mock_is_pypi_installed_bentoml: MagicMock,
        conda_option: CondaOptions,
        test_fs: FS,
        monkeypatch: MonkeyPatch,
    ):
        from bentoml._internal.bento.build_dev_bentoml_whl import BENTOML_DEV_BUILD

        monkeypatch.setenv(BENTOML_DEV_BUILD, str(False))
        mock_is_pypi_installed_bentoml.return_value = True

        DockerOptions().with_defaults().write_to_bento(
            test_fs, os.getcwd(), conda_option.with_defaults()
        )
        docker_fs = test_fs.opendir("/env/docker")

        assert os.path.isdir(docker_fs.getsyspath("/"))
        assert docker_fs.exists("Dockerfile")
        assert docker_fs.exists("entrypoint.sh")
        assert not os.path.exists(docker_fs.getsyspath("/whl"))

        DockerOptions(
            setup_script="./testdata/scripts/setup_script"
        ).with_defaults().write_to_bento(
            test_fs, os.getcwd(), conda_option.with_defaults()
        )

        assert docker_fs.exists("setup_script")

        with pytest.raises(InvalidArgument):
            DockerOptions(
                setup_script="./path/does/not/exist"
            ).with_defaults().write_to_bento(
                test_fs, os.getcwd(), conda_option.with_defaults()
            )

    @pytest.mark.usefixtures("change_test_dir")
    def test_structure_docker_options(self):
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


@pytest.mark.incremental
class TestCondaOptions:
    @parametrize_options(CondaOptions, "valid")
    def test_valid_conda_options(self, options: dict[str, t.Any]):
        assert CondaOptions(**options)

    @parametrize_options(CondaOptions, "invalid")
    def test_invalid_conda_options(self, options: dict[str, t.Any]):
        with pytest.raises(TypeError) as excinfo:
            CondaOptions(**options)
            assert "must be <class 'list'>" in str(excinfo.value)

    @parametrize_options(CondaOptions, "invalid", attribute="dependencies")
    def test_dependencies_validator(self, options: dict[str, t.Any]):
        with pytest.raises(InvalidArgument):
            CondaOptions(**options)

    @pytest.mark.usefixtures("change_test_dir")
    @parametrize_options(CondaOptions, "raisewarning")
    def test_raises_warning_conda_options(
        self, options: dict[str, t.Any], caplog: LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert CondaOptions(**options)
        assert "option is ignored" in caplog.text

    @pytest.mark.usefixtures("test_fs")
    def test_write_to_bento(self, test_fs: FS):
        CondaOptions().with_defaults().write_to_bento(test_fs, os.getcwd())

        conda_dir = test_fs.opendir("/env/conda")
        assert os.path.isdir(conda_dir.getsyspath("/"))
        assert not os.path.exists(conda_dir.getsyspath("/environment.yml"))

        CondaOptions(
            dependencies=["numpy"], pip=["aiohttp"]
        ).with_defaults().write_to_bento(test_fs, os.getcwd())
        assert os.path.exists(conda_dir.getsyspath("/environment.yml"))
        with open(conda_dir.getsyspath("/environment.yml"), "r") as f:
            assert yaml.safe_load(f) == {
                "channels": ["defaults"],
                "dependencies": ["numpy", {"pip": ["aiohttp"]}],
            }

        CondaOptions(
            environment_yml="./testdata/configuration/example_environment.yml"
        ).with_defaults().write_to_bento(test_fs, os.getcwd())

        assert os.path.exists(conda_dir.getsyspath("/environment.yml"))
        with open(conda_dir.getsyspath("/environment.yml"), "r") as f1, open(
            "./testdata/configuration/example_environment.yml", "r"
        ) as f2:
            assert yaml.safe_load(f1) == yaml.safe_load(f2)

    @pytest.mark.usefixtures("change_test_dir")
    @parametrize_options(CondaOptions, "structure", _argvalue="structure_data")
    def test_structure_conda_options(self, structure_data: dict[str, t.Any]):
        data = bentoml_cattr.structure(structure_data, CondaOptions)
        assert data == CondaOptions(**structure_data)

        with pytest.raises(InvalidArgument) as excinfo:
            bentoml_cattr.structure({"dependencies": "not a list"}, CondaOptions)
            assert "type list" in str(excinfo.value)


@pytest.mark.incremental
class TestPythonOptions:
    @parametrize_options(PythonOptions, "valid")
    def test_valid_python_options(self, options: dict[str, t.Any]):
        assert PythonOptions(**options)

    @parametrize_options(PythonOptions, "invalid")
    def test_invalid_python_options(self, options: dict[str, t.Any]):
        with pytest.raises(TypeError) as excinfo:
            PythonOptions(**options)
            assert "must be <class" in str(excinfo.value)

    @pytest.mark.usefixtures("change_test_dir")
    @parametrize_options(PythonOptions, "raisewarning")
    def test_raises_warning_python_options(
        self, options: dict[str, t.Any], caplog: LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert PythonOptions(**options)
        assert "Build option python" in caplog.text

    @pytest.mark.usefixtures("test_fs")
    def test_write_to_bento(self, test_fs: FS):
        PythonOptions().with_defaults().write_to_bento(test_fs, os.getcwd())

        python_dir = test_fs.opendir("/env/python")
        assert os.path.isdir(python_dir.getsyspath("/"))
        assert not os.path.isdir(python_dir.getsyspath("/wheels"))
