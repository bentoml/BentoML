from __future__ import annotations

import os
import re
import typing as t
import logging
from io import BytesIO
from sys import version_info
from uuid import uuid4
from typing import TYPE_CHECKING
from pathlib import Path
from unittest.mock import patch

import fs
import attr
import yaml
import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml._internal.utils import bentoml_cattr
from bentoml._internal.bento.build_config import CondaOptions
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.bento.build_config import PythonOptions
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.bento.build_config import DEFAULT_DOCKER_DISTRO

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
    tmppath = Path(tmpdir, uuid4().hex)
    tmppath.mkdir(parents=True, exist_ok=True)
    yield fs.open_fs(tmppath.__fspath__())
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
    options_cls: t.Type[
        DockerOptions | CondaOptions | PythonOptions | BentoBuildConfig
    ],
    test_type: t.Literal["valid", "raisewarning", "invalid", "structure", "build"],
    *,
    attribute: str | None = None,
    _argvalue: str | None = None,
) -> MarkDecorator:
    if not hasattr(options_cls, "__attrs_attrs__"):
        raise ValueError(f"{options_cls} is not an attrs class")
    if test_type not in ["valid", "invalid", "raisewarning", "structure", "build"]:
        raise InvalidArgument(f"Unknown test type {test_type}")

    if attribute is not None:
        if attribute not in [a.name for a in attr.fields(options_cls)]:
            raise InvalidArgument(f"{options_cls} has no attribute {attribute}")
        rgx = re.compile(f"^{options_cls.__name__.lower()}_{test_type}_{attribute}*")
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


def fs_identical(fs1: FS, fs2: FS):
    for path in fs1.walk.dirs():
        assert fs2.isdir(path)

    for path in fs1.walk.files():
        assert fs2.isfile(path)
        assert fs1.readbytes(path) == fs2.readbytes(path)


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
        assert PythonOptions().with_defaults().lock_packages

        PythonOptions(lock_packages=False).with_defaults().write_to_bento(
            test_fs, os.getcwd()
        )

        python_dir = test_fs.opendir("/env/python")
        assert os.path.isdir(python_dir.getsyspath("/"))
        assert not os.path.isdir(python_dir.getsyspath("/wheels"))

        with open(python_dir.getsyspath("/version.txt"), "r") as f:
            assert (
                f.read()
                == f"{version_info.major}.{version_info.minor}.{version_info.micro}"
            )
        PythonOptions(
            packages=["numpy", "pandas"], lock_packages=False
        ).with_defaults().write_to_bento(test_fs, os.getcwd())

        assert python_dir.exists("requirements.txt")

    @pytest.mark.usefixtures("test_fs")
    def test_wheels_include(self, test_fs: FS):
        try:
            wheel_fs = fs.open_fs("./testdata/wheels/", create=True)
            wheel_fs.touch("test.whl")
            wheel_fs.touch("another.whl")

            PythonOptions(
                wheels=[f"./testdata/wheels/{file}" for file in wheel_fs.listdir("/")],
                requirements_txt="./testdata/configuration/example_requirements.txt",
            ).with_defaults().write_to_bento(test_fs, os.getcwd())

            python_dir = test_fs.opendir("/env/python")
            assert os.path.isdir(python_dir.getsyspath("/wheels"))
            fs_identical(test_fs.opendir("/env/python/wheels"), wheel_fs)
            with python_dir.open("requirements.txt", "r") as f1, open(  # type: ignore
                "./testdata/configuration/example_requirements.txt", "r"
            ) as f2:
                assert f1.read() == f2.read()
        finally:
            fs.open_fs("./testdata").removetree("wheels")

    def build_cmd_args(self, args: dict[str, str | bool | list[str]]) -> list[str]:
        result: list[str] = []
        for k, v in args.items():
            if k in ["wheels", "lock_packages", "packages", "requirements_txt"]:
                continue
            key = k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    result.append(f"--{key}")
            elif isinstance(v, list):
                result.extend([f"--{key}={item}" for item in v])
            elif isinstance(v, str):
                if re.match(r"^(?:--)\w*", v):
                    result.append(v)
                else:
                    result.append(f"--{key}={v}")
            else:
                raise ValueError(f"Unsupported type {type(v)}")
        return result

    @pytest.mark.usefixtures("test_fs")
    @parametrize_options(PythonOptions, "build", _argvalue="build_pip_args")
    def test_write_to_bento_pip_args(
        self, build_pip_args: dict[str, str | bool | list[str]], test_fs: FS
    ):
        results = self.build_cmd_args(build_pip_args)
        PythonOptions(**build_pip_args).with_defaults().write_to_bento(
            test_fs, os.getcwd()
        )

        python_dir = test_fs.opendir("/env/python")
        assert python_dir.exists("/pip_args.txt")
        with python_dir.open("pip_args.txt", "r") as f:  # type: ignore (unfinished FS stub)
            assert len(f.read()) == len(" ".join(results))  # type: ignore

    @pytest.mark.usefixtures("change_test_dir")
    @parametrize_options(PythonOptions, "structure", _argvalue="structure_data")
    def test_structure_python_options(self, structure_data: dict[str, t.Any]):
        data = bentoml_cattr.structure(structure_data, PythonOptions)
        assert data == PythonOptions(**structure_data)


@pytest.mark.parametrize(
    ["options_cls", "value"],
    [
        (DockerOptions, {"distro": "alpine"}),
        (
            CondaOptions,
            {"environment_yml": "./testdata/configuration/example_environment.yml"},
        ),
        (
            PythonOptions,
            {"requirements_txt": "./testdata/configuration/another-requirements.txt"},
        ),
    ],
)
def test_dict_options_converter(
    options_cls: t.Type[DockerOptions | CondaOptions | PythonOptions],
    value: dict[str, str],
):
    from bentoml._internal.bento.build_config import dict_options_converter

    assert options_cls(**value) == dict_options_converter(options_cls)(value)

    assert dict_options_converter(DockerOptions)(DockerOptions()) == DockerOptions()


@pytest.mark.incremental
class TestBentoBuildConfig:
    @pytest.mark.usefixtures("change_test_dir")
    @parametrize_options(BentoBuildConfig, "valid")
    def test_valid_build_config_init(self, options: dict[str, t.Any]):
        assert BentoBuildConfig(**options)

    @pytest.mark.usefixtures("change_test_dir")
    @parametrize_options(BentoBuildConfig, "raisewarning")
    def test_raises_warning_build_config(
        self, options: dict[str, t.Any], caplog: LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            assert BentoBuildConfig(**options)
        assert "Conda will be ignored" in caplog.text

    @pytest.mark.usefixtures("change_test_dir")
    @parametrize_options(BentoBuildConfig, "invalid")
    def test_raise_exception_build_config(self, options: dict[str, t.Any]):
        with pytest.raises(BentoMLException) as excinfo:
            BentoBuildConfig(**options)

        assert "not support" in str(excinfo.value)

    def test_with_defaults(self):
        filed = BentoBuildConfig(service="hello.py:svc").with_defaults()
        assert filed.labels == {}
        assert filed.include == ["*"]
        assert not filed.exclude
        assert filed.docker.distro == DEFAULT_DOCKER_DISTRO
        assert filed.python.lock_packages
        assert not filed.conda.environment_yml

    @pytest.mark.usefixtures("change_test_dir")
    def test_from_yaml(self):
        with open("./testdata/configuration/realworld_bentofile.yaml", "r") as f:
            data = BentoBuildConfig.from_yaml(f)
        assert data.service == "service.py:svc"
        assert data.labels == {"owner": "foo", "project": "bar"}
        assert isinstance(data.include, list)

    def test_raise_exception_from_yaml(self):
        with pytest.raises(yaml.YAMLError) as excinfo:
            missed = """\
service: @test
include:
- "*.py"
docker:
  distro: "debian"
  cuda_version: default
        """
            _ = BentoBuildConfig.from_yaml(BytesIO(missed.encode("utf-8")))
        assert "while scanning for the next" in str(excinfo.value)
