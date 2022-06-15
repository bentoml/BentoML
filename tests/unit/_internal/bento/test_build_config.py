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


def parametrize_options(
    options_cls: t.Type[
        DockerOptions | CondaOptions | PythonOptions | BentoBuildConfig
    ],
    test_type: t.Literal["valid", "raisewarning", "invalid"],
    *,
    attribute: str | None = None,
    _argvalue: str | None = None,
    _as_file_name: bool = False,
) -> MarkDecorator:
    if not hasattr(options_cls, "__attrs_attrs__"):
        raise ValueError(f"{options_cls} is not an attrs class")
    if test_type not in ["valid", "invalid", "raisewarning"]:
        raise InvalidArgument(f"Unknown test type {test_type}")

    if attribute is not None:
        if attribute not in [a.name for a in attr.fields(options_cls)]:
            raise InvalidArgument(f"{options_cls} has no attribute {attribute}")
        rgx = re.compile(f"^{options_cls.__name__.lower()}_{test_type}_{attribute}*")
    else:
        rgx = re.compile(rf"^{options_cls.__name__.lower()}_{test_type}\d+")

    testdata_dir = Path(__file__).parent / "testdata"
    testdata: list[dict[str, t.Any] | Path] = []

    for data in [f for f in testdata_dir.iterdir() if rgx.match(f.name)]:
        if _as_file_name:
            testdata.append(data)
        else:
            with open(data, "r") as f:
                testdata.append(yaml.safe_load(f))

    if not _argvalue:
        _argvalue = "options"

    return pytest.mark.parametrize(str(_argvalue), testdata)


@parametrize_options(DockerOptions, "valid")
def test_valid_docker_options(options: dict[str, t.Any]):
    assert DockerOptions(**options)


@parametrize_options(DockerOptions, "raisewarning")
def test_raises_warning_docker_options(
    options: dict[str, t.Any], caplog: LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        assert DockerOptions(**options)
    assert "option is ignored" in caplog.text


def test_invalid_cuda_supports_distro():
    with pytest.raises(BentoMLException):
        DockerOptions(distro="alpine", cuda_version="11.6.2")


@pytest.mark.parametrize(
    "options", [{"python_version": "3.6"}, {"distro": "debian", "cuda_version": "12.0"}]
)
def test_invalid_options(options: dict[str, t.Any]):
    with pytest.raises(ValueError):
        DockerOptions(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"distro": "amazonlinux", "cuda_version": "default"},
        {"distro": "alpine", "cuda_version": "default"},
    ],
)
def test_invalid_cuda_version(options: dict[str, t.Any]):
    with pytest.raises(BentoMLException):
        DockerOptions(**options)


@pytest.mark.parametrize(
    "options",
    [{"python_version": "3.7", "env": {"FOO": "bar"}}, {"system_packages": ["libffi"]}],
)
def test_with_defaults_distro(options: dict[str, t.Any]):
    filled = DockerOptions(**options).with_defaults()
    assert filled.distro == DEFAULT_DOCKER_DISTRO
    assert not filled.dockerfile_template


@pytest.mark.usefixtures("test_fs")
@patch("bentoml._internal.configuration.is_pypi_installed_bentoml")
@pytest.mark.parametrize(
    "conda_option", [CondaOptions(), CondaOptions(channels=["default"])]
)
def test_docker_write_to_bento(
    mock_is_pypi_installed_bentoml: MagicMock, conda_option: CondaOptions, test_fs: FS
):
    mock_is_pypi_installed_bentoml.return_value = True

    DockerOptions().with_defaults().write_to_bento(
        test_fs, os.getcwd(), conda_option.with_defaults()
    )
    docker_fs = test_fs.opendir("/env/docker")

    assert os.path.isdir(docker_fs.getsyspath("/"))
    assert docker_fs.exists("Dockerfile")
    assert docker_fs.exists("entrypoint.sh")

    DockerOptions(
        setup_script="./testdata/scripts/setup_script"
    ).with_defaults().write_to_bento(test_fs, os.getcwd(), conda_option.with_defaults())

    assert docker_fs.exists("setup_script")

    with pytest.raises(InvalidArgument):
        DockerOptions(
            setup_script="./path/does/not/exist"
        ).with_defaults().write_to_bento(
            test_fs, os.getcwd(), conda_option.with_defaults()
        )


@pytest.mark.usefixtures("change_test_dir")
def test_structure_docker_options():
    with_env_options = {"distro": "debian", "env": "./testdata/scripts/dot_env"}
    assert bentoml_cattr.structure(with_env_options, DockerOptions) == DockerOptions(
        **with_env_options
    )

    with pytest.raises(BentoMLException) as excinfo:
        bentoml_cattr.structure(
            {"distro": "alpine", "env": "./testdata/script/doesnotexist"},
            DockerOptions,
        )
        assert "Invalid env file path" in str(excinfo.value)


@parametrize_options(CondaOptions, "valid")
def test_valid_conda_options(options: dict[str, t.Any]):
    assert CondaOptions(**options)


@pytest.mark.parametrize(
    "options", [{"channels": "conda-forge"}, {"channels": {"fail": "test"}}]
)
def test_invalid_conda_options(options: dict[str, t.Any]):
    with pytest.raises(TypeError) as excinfo:
        CondaOptions(**options)
        assert "must be <class 'list'>" in str(excinfo.value)


@pytest.mark.parametrize(
    "options",
    [
        {"dependencies": "conda-forge"},
        {
            "dependencies": [
                "numpy=1.0",
                {"pip": ["more=than"]},
                {"wrong_key": ["foo=bar"]},
            ]
        },
        {
            "dependencies": [
                "numpy=1.0",
                {"pip": ["spacy", 123123]},
            ]
        },
        {
            "dependencies": [
                "numpy=1.0",
                {"wrong_key": ["foo=bar"]},
            ]
        },
    ],
)
def test_dependencies_validator(options: dict[str, t.Any]):
    with pytest.raises(InvalidArgument):
        CondaOptions(**options)


@pytest.mark.usefixtures("change_test_dir")
@parametrize_options(CondaOptions, "raisewarning")
def test_raises_warning_conda_options(
    options: dict[str, t.Any], caplog: LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        assert CondaOptions(**options)
    assert "option is ignored" in caplog.text


@pytest.mark.usefixtures("test_fs")
def test_conda_write_to_bento(test_fs: FS):
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
            "channels": ["conda-forge"],
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
@pytest.mark.parametrize(
    "structure_data",
    [
        {
            "pip": [
                "aiohttp",
                "git+https://github.com/blaze/dask.git#egg=dask[complete]",
            ]
        },
        {
            "dependencies": [
                "python=3.4",
                "numpy",
                "toolz",
                "dill",
                "pandas",
                "partd",
                "bokeh",
                {
                    "pip": [
                        "numpy",
                        "matplotlib==3.5.1",
                        "package>=0.2,<0.3",
                        "torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu",
                        "git+https://github.com/bentoml/bentoml.git@main",
                    ]
                },
            ]
        },
    ],
)
def test_structure_conda_options(structure_data: dict[str, t.Any]):
    data = bentoml_cattr.structure(structure_data, CondaOptions)
    assert data == CondaOptions(**structure_data)

    with pytest.raises(InvalidArgument) as excinfo:
        bentoml_cattr.structure({"dependencies": "not a list"}, CondaOptions)
        assert "type list" in str(excinfo.value)


@pytest.mark.usefixtures("change_test_dir")
def test_structure_conda_options_empty():
    data = bentoml_cattr.structure({}, CondaOptions)
    assert data == CondaOptions()


def fs_identical(fs1: FS, fs2: FS):
    for path in fs1.walk.dirs():
        assert fs2.isdir(path)

    for path in fs1.walk.files():
        assert fs2.isfile(path)
        assert fs1.readbytes(path) == fs2.readbytes(path)


@parametrize_options(PythonOptions, "valid")
def test_valid_python_options(options: dict[str, t.Any]):
    assert PythonOptions(**options)


@pytest.mark.parametrize(
    "options",
    [
        {
            "requirements_txt": ["notaccepted", "dont do this"],
            "packages": [{"matplotlib": "3.5.1"}, {"packaging": ">=0.2,<0.3"}],
        },
        {
            "lock_packages": "False",
            "index_url": ["not string", "just input one string"],
        },
    ],
)
def test_invalid_python_options(options: dict[str, t.Any]):
    with pytest.raises(TypeError) as excinfo:
        PythonOptions(**options)
        assert "must be <class" in str(excinfo.value)


@pytest.mark.usefixtures("change_test_dir")
@parametrize_options(PythonOptions, "raisewarning")
def test_raises_warning_python_options(
    options: dict[str, t.Any], caplog: LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        assert PythonOptions(**options)
    assert "Build option python" in caplog.text


@pytest.mark.usefixtures("test_fs")
def test_python_write_to_bento(test_fs: FS):
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
def test_wheels_include(test_fs: FS):
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


@pytest.mark.usefixtures("test_fs")
def test_wheels_include_local_bento(test_fs: FS, monkeypatch: MonkeyPatch):
    from bentoml._internal.bento.build_dev_bentoml_whl import BENTOML_DEV_BUILD

    monkeypatch.setenv(BENTOML_DEV_BUILD, "True")
    PythonOptions(lock_packages=False).with_defaults().write_to_bento(
        test_fs, os.getcwd()
    )

    python_dir = test_fs.opendir("/env/python")
    assert os.path.isdir(python_dir.getsyspath("/"))
    assert os.path.isdir(python_dir.getsyspath("/wheels"))


def build_cmd_args(args: dict[str, str | bool | list[str]]) -> list[str]:
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
@pytest.mark.parametrize(
    "build_pip_args",
    [
        {
            "requirements_txt": "./testdata/configuration/example_requirements.txt",
            "pip_args": "--proxy=foo --cert=bar",
            "trusted_host": ["pypi.python.org", "foo.cdns.com"],
            "find_links": ["https://download.pytorch.org/whl/cu80/stable.html"],
            "extra_index_url": ["https://mirror.baidu.cn/simple"],
        },
        {
            "requirements_txt": "./testdata/configuration/example_requirements.txt",
            "no_index": True,
            "index_url": "http://pypi.python.org/simple",
        },
    ],
)
def test_write_to_bento_pip_args(
    build_pip_args: dict[str, str | bool | list[str]], test_fs: FS
):
    results = build_cmd_args(build_pip_args)
    PythonOptions(**build_pip_args).with_defaults().write_to_bento(test_fs, os.getcwd())

    python_dir = test_fs.opendir("/env/python")
    assert python_dir.exists("/pip_args.txt")
    with python_dir.open("pip_args.txt", "r") as f:  # type: ignore (unfinished FS stub)
        assert len(f.read()) == len(" ".join(results))  # type: ignore


@pytest.mark.usefixtures("change_test_dir")
@pytest.mark.parametrize(
    "structure_data",
    [
        {
            "trusted_host": "https://foo:bar",
            "find_links": "https://download.pytorch.org/whl/cu80/stable.html",
            "extra_index_url": "https://pypi.python.org/simple",
        },
    ],
)
def test_structure_python_options(structure_data: dict[str, t.Any]):
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


@pytest.mark.usefixtures("change_test_dir")
@pytest.mark.parametrize(
    "options",
    [
        {
            "service": "service.py:svc",
            "labels": {"owner": "bentoml-team", "project": "gallery"},
            "include": ["*.py"],
            "python": {"packages": ["scikit-learn", "pandas"]},
        },
        {"service": "service.py:svc"},
    ],
)
def test_valid_build_config_init(options: dict[str, t.Any]):
    assert BentoBuildConfig(**options)


def test_ignore_conda_build_config(caplog: LogCaptureFixture):
    options = {
        "service": "service.py:svc",
        "labels": {"owner": "bentoml-team", "project": "gallery"},
        "include": ["*.py"],
        "python": {"packages": ["scikit-learn", "pandas"]},
        "conda": {
            "environment_yml": "./testdata/configuration/example_environment.yml"
        },
        "docker": {"distro": "debian", "cuda_version": "default"},
    }
    with caplog.at_level(logging.WARNING):
        assert BentoBuildConfig(**options)
    assert "Conda will be ignored" in caplog.text


@pytest.mark.usefixtures("change_test_dir")
@parametrize_options(BentoBuildConfig, "invalid")
def test_raise_exception_build_config(options: dict[str, t.Any]):
    with pytest.raises(BentoMLException) as excinfo:
        BentoBuildConfig(**options)

    assert "not support" in str(excinfo.value)


def test_raise_exception_docker_env_build_config():
    invalid = """\
service: "service.py:svc"
labels:
  owner: bentoml-team
  project: gallery
include:
- "*.py"
docker:
  env:
    - test=foo
    - notaccepted
            """
    with pytest.raises(BentoMLException) as excinfo:
        assert BentoBuildConfig.from_yaml(BytesIO(invalid.encode("utf-8")))
    assert "must follow" in str(excinfo.value)

    invalid = """\
service: "service.py:svc"
labels:
  owner: bentoml-team
  project: gallery
include:
- "*.py"
docker:
  env: 123
            """
    with pytest.raises(BentoMLException) as excinfo:
        assert BentoBuildConfig.from_yaml(BytesIO(invalid.encode("utf-8")))
    assert "must be either a list or a dict" in str(excinfo.value)


def test_with_defaults():
    filed = BentoBuildConfig(service="hello.py:svc").with_defaults()
    assert filed.labels == {}
    assert filed.include == ["*"]
    assert not filed.exclude
    assert filed.docker.distro == DEFAULT_DOCKER_DISTRO
    assert filed.python.lock_packages
    assert not filed.conda.environment_yml


@pytest.mark.usefixtures("change_test_dir")
def test_from_yaml():
    with open("./testdata/configuration/realworld_bentofile.yaml", "r") as f:
        data = BentoBuildConfig.from_yaml(f)
    assert data.service == "service.py:svc"
    assert data.labels == {"owner": "foo", "project": "bar"}
    assert isinstance(data.include, list)


@pytest.mark.usefixtures("change_test_dir")
def test_envar_warning_from_yaml(caplog: LogCaptureFixture):
    warning = """\
service: name
docker:
  env:
    FOO:
            """
    with caplog.at_level(logging.WARNING):
        assert BentoBuildConfig.from_yaml(BytesIO(warning.encode("utf-8")))
    assert "`env` dictionary contains 'None' value" in caplog.text


def test_raise_exception_from_yaml():
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

    with pytest.raises(InvalidArgument) as excinfo:
        missed = """\
include:
- "*.py"
docker:
  distro: "debian"
  cuda_version: default
    """
        _ = BentoBuildConfig.from_yaml(BytesIO(missed.encode("utf-8")))
    assert 'Missing required build config field "service"' in str(excinfo.value)

    with pytest.raises(InvalidArgument) as excinfo:
        missed = """\
service: ""
include:
- "*.py"
docker:
  notexists:
    """
        _ = BentoBuildConfig.from_yaml(BytesIO(missed.encode("utf-8")))
    assert "got an unexpected keyword argument" in str(excinfo.value)
