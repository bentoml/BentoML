from __future__ import annotations

import typing as t
import logging
import contextlib
from typing import TYPE_CHECKING
from unittest.mock import patch
from unittest.mock import PropertyMock

import pytest
from jinja2.loaders import DictLoader

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml._internal.bento.gen import ENVIRONMENT
from bentoml._internal.bento.gen import J2_FUNCTION
from bentoml._internal.bento.gen import generate_dockerfile
from bentoml._internal.bento.gen import get_docker_variables
from bentoml._internal.bento.gen import clean_bentoml_version
from bentoml._internal.bento.gen import validate_user_template
from bentoml._internal.bento.docker import DistroSpec
from bentoml._internal.bento.docker import get_supported_spec
from bentoml._internal.bento.build_config import DockerOptions

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from _pytest.logging import LogCaptureFixture


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
def test_get_docker_variables_with_default_image(caplog: LogCaptureFixture):
    with pytest.raises(BentoMLException) as excinfo:
        get_docker_variables(
            DockerOptions(base_image="bentoml/model-server:latest"), None
        )
    assert "Distro spec is required" in str(excinfo.value)

    # with cuda
    distro = "ubi8"
    docker = DockerOptions(distro=distro, cuda_version="default").with_defaults()
    spec = DistroSpec.from_distro(distro, cuda=True)
    opts = get_docker_variables(docker, spec)
    assert opts["__options__distro"] == distro
    assert opts["__options__cuda_version"] == "11.6.2"
    assert not opts["__options__base_image"]
    assert opts["__base_image__"] == "nvidia/cuda:11.6.2-cudnn8-runtime-ubi8"

    with setup_mock_version_info((3, 8, 13)) as version_info:
        distro = "ubi8"
        docker = DockerOptions(distro=distro).with_defaults()
        spec = DistroSpec.from_distro(distro)
        opts = get_docker_variables(docker, spec)
        assert (
            opts["__base_image__"]
            == f"registry.access.redhat.com/ubi8/python-{version_info.major}{version_info.minor}:1"
        )

    docker = DockerOptions(
        base_image="tensorflow/tensorflow:latest-devel-gpu"
    ).with_defaults()
    spec = DistroSpec.from_distro("debian")
    with caplog.at_level(logging.INFO):
        opts = get_docker_variables(docker, spec)
    assert (
        "BentoML will not install Python to custom base images; ensure the base image"
        in caplog.text
    )


def test_invalid_user_template():
    without_extends_block = """\
{% block SETUP_BENTO_COMPONENTS %}
{{ super() }}
{% endblock %}
    """
    parsed = ENVIRONMENT.parse(without_extends_block, name="without_extends_block")
    template = ENVIRONMENT.from_string(parsed)
    with pytest.raises(BentoMLException) as excinfo:
        validate_user_template(template, DictLoader({}))
    assert "does not contain `bento__dockerfile`" in str(excinfo.value)


def test_no_name_template():
    no_name_template = """\
{% extends bento__dockerfile %}
{% block SETUP_BENTO_COMPONENTS %}
{{ super() }}
{% endblock %}
    """
    parsed = ENVIRONMENT.parse(no_name_template)
    base_template = ENVIRONMENT.get_template("base.j2", globals=J2_FUNCTION)
    template = ENVIRONMENT.from_string(
        parsed, globals={"bento__dockerfile": base_template}
    )
    with pytest.raises(BentoMLException) as excinfo:
        validate_user_template(template, DictLoader({}))
    assert "Template name is invalid" in str(excinfo.value)


def test_not_allowed_setup_block():
    not_allowed = """\
{% extends bento__dockerfile %}
{% block SETUP_BENTO_NOT_ALLOWED %}
RUN echo "This is not allowed"
{% endblock %}
    """
    parsed = ENVIRONMENT.parse(not_allowed, name="not_allowed")
    base_template = ENVIRONMENT.get_template("base.j2", globals=J2_FUNCTION)
    template = ENVIRONMENT.from_string(
        parsed, globals={"bento__dockerfile": base_template}
    )
    template.name = "not_allowed"  # We have to monkeypatch here for tests, FileSystemLoader will handle template name properly

    with pytest.raises(BentoMLException) as excinfo:
        validate_user_template(template, DictLoader({"not_allowed": not_allowed}))
    assert "Unknown SETUP block in" in str(excinfo.value)
    assert "All supported blocks include" in str(excinfo.value)


def test_use_reserved_env():
    forbidden = """\
{% extends bento__dockerfile %}
{% set __base_image__ = "bentoml/bento-server:latest" %}
{% block SETUP_BENTO_ENTRYPOINT %}
{{ super() }}
RUN --network=none python -m bentoml --version
{% endblock %}
    """
    parsed = ENVIRONMENT.parse(forbidden, name="forbidden")
    base_template = ENVIRONMENT.get_template("base.j2", globals=J2_FUNCTION)
    template = ENVIRONMENT.from_string(
        parsed, globals={"bento__dockerfile": base_template}
    )
    template.name = "forbidden"

    with pytest.raises(BentoMLException) as excinfo:
        validate_user_template(template, DictLoader({"forbidden": forbidden}))
    assert "User defined Dockerfile template contains reserved variables." in str(
        excinfo.value
    )

    forbidden_options = """\
{% extends bento__dockerfile %}
{% set __options__distro = "centos" %}
{% block SETUP_BENTO_ENTRYPOINT %}
{{ super() }}
RUN --network=none python -m bentoml --version
{% endblock %}
    """
    parsed = ENVIRONMENT.parse(forbidden_options, name="forbidden_options")
    base_template = ENVIRONMENT.get_template("base.j2", globals=J2_FUNCTION)
    template = ENVIRONMENT.from_string(
        parsed, globals={"bento__dockerfile": base_template}
    )
    template.name = "forbidden_options"

    with pytest.raises(BentoMLException) as excinfo:
        validate_user_template(
            template, DictLoader({"forbidden_options": forbidden_options})
        )
    assert "User defined Dockerfile template contains reserved variables." in str(
        excinfo.value
    )


def test_generate_options_no_defaults_call():
    with pytest.raises(BentoMLException) as excinfo:
        _ = generate_dockerfile(DockerOptions(), use_conda=False)
    assert "function is called before with_defaults() is invoked." in str(excinfo.value)


@pytest.mark.usefixtures("change_test_dir")
def test_generate_dockerfile():
    res = generate_dockerfile(
        DockerOptions(
            dockerfile_template="./testdata/configuration/Dockerfile.template"
        ).with_defaults(),
        use_conda=False,
    )
    expected = '# syntax = docker/dockerfile:1.4-labs\n#\n# ===========================================\n#\n# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT\n#\n# ===========================================\n\n\nFROM --platform=$BUILDPLATFORM python:3.7-slim as buildstage\n\nRUN mkdir /tmp/mypackage\n\nWORKDIR /tmp/mypackage/\nCOPY . .\nRUN python setup.py sdist && mv dist/*.tar.gz mypackage.tar.gz\n\nFROM python:3.10-slim as cached\n\nFROM python:3.10-slim as base\n\nFROM base as base-amd64\nFROM base as base-arm64\nFROM base as base-ppc64le\nFROM base as base-s390x\n\nFROM base-${TARGETARCH}\n\nARG TARGETARCH\n\nARG TARGETPLATFORM\n\nENV LANG=C.UTF-8\n\nENV LC_ALL=C.UTF-8\n\nENV PYTHONIOENCODING=UTF-8\n\nENV PYTHONUNBUFFERED=1\n\nENV PATH /home/bentoml/bento/env/docker:/usr/local/bin:$PATH\n\n\nUSER root\n\nENV DEBIAN_FRONTEND=noninteractive\nRUN rm -f /etc/apt/apt.conf.d/docker-clean; echo \'Binary::apt::APT::Keep-Downloaded-Packages "true";\' > /etc/apt/apt.conf.d/keep-cache\nRUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/apt \\\n    --mount=type=cache,from=cached,sharing=shared,target=/var/lib/apt \\\n\tapt-get update -y \\\n    && apt-get install -q -y --no-install-recommends --allow-remove-essential \\\n\t\tca-certificates gnupg2 bash build-essential \\\n    && apt-get clean\n\nRUN rm -rf /var/lib/{apt,cache,log}\n\n\nARG BENTO_USER=bentoml\nARG BENTO_USER_UID=1034\nARG BENTO_USER_GID=1034\nRUN groupadd -g $BENTO_USER_GID -o $BENTO_USER && useradd -m -u $BENTO_USER_UID -g $BENTO_USER_GID -o -r $BENTO_USER\n\nSHELL [ "/bin/bash", "-eo", "pipefail", "-c" ]\n\nARG BENTO_PATH=/home/bentoml/bento\nENV BENTO_PATH=$BENTO_PATH\nENV BENTOML_HOME=/home/bentoml/\n\nRUN mkdir $BENTO_PATH && chown bentoml:bentoml $BENTO_PATH -R\nWORKDIR $BENTO_PATH\n\n# init related components\nCOPY --chown=bentoml:bentoml . ./\n\n\n# Install bentoml from editable wheel\nENV BENTOML_VERSION=1.0.0rc0.post21+gee90e743.d20220605\nRUN --mount=type=cache,mode=0777,target=/root/.cache/pip \\\n    pip install --no-cache-dir /home/bentoml/bento/env/docker/whl/bentoml-1.0.0rc0.post21+gee90e743.d20220605-py3-none-any.whl\n\n# Install python package dependencies with pip\n# BentoML by default generates two requirement files:\n#  - ./env/python/requirements.lock.txt: all dependencies locked to its version presented during `build`\n#  - ./env/python/requirements.txt: all dependencies as user specified in code or requirements.txt file\n# We will try to install from  requirements.txt.lock to install package with -U,\n# else we will try to install from requirements.txt.\nRUN --mount=type=cache,mode=0777,target=/root/.cache/pip bash <<EOF\nif [ -f /home/bentoml/bento/env/python/pip_args.txt ]; then\n  EXTRA_PIP_INSTALL_ARGS=$(cat /home/bentoml/bento/env/python/pip_args.txt)\nfi\nif [ -f /home/bentoml/bento/env/python/requirements.lock.txt ]; then\n  echo "Installing pip packages from \'requirements.lock.txt\'.."\n  pip install -r /home/bentoml/bento/env/python/requirements.lock.txt -U $EXTRA_PIP_INSTALL_ARGS\nelse\n  if [ -f /home/bentoml/bento/env/python/requirements.txt ]; then\n    echo "Installing pip packages from \'requirements.txt\'.."\n    pip install -r /home/bentoml/bento/env/python/requirements.txt $EXTRA_PIP_INSTALL_ARGS\n  fi\nfi\nEOF\n\n# install wheels included in Bento\nRUN bash <<EOF\nif [ -d /home/bentoml/bento/env/python/wheels ]; then\n  echo "Installing wheels.."\n  pip install /home/bentoml/bento/env/python/wheels/*.whl\nfi\nEOF\n\n# Run user setup scripts if exists\nRUN bash <<EOF\nif [ -f /home/bentoml/bento/env/docker/setup_script ]; then\n  chmod +x /home/bentoml/bento/env/docker/setup_script\n  /home/bentoml/bento/env/docker/setup_script\nfi\nEOF\n\n\nCOPY --from=buildstage mypackage.tar.gz /tmp/wheels/\nRUN --network=none pip install --find-links /tmp/wheels mypackage\n# Default port for BentoServer\nEXPOSE 3000\n\nRUN chmod +x /home/bentoml/bento/env/docker/entrypoint.sh\n\nUSER bentoml\n\nENTRYPOINT [ "/home/bentoml/bento/env/docker/entrypoint.sh" ]\n\nCMD [ "bentoml", "serve", "/home/bentoml/bento", "--production" ]\n'
    assert res == expected
