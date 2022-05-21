from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import fs
from jinja2 import Environment

from ..utils import bentoml_cattr
from .docker import make_cuda_cls
from .docker import make_distro_cls
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    from .docker import _CUDASpec10Type
    from .docker import _CUDASpec11Type
    from .docker import _DistroSpecWrapper
    from .build_config import DockerOptions

    CUDAType: t.TypeAlias = _CUDASpec10Type | _CUDASpec11Type | None
    DistroType: t.TypeAlias = _DistroSpecWrapper | None

bentoml_version = BENTOML_VERSION.rsplit(".", maxsplit=2)[0]

HEADERS_TEMPLATE = """\
# syntax = docker/dockerfile:1.4-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================
        """

SETUP_UID_GID_TEMPLATE = """\
ARG UID=1034
ARG GID=1034
RUN groupadd -g $GID -o bentoml && useradd -m -u $UID -g $GID -o -r bentoml
"""

SETUP_BENTO_ENV_TEMPLATE = """\
ARG BENTO_PATH=/home/bentoml/bento
ENV BENTO_PATH=$BENTO_PATH
ENV BENTOML_HOME=/home/bentoml/

RUN mkdir $BENTO_PATH && chown bentoml:bentoml $BENTO_PATH -R
WORKDIR $BENTO_PATH

# init related components
COPY --chown=bentoml:bentoml ./env ./env

# install python package with wheels
# BentoML by default generates two requirment files:
#  - ./env/python/requirements.lock.txt: all dependencies locked to its version presented during `build`
#  - ./env/python/requirements.txt: all dependecies as user specified in code or requirements.txt file
# We will only copy over the requirements.txt.lock to install package with -U
RUN bash <<EOF
if [ -f ./env/python/pip_args.txt ]; then
  EXTRA_PIP_INSTALL_ARGS=$(cat ./env/python/pip_args.txt)
fi
if [ -f ./env/python/requirements.lock.txt ]; then
  echo "Installing pip packages from 'requirements.lock.txt'.."
  pip install -r ./env/python/requirements.lock.txt -U --no-cache-dir $EXTRA_PIP_INSTALL_ARGS
fi
EOF

# Run user setup scripts if exists
RUN bash <<EOF
if [ -f ./env/docker/setup_script ]; then
  chmod +x ./env/docker/setup_script
  ./env/docker/setup_script
fi
EOF

# copy over all remaining bento files
COPY --chown=bentoml:bentoml . ./

# Default port for BentoServer
EXPOSE 3000

RUN <<EOF > ./env/docker/entrypoint.sh
#!/usr/bin/env bash
set -Eeuo pipefail

# check to see if this file is being run or sourced from another script
_is_sourced() {
  # https://unix.stackexchange.com/a/215279
  [ "${#FUNCNAME[@]}" -ge 2 ] \
    && [ "${FUNCNAME[0]}" = '_is_sourced' ] \
    && [ "${FUNCNAME[1]}" = 'source' ]
}

_main() {
  # if first arg looks like a flag, assume we want to start bentoml YataiService
  if [ "${1:0:1}" = '-' ]; then
    set -- bentoml serve --production "$@" $BENTO_PATH
  fi

  # Overide the BENTOML_PORT if PORT env var is present. Used for Heroku
  if [[ -v PORT ]]; then
    echo "\\$PORT is set! Overiding \\$BENTOML_PORT with \\$PORT ($PORT)"
    export BENTOML_PORT=$PORT
  fi

  exec "$@"
}

if ! _is_sourced; then
  _main "$@"
fi
EOF

RUN chmod +x ./env/docker/entrypoint.sh

USER bentoml

ENTRYPOINT [ "./env/docker/entrypoint.sh" ]

CMD ["bentoml", "serve", ".", "--production"]
        """


def architecture_definition(cuda_spec: CUDAType) -> str:
    if cuda_spec is None:
        # No CUDA support
        return ""
    else:
        if cuda_spec.version.major == "10":
            return ""
        else:
            return ""


def generate_alpine_dockerfile(
    docker_options: DockerOptions, cuda_spec: CUDAType, distro_spec: DistroType
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Alpine image.
    """
    return {
        "header": HEADERS_TEMPLATE,
        "setup_uid_gid": SETUP_UID_GID_TEMPLATE,
        "setup_bento_env": SETUP_BENTO_ENV_TEMPLATE,
    }


def generate_alpine_miniconda_dockerfile(
    docker_options: DockerOptions, cuda_spec: CUDAType, distro_spec: DistroType
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Alpine miniconda image.
    """


def generate_ubi8_dockerfile(
    docker_options: DockerOptions, cuda_spec: CUDAType, distro_spec: DistroType
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the UBI8 image.
    """
    print(cuda_spec)
    return ""


def generate_debian_dockerfile(
    docker_options: DockerOptions, cuda_spec: CUDAType, distro_spec: DistroType
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Debian image.
    """


def generate_amazonlinux_dockerfile(
    docker_options: DockerOptions, cuda_spec: CUDAType, distro_spec: DistroType
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Amazon Linux image.
    """


def generate_debian_miniconda_dockerfile(
    docker_options: DockerOptions, cuda_spec: CUDAType, distro_spec: DistroType
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Debian MinConda image.
    """


def get_docker_py_format(docker_options: DockerOptions) -> str:
    if docker_options.distro == "ubi8":
        return docker_options.python_version.replace(".", "")
    return docker_options.python_version


def generate_dockerfile(docker_options: DockerOptions) -> str:
    docker_dir = fs.path.combine(fs.path.dirname(__file__), "docker")
    j2_template = fs.path.combine(docker_dir, f"Dockerfile-{docker_options.distro}.j2")
    template_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
    )

    cuda_spec = make_cuda_cls(docker_options.cuda_version)
    distro_spec = make_distro_cls(docker_options.distro)
    if distro_spec is None:
        raise BentoMLException(
            "Something went wrong in distros validation for DockerOptions."
        )

    if docker_options.base_image is None:
        base_image = distro_spec.base_image
    else:
        base_image = docker_options.base_image

    if docker_options.distro == "debian":
        content = generate_debian_dockerfile(
            docker_options, cuda_spec=cuda_spec, distro_spec=distro_spec
        )
    elif docker_options.distro == "alpine":
        content = generate_alpine_dockerfile(
            docker_options, cuda_spec=cuda_spec, distro_spec=distro_spec
        )
    elif docker_options.distro == "amazonlinux":
        content = generate_amazonlinux_dockerfile(
            docker_options, cuda_spec=cuda_spec, distro_spec=distro_spec
        )
    elif docker_options.distro == "ubi8":
        content = generate_ubi8_dockerfile(
            docker_options, cuda_spec=cuda_spec, distro_spec=distro_spec
        )
    elif docker_options.distro == "debian-miniconda":
        content = generate_debian_miniconda_dockerfile(
            docker_options, cuda_spec=cuda_spec, distro_spec=distro_spec
        )
    elif docker_options.distro == "alpine-miniconda":
        content = generate_alpine_miniconda_dockerfile(
            docker_options, cuda_spec=cuda_spec, distro_spec=distro_spec
        )
    else:
        content = {}

    template_context = {
        "base_image": base_image.format(
            python_version=get_docker_py_format(docker_options)
        ),
        "docker_options": bentoml_cattr.unstructure(docker_options),
        "distro_spec": bentoml_cattr.unstructure(distro_spec),
        "cuda_spec": bentoml_cattr.unstructure(cuda_spec),
        "bentoml_version": bentoml_version,  # ensure that we don't have a dirty version.
        **content,
    }

    with open(j2_template, "r", encoding="utf-8") as f:
        template = template_env.from_string(f.read())
        return template.render(**template_context)
