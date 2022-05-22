from __future__ import annotations

import re
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

    from .docker import CUDA10x
    from .docker import CUDA11x
    from .docker import DistroSpecWrapper
    from .build_config import DockerOptions

    CUDAType: t.TypeAlias = CUDA10x | CUDA11x | None
    DistroType: t.TypeAlias = DistroSpecWrapper | None


def clean_bentoml_version() -> str:
    post_version = BENTOML_VERSION.split("+")[0]
    match = re.match(r"^(\d+).(\d+).(\d+)(?:a\d)", post_version)
    if match is None:
        raise BentoMLException("Errors while parsing BentoML version.")
    return match.group()


DOCKERFILE_COMPONENTS = [
    "base_env",
    "header",
    "setup_bento_env",
    "setup_uid_gid",
    "setup_python_package",
    "setup_entrypoint",
    "install_python_with_conda",
]

BASE_ENV_TEMPLATE = """\
COPY --from=xx / /

ARG TARGETARCH

ARG TARGETPLATFORM

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1

ENV LANG=C.UTF-8

ENV LC_ALL=C.UTF-8
"""

HEADER_TEMPLATE = """\
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
"""

SETUP_PYTHON_PACKAGE_TEMPLATE = """\
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
    """

SETUP_ENTRYPOINT_TEMPLATE = """\

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


INSTALL_PYTHON_WITH_CONDA_TEMPLATE = """\
RUN bash <<EOF
SAVED_PYTHON_VERSION=$(cat ./env/python/version.txt)
PYTHON_VERSION=${PYTHON_VERSION%.*}

echo "Installing Python $PYTHON_VERSION with conda.."
conda install -y -n base pkgs/main::python=$PYTHON_VERSION pip

if [ -f ./env/conda/environment.yml ]; then
  # set pip_interop_enabled to improve conda-pip interoperability. Conda can use
  # pip-installed packages to satisfy dependencies.
  echo "Updating conda base environment with environment.yml"
  conda config --set pip_interop_enabled True || true
  conda env update -n base -f ./env/conda/environment.yml
  conda clean --all
fi
EOF
        """


def generate_cuda_instructions(cuda_version: str) -> str:
    """
    Generate instructions for cuda installation.
    """
    cuda_spec = make_cuda_cls(cuda_version)
    if cuda_version.startswith("11"):
        # return instruction set for CUDA 11.x
        return f"""\

                """
    elif cuda_version.startswith("10"):
        # return instruction set for CUDA 10.x
        raise NotImplementedError
    else:
        raise BentoMLException(
            f"Unsupported CUDA version: {cuda_version}. Supported versions: 11.x, 10.x"
        )


def alpine_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Alpine image.
    """
    return {}


def alpine_miniconda_template_context(
    docker_options: DockerOptions,
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Alpine miniconda image.
    """
    return {}


def debian_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Debian image.
    """
    base_context = {"setup_uid_gid": SETUP_UID_GID_TEMPLATE}
    if docker_options.cuda_version is not None:
        # cuda version is set, return instruction sets for cuda.
        base_context["setup_cuda"] = generate_cuda_instructions(
            docker_options.cuda_version
        )
    return base_context


def debian_miniconda_template_context(
    docker_options: DockerOptions,
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Debian MinConda image.
    """
    SETUP_UID_GID_MICROMAMBA_TEMPLATE = """\
            """
    base_context = {
        "setup_uid_gid": SETUP_UID_GID_MICROMAMBA_TEMPLATE,
        "install_python_with_conda": INSTALL_PYTHON_WITH_CONDA_TEMPLATE,
    }
    if docker_options.cuda_version is not None:
        # cuda version is set, return instruction sets for cuda.
        base_context["setup_cuda"] = generate_cuda_instructions(
            docker_options.cuda_version
        )
    return base_context


def amazonlinux_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Amazon Linux image.
    """
    raise NotImplementedError


def ubi8_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the UBI8 image.
    """
    raise NotImplementedError


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

    distro_spec = make_distro_cls(docker_options.distro)
    if distro_spec is None:
        raise BentoMLException(
            "Something went wrong in distros validation for DockerOptions."
        )

    user_defined_image = False
    if docker_options.base_image is None:
        base_image = distro_spec.base_image.format(
            python_version=get_docker_py_format(docker_options)
        )
    else:
        base_image = docker_options.base_image
        user_defined_image = True

    if user_defined_image:
        logger.warning(f"Make sure to have Python installed for {base_image}.")

    context_mapping = {
        "alpine": alpine_template_context,
        "alpine-miniconda": alpine_miniconda_template_context,
        "debian": debian_template_context,
        "debian-miniconda": debian_miniconda_template_context,
        "amazonlinux": amazonlinux_template_context,
        "ubi8": ubi8_template_context,
    }

    template_context = {
        "base_image": base_image,
        "user_defined_image": user_defined_image,
        "bentoml_version": clean_bentoml_version(),  # ensure that we don't have a dirty version.
        "docker_options": bentoml_cattr.unstructure(docker_options),  # type: ignore
        "distro_spec": bentoml_cattr.unstructure(distro_spec),  # type: ignore
        "use_cuda": docker_options.cuda_version is not None,
        "header": HEADER_TEMPLATE,
        "base_env": BASE_ENV_TEMPLATE,
        "setup_bento_env": SETUP_BENTO_ENV_TEMPLATE,
        "setup_python_package": SETUP_PYTHON_PACKAGE_TEMPLATE,
        "setup_entrypoint": SETUP_ENTRYPOINT_TEMPLATE,
        **context_mapping[docker_options.distro](docker_options),
    }

    with open(j2_template, "r", encoding="utf-8") as f:
        template = template_env.from_string(f.read())
        return template.render(**template_context)
