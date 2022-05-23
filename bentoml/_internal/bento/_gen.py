from __future__ import annotations

import re
import typing as t
import logging
from typing import TYPE_CHECKING
from functools import wraps

import fs
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import bentoml_cattr
from .docker import make_cuda_cls
from .docker import make_distro_cls
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")

    from .docker import CUDA
    from .build_config import DockerOptions

    TemplateFunc = t.Callable[[DockerOptions], t.Dict[str, t.Any]]


def clean_bentoml_version() -> str:
    post_version = BENTOML_VERSION.split("+")[0]
    match = re.match(r"^(\d+).(\d+).(\d+)(?:a\d)", post_version)
    if match is None:
        raise BentoMLException("Errors while parsing BentoML version.")
    return match.group()


def ensure_components(
    _func: TemplateFunc | None = None,
    *,
    conda: bool = False,
    cuda: bool = False,
):

    _require_setup = ["setup_distro_env", "setup_uid_gid", "cleanup"]
    if conda:
        _require_setup += ["install_python_with_conda"]
    if cuda:
        _require_setup += ["setup_cuda"]

    def decorator(func: TemplateFunc) -> TemplateFunc:
        @wraps(func)
        def wrapper(opts: DockerOptions) -> t.Dict[str, t.Any]:
            ret = func(opts)
            missing = list(filter(lambda x: x not in ret, _require_setup))
            if len(missing) > 0:
                raise BentoMLException(
                    f"`{func.__name__}` returns template context that miss "
                    f"setup components. Missing components: {','.join(missing)}"
                )
            return ret

        return wrapper

    if _func is not None:
        return decorator(_func)
    else:
        return decorator


HEADER_TEMPLATE = """\
# syntax = docker/dockerfile:1.4-labs
#
# ===========================================
#
# THIS IS A GENERATED DOCKERFILE. DO NOT EDIT
#
# ===========================================
        """

BASE_ENV_TEMPLATE = """\
COPY --link --from=xx / /

ARG TARGETARCH

ARG TARGETPLATFORM

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1

ENV LANG=C.UTF-8

ENV LC_ALL=C.UTF-8
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
COPY --link --chown=bentoml:bentoml ./env ./env
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


SETUP_ALPINE_ENV_TEMPLATE = """\
ENV PATH /usr/local/bin:$PATH

ENV ENV /root/.bashrc

# Install helpers
RUN --mount=type=cache,from=cached,target=/var/cache/apk \\
    xx-apk add --update bash gcc libc-dev shadow musl-dev build-base \\
    linux-headers g++
"""

CLEANUP_ALPINE_TEMPLATE = """\
RUN rm -rf /var/cache/apk/*
        """


def alpine_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Alpine image.
    """
    return {
        "setup_distro_env": SETUP_ALPINE_ENV_TEMPLATE,
        "setup_uid_gid": SETUP_UID_GID_TEMPLATE,
        "cleanup": CLEANUP_ALPINE_TEMPLATE,
    }


def alpine_miniconda_template_context(
    docker_options: DockerOptions,
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Alpine miniconda image.
    """
    return {
        "setup_distro_env": SETUP_ALPINE_ENV_TEMPLATE,
        "setup_uid_gid": SETUP_UID_GID_TEMPLATE,
        "cleanup": CLEANUP_ALPINE_TEMPLATE,
        "install_python_with_conda": INSTALL_PYTHON_WITH_CONDA_TEMPLATE,
    }


SETUP_DEBIAN_ENV_TEMPLATE = """\
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/apt \\
    --mount=type=cache,from=cached,sharing=shared,target=/var/lib/apt \\
    xx-apt-get install -q -y --no-install-recommends --allow-remove-essential \\
    bash build-essential \\
    && xx-apt-get clean \\
    && rm -rf /var/lib/apt/lists/*
"""

CLEANUP_DEBIAN_TEMPLATE = """\
RUN rm -rf /var/lib/{apt,dpkg,cache,log}
        """


def setup_cuda_debian(
    repository: str, distros: str, cuda_major_version: str, cuda_minor_version: str
):
    SETUP_CUDA_DEBIAN_TEMPLATE = (
        f"""\
RUN xx-apt-get install -y --no-install-recommends \\
        gnupg2 curl ca-certificates \\
        && curl -fsSLO {repository}/{distros}/"""
        + "${NVARCH}"
        + f"""/cuda-keyring_1.0-1_all.deb \\
        && dpkg -i cuda-keyring_1.0-1_all.deb \\
        && rm cuda-keyring_1.0-1_all.deb \\
        && xx-apt-get install -y --no-install-recommends \\
        cuda-cudart-{cuda_major_version}-{cuda_minor_version}="""
        + "${NVIDIA_CUDA_CUDART_VERSION}"
        + """ \\
            """
        + "${NVIDIA_CUDA_COMPAT_PACKAGE}"
        + f""" \\
        && ln -s cuda-{cuda_major_version}.{cuda_minor_version} /usr/local/cuda \\
        && xx-apt-get purge --autoremove -y curl \\
        && rm -rf /var/lib/apt/lists/*
        """
    )
    return SETUP_CUDA_DEBIAN_TEMPLATE


def debian_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Debian image.
    """
    cuda_version = docker_options.cuda_version
    if cuda_version is not None:
        cuda: CUDA = make_cuda_cls(docker_options.cuda_version)  # type: ignore
        if "debian" in docker_options.distro:
            if docker_options.cuda_version.startswith("10"):
                distros = "ubuntu1804"
            else:
                distros = "ubuntu2004"
        else:
            distros = "rhel8"
        cuda_debian_setup = setup_cuda_debian(
            repository=cuda.repository,
            distros=distros,
            cuda_major_version=cuda.version.major,
            cuda_minor_version=cuda.version.minor,
        )
    else:
        cuda_debian_setup = ""
    return {
        "setup_uid_gid": SETUP_UID_GID_TEMPLATE,
        "setup_distro_env": SETUP_DEBIAN_ENV_TEMPLATE,
        "cleanup": CLEANUP_DEBIAN_TEMPLATE,
        "setup_cuda": cuda_debian_setup,
    }


def debian_miniconda_template_context(
    docker_options: DockerOptions,
) -> t.Dict[str, t.Any]:
    """
    Generate a Dockerfile for the Debian MinConda image.
    """
    SETUP_UID_GID_MICROMAMBA_TEMPLATE = """\
            """

    cuda_version = docker_options.cuda_version
    if cuda_version is not None:
        cuda: CUDA = make_cuda_cls(docker_options.cuda_version)  # type: ignore
        if "debian" in docker_options.distro:
            if docker_options.cuda_version.startswith("10"):
                distros = "ubuntu1804"
            else:
                distros = "ubuntu2004"
        else:
            distros = "rhel8"
        cuda_debian_setup = setup_cuda_debian(
            repository=cuda.repository,
            distros=distros,
            cuda_major_version=cuda.version.major,
            cuda_minor_version=cuda.version.minor,
        )
    else:
        cuda_debian_setup = ""
    return {
        "setup_uid_gid": SETUP_UID_GID_MICROMAMBA_TEMPLATE,
        "setup_distro_env": SETUP_DEBIAN_ENV_TEMPLATE,
        "install_python_with_conda": INSTALL_PYTHON_WITH_CONDA_TEMPLATE,
        "cleanup": CLEANUP_DEBIAN_TEMPLATE,
        "setup_cuda": cuda_debian_setup,
    }


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


DOCKERFILE_COMPONENTS = {
    "header": HEADER_TEMPLATE,
    "base_env": BASE_ENV_TEMPLATE,
    "setup_bentoml_env": SETUP_BENTO_ENV_TEMPLATE,
    "setup_python_package": SETUP_PYTHON_PACKAGE_TEMPLATE,
    "setup_entrypoint": SETUP_ENTRYPOINT_TEMPLATE,
}

TEMPLATE_MAP = {
    "alpine": alpine_template_context,
    "alpine-miniconda": alpine_miniconda_template_context,
    "debian": debian_template_context,
    "debian-miniconda": debian_miniconda_template_context,
    "amazonlinux": amazonlinux_template_context,
    "ubi8": ubi8_template_context,
}


def generate_dockerfile(docker_options: DockerOptions) -> str:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version
    python_version = docker_options.python_version

    distro_spec = make_distro_cls(distro)
    cuda_spec = make_cuda_cls(cuda_version)

    docker_dir = fs.path.combine(fs.path.dirname(__file__), "docker")
    j2_template = fs.path.combine(docker_dir, f"Dockerfile.j2")
    template_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(docker_dir, followlinks=True),
    )

    if docker_options.base_image is None:
        if docker_options.distro == "ubi8":
            python_version = python_version.replace(".", "")
        else:
            python_version = python_version
        base_image = distro_spec.base_image.format(python_version=python_version)
    else:
        base_image = docker_options.base_image
        logger.warning(f"Make sure to have Python installed for {base_image}.")

    context_mapping: dict[str, TemplateFunc] = {  # type: ignore
        k: ensure_components(v, conda="conda" in k, cuda=cuda_version is not None)
        for k, v in TEMPLATE_MAP.items()
    }

    template_context = {
        "base_image": base_image,
        "user_defined_image": docker_options._user_defined_image,  # type: ignore
        "bentoml_version": clean_bentoml_version(),  # ensure that we don't have a dirty version.
        "docker_options": bentoml_cattr.unstructure(docker_options),  # type: ignore
        "distro_spec": bentoml_cattr.unstructure(distro_spec),  # type: ignore
        "cuda_spec": bentoml_cattr.unstructure(cuda_spec),  # type: ignore
        **DOCKERFILE_COMPONENTS,
        **context_mapping[distro](docker_options),
    }

    with open(j2_template, "r", encoding="utf-8") as f:
        template = template_env.from_string(f.read())
        return template.render(**template_context)
