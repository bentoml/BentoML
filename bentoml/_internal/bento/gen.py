from __future__ import annotations

import re
import typing as t
import logging
from typing import TYPE_CHECKING

import fs
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from ..utils import bentoml_cattr
from .docker import DistroSpec
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from .build_config import DockerOptions

    TemplateFunc = t.Callable[[DockerOptions], t.Dict[str, t.Any]]


def clean_bentoml_version() -> str:
    post_version = BENTOML_VERSION.split("+")[0]
    match = re.match(r"^(\d+).(\d+).(\d+)(?:(a|rc)\d)", post_version)
    if match is None:
        raise BentoMLException("Errors while parsing BentoML version.")
    return match.group()


# SETUP_DEBIAN_ENV_TEMPLATE = """\
# ENV DEBIAN_FRONTEND=noninteractive

# RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
# RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/apt \\
#     --mount=type=cache,from=cached,sharing=shared,target=/var/lib/apt \\
#     xx-apt-get install -q -y --no-install-recommends --allow-remove-essential \\
#     bash build-essential \\
#     && xx-apt-get clean \\
#     && rm -rf /var/lib/apt/lists/*
# """

# CLEANUP_DEBIAN_TEMPLATE = """\
# RUN rm -rf /var/lib/{apt,cache,log}
#         """


# def _install_debian_system_packages(docker_options: DockerOptions) -> str:
#     if docker_options.system_packages is not None:
#         return f"""\
# RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/apt \\
#     --mount=type=cache,from=cached,sharing=shared,target=/var/lib/apt \\
#     xx-apt-get install -q -y --no-install-recommends --allow-remove-essential \\
#     {" ".join(docker_options.system_packages)} \\
#     && xx-apt-get clean \\
#     && rm -rf /var/lib/apt/lists/*
#             """
#     return ""


# def _setup_debian_base_cuda(
#     repository: str, distros: str, cuda_major_version: str, cuda_minor_version: str
# ):
#     SETUP_DEBIAN_BASE_CUDA_TEMPLATE = f"""\
# # For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
# RUN xx-apt-get update && xx-apt-get install -y --no-install-recommends \\
#         gnupg2 curl ca-certificates \\
#         && curl -fsSLO {repository}/{distros}/{"${NVARCH}"}/cuda-keyring_1.0-1_all.deb \\
#         && dpkg -i cuda-keyring_1.0-1_all.deb \\
#         && rm cuda-keyring_1.0-1_all.deb \\
#         && xx-apt-get install -y --no-install-recommends \\
#         {"${NVIDIA_CUDA_CUDART_PACKAGE}"} \\
#         {"${NVIDIA_CUDA_COMPAT_PACKAGE}"} \\
#         && ln -s cuda-{cuda_major_version}.{cuda_minor_version} /usr/local/cuda \\
#         && xx-apt-get purge --autoremove -y curl \\
#         && rm -rf /var/lib/apt/lists/*
#         """
#     return SETUP_DEBIAN_BASE_CUDA_TEMPLATE


# def debian_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
#     """
#     Generate a Dockerfile for the Debian image.
#     """
#     cuda_version = docker_options.cuda_version
#     if cuda_version is not None:
#         cuda: CUDA = make_cuda_cls(docker_options.cuda_version)  # type: ignore
#         if cuda.version.major == "10":
#             distros = "ubuntu1804"
#         else:
#             distros = "ubuntu2004"
#         cuda_debian_setup = _setup_debian_base_cuda(
#             repository=cuda.repository,
#             distros=distros,
#             cuda_major_version=cuda.version.major,
#             cuda_minor_version=cuda.version.minor,
#         )
#     else:
#         cuda_debian_setup = ""
#     return {
#         "setup_uid_gid": SETUP_UID_GID_TEMPLATE,
#         "setup_distro_env": SETUP_DEBIAN_ENV_TEMPLATE,
#         "cleanup": CLEANUP_DEBIAN_TEMPLATE,
#         "setup_cuda": cuda_debian_setup,
#         "install_user_system_packages": _install_debian_system_packages(docker_options),
#     }


# def debian_miniconda_template_context(
#     docker_options: DockerOptions,
# ) -> t.Dict[str, t.Any]:
#     """
#     Generate a Dockerfile for the Debian miniconda image.
#     Refers to https://github.com/mamba-org/micromamba-docker
#     """
#     SETUP_UID_GID_MICROMAMBA_TEMPLATE = """\
# ARG NEW_MAMBA_USER=bentoml
# ARG NEW_MAMBA_USER_ID=1034
# ARG NEW_MAMBA_USER_GID=1034

# RUN bash <<EOF
# usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}"
# groupmod "--new-name=${NEW_MAMBA_USER}" "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}"
# # Update the expected value of MAMBA_USER for the _entrypoint.sh consistency check.
# echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user"
# :
# EOF

# ENV MAMBA_USER=$NEW_MAMBA_USER
#             """

#     INSTALL_PYTHON_WITH_MICROMAMBA_TEMPLATE = """\
# ARG MAMBA_DOCKERFILE_ACTIVATE=1

# RUN --mount=type=cache,mode=0777,target=/root/.cache/pip bash <<EOF
# SAVED_PYTHON_VERSION=$(cat ./env/python/version.txt)
# PYTHON_VERSION=${SAVED_PYTHON_VERSION%.*}

# echo "Installing Python $PYTHON_VERSION with micromamba..."
# micromamba install --yes --name base --channel conda-forge pkgs/main::python=$PYTHON_VERSION pip && micromamba clean --all --yes

# if [ -f ./env/conda/environment.yml ]; then
#   # set pip_interop_enabled to improve conda-pip interoperability. Conda can use
#   # pip-installed packages to satisfy dependencies.
#   echo "Updating conda base environment with environment.yml"
#   micromamba config --set pip_interop_enabled True || true
#   micromamba env update -n base -f ./env/conda/environment.yml
#   micromamba clean --all
# fi
# EOF
#             """

#     cuda_version = docker_options.cuda_version
#     if cuda_version is not None:
#         cuda: CUDA = make_cuda_cls(docker_options.cuda_version)  # type: ignore
#         if cuda.version.major == "10":
#             distros = "ubuntu1804"
#         else:
#             distros = "ubuntu2004"
#         cuda_debian_setup = _setup_debian_base_cuda(
#             repository=cuda.repository,
#             distros=distros,
#             cuda_major_version=cuda.version.major,
#             cuda_minor_version=cuda.version.minor,
#         )
#     else:
#         cuda_debian_setup = ""
#     return {
#         "setup_uid_gid": SETUP_UID_GID_MICROMAMBA_TEMPLATE,
#         "setup_distro_env": SETUP_DEBIAN_ENV_TEMPLATE,
#         "install_python_with_conda": INSTALL_PYTHON_WITH_MICROMAMBA_TEMPLATE,
#         "cleanup": CLEANUP_DEBIAN_TEMPLATE,
#         "setup_cuda": cuda_debian_setup,
#         "install_user_system_packages": _install_debian_system_packages(docker_options),
#         "setup_entrypoint": SETUP_ENTRYPOINT_TEMPLATE,
#     }


# SETUP_RHEL_ENV_TEMPLATE = """\
# RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/yum \\
#     yum upgrade -y \\
#     && yum install -y ca-certificates curl gcc gcc-c++ make \\
#     && yum clean all
#         """

# CLEANUP_RHEL_TEMPLATE = """\
# RUN yum clean all && rm -rf /var/cache/yum
#         """


# def _setup_rhel_base_cuda(
#     repository: str, distros: str, cuda_major_version: str, cuda_minor_version: str
# ):
#     SETUP_RHEL_BASE_CUDA_TEMPLATE = f"""\
# RUN NVIDIA_GPGKEY_SUM=d0664fbbdb8c32356d45de36c5984617217b2d0bef41b93ccecd326ba3b80c87 \\
#     && curl -fsSL {repository}/{distros}/{'${NVARCH}'}/D42D0685.pub | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA \\
#     && echo "$NVIDIA_GPGKEY_SUM /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

# # For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
# RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/yum \\
#         yum upgrade -y && yum install -y \\
#         {'${NVIDIA_CUDA_CUDART_PACKAGE}'} \\
#         {'${NVIDIA_CUDA_COMPAT_PACKAGE}'} \\
#         && ln -s cuda-{cuda_major_version}.{cuda_minor_version} /usr/local/cuda \\
#         && yum remove -y curl \\
#         && yum clean all \\
#         && rm -rf /var/cache/yum
#         """
#     return SETUP_RHEL_BASE_CUDA_TEMPLATE


# def _install_rhel_system_packages(docker_options: DockerOptions) -> str:
#     if docker_options.system_packages is not None:
#         return f"""\
# RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/yum \\
#     yum install -y \\
#     {" ".join(docker_options.system_packages)} \\
#     && yum clean all \\
#     && rm -rf /var/cache/yum
#             """
#     return ""


# def _setup_amazonlinux_env_template(docker_options: DockerOptions) -> str:
#     SETUP_AMAZONLINUX_ENV_TEMPLATE = f"""\
# RUN --mount=type=cache,from=cached,sharing=shared,target=/var/cache/yum \\
#     yum install -y amazon-linux-extras \\
#     && amazon-linux-extras enable python{docker_options.python_version} \\
#     && yum install python{docker_options.python_version}

# {SETUP_RHEL_ENV_TEMPLATE}
#                 """
#     return SETUP_AMAZONLINUX_ENV_TEMPLATE


# def amazonlinux_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
#     """
#     Generate a Dockerfile for the Amazon Linux image.
#     """
#     cuda_version = docker_options.cuda_version
#     if cuda_version is not None:
#         cuda: CUDA = make_cuda_cls(docker_options.cuda_version)  # type: ignore
#         if docker_options.distro in ["rhel7", "ubi7"]:
#             distros = "rhel7"
#         else:
#             distros = "rhel8"
#         cuda_rhel_setup = _setup_rhel_base_cuda(
#             repository=cuda.repository,
#             distros=distros,
#             cuda_major_version=cuda.version.major,
#             cuda_minor_version=cuda.version.minor,
#         )
#     else:
#         cuda_rhel_setup = ""
#     return {
#         "setup_uid_gid": SETUP_UID_GID_TEMPLATE,
#         "setup_distro_env": _setup_amazonlinux_env_template(docker_options),
#         "cleanup": CLEANUP_RHEL_TEMPLATE,
#         "setup_cuda": cuda_rhel_setup,
#         "install_user_system_packages": _install_rhel_system_packages(docker_options),
#     }


# def ubi8_template_context(docker_options: DockerOptions) -> t.Dict[str, t.Any]:
#     """
#     Generate a Dockerfile for the UBI8 image.
#     """
#     cuda_version = docker_options.cuda_version
#     if cuda_version is not None:
#         cuda: CUDA = make_cuda_cls(docker_options.cuda_version)  # type: ignore
#         if docker_options.distro in ["rhel7", "ubi7"]:
#             distros = "rhel7"
#         else:
#             distros = "rhel8"
#         cuda_rhel_setup = _setup_rhel_base_cuda(
#             repository=cuda.repository,
#             distros=distros,
#             cuda_major_version=cuda.version.major,
#             cuda_minor_version=cuda.version.minor,
#         )
#     else:
#         cuda_rhel_setup = ""
#     return {
#         "setup_uid_gid": SETUP_UID_GID_TEMPLATE,
#         "setup_distro_env": SETUP_RHEL_ENV_TEMPLATE,
#         "cleanup": CLEANUP_RHEL_TEMPLATE,
#         "setup_cuda": cuda_rhel_setup,
#         "install_user_system_packages": _install_rhel_system_packages(docker_options),
#     }


# template_context = {
#     "base_image": base_image,
#     "release_stage": _release_stage_name,
#     "user_defined_image": docker_options._user_defined_image,  # type: ignore
#     "bentoml_version": clean_bentoml_version(),  # ensure that we don't have a dirty version.
#     "docker_options": bentoml_cattr.unstructure(docker_options),  # type: ignore
#     "distro_spec": bentoml_cattr.unstructure(distro_spec),  # type: ignore
#     "cuda_spec": bentoml_cattr.unstructure(cuda_spec),  # type: ignore
#     "final_release_stage": setup_release_stage_name(
#         docker_options=docker_options,
#         distro_spec=distro_spec,
#         release_stage_name=_release_stage_name,
#     ),
#     **DOCKERFILE_COMPONENTS,
#     **context_mapping[distro](docker_options),
# }

DEFAULT_UID_GID = 1034
DEFAULT_BENTO_USER = "bentoml"
DEFAULT_BENTO_HOME = f"/home/{DEFAULT_BENTO_USER}"
DEFAULT_BENTO_PATH = f"{DEFAULT_BENTO_HOME}/bento"


def get_template_env(docker_options: DockerOptions) -> dict[str, t.Any]:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version
    python_version = docker_options.python_version

    distro_spec = DistroSpec.from_distro(distro, cuda=cuda_version not in (None, ""))

    if docker_options.base_image is None:
        if cuda_version not in ("", None):
            base_image = distro_spec.image.format(cuda_version=cuda_version)
        else:
            if distro in ["ubi8"]:
                python_version = python_version.replace(".", "")
            else:
                python_version = python_version
        base_image = distro_spec.image.format(python_version=python_version)
    else:
        base_image = docker_options.base_image
        logger.warning(f"Make sure to have Python installed for {base_image}.")

    return {
        "base_image": base_image,
        "bentoml_version": clean_bentoml_version(),
        "default_uid_gid": DEFAULT_UID_GID,
        "default_bento_user": DEFAULT_BENTO_USER,
        "default_bento_path": DEFAULT_BENTO_PATH,
        "default_bento_home": DEFAULT_BENTO_HOME,
        "user_defined_image": docker_options.base_image is not None,
        "docker_options": bentoml_cattr.unstructure(docker_options),  # type: ignore
        "distro_spec": bentoml_cattr.unstructure(distro_spec),  # type: ignore
    }


def generate_dockerfile(docker_options: DockerOptions) -> str:
    distro = docker_options.distro
    cuda_version = docker_options.cuda_version

    distro_spec = DistroSpec.from_distro(distro, cuda=cuda_version not in ("", None))

    template_dir = fs.path.join(fs.path.dirname(__file__), "docker", "templates")
    j2_template = f"{distro_spec.release_type}_{distro}.j2"
    dockerfile_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
        loader=FileSystemLoader(template_dir, followlinks=True),
    )

    bento_dockerfile_tmpl = dockerfile_env.get_template(j2_template)

    if docker_options.dockerfile_template != "":
        template = dockerfile_env.get_template(
            docker_options.dockerfile_template,
            globals={"bento_dockerfile": bento_dockerfile_tmpl},
        )
    else:
        template = bento_dockerfile_tmpl

    return template.render(**get_template_env(docker_options))
